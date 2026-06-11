"""
Graph Attention Network (GAT) Module
=====================================
Implementation từ scratch của GAT layer dựa trên paper:
"Graph Attention Networks" - Veličković et al., ICLR 2018
https://arxiv.org/abs/1710.10903

Module này được tích hợp vào pipeline Faster R-CNN để cải thiện
object detection bằng cách mô hình hoá quan hệ giữa các region proposals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """Một lớp GAT đơn (single attention head).
    
    Công thức:
        e_ij = LeakyReLU(a^T · [W·h_i ‖ W·h_j])    # attention score
        α_ij = softmax_j(e_ij)                       # normalize
        h_i' = σ(Σ_j α_ij · W·h_j)                   # aggregate
    
    Args:
        in_features:  số chiều input của mỗi node
        out_features: số chiều output của mỗi node  
        dropout:      dropout rate cho attention coefficients
        alpha:        negative slope của LeakyReLU (mặc định 0.2 theo paper)
    """
    
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        # W: ma trận trọng số tuyến tính (shared)
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # a: vector attention học được (kích thước 2 * out_features vì concat)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, h, adj):
        """
        Args:
            h:   [N, in_features]   node features
            adj: [N, N]             adjacency matrix (1 nếu có cạnh, 0 nếu không)
        
        Returns:
            h_prime: [N, out_features]   node features mới
        """
        # Bước 1: Linear transformation - Wh shape [N, out_features]
        Wh = torch.mm(h, self.W)
        
        # Bước 2: Tính attention scores cho mọi cặp (i, j)
        # Vector hoá: e_ij = LeakyReLU(a^T · [Wh_i ‖ Wh_j])
        e = self._compute_attention_scores(Wh)  # [N, N]
        
        # Bước 3: Mask out các cặp không có cạnh trong adjacency matrix
        # Đặt -inf để softmax cho ra 0
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        # Bước 4: Chuẩn hoá bằng softmax theo từng row (mỗi node i)
        attention = F.softmax(attention, dim=1)  # [N, N]
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # Bước 5: Aggregate - h_i' = Σ_j α_ij · Wh_j
        h_prime = torch.matmul(attention, Wh)  # [N, out_features]
        
        return h_prime
    
    def _compute_attention_scores(self, Wh):
        """Tính e_ij = LeakyReLU(a^T · [Wh_i ‖ Wh_j]) cho mọi cặp (i, j).
        
        Trick để vector hoá: tách a thành 2 phần a1, a2 (mỗi phần out_features)
        Khi đó a^T · [Wh_i ‖ Wh_j] = a1^T · Wh_i + a2^T · Wh_j
        Tính riêng từng phần rồi broadcast cộng → matrix [N, N]
        """
        # Tách a thành 2 phần
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])  # [N, 1]
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])  # [N, 1]
        
        # Broadcast: Wh1 shape [N, 1] và Wh2.T shape [1, N] → e shape [N, N]
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)


class MultiHeadGATLayer(nn.Module):
    """Multi-head GAT layer.
    
    Chạy K attention heads song song, sau đó concatenate (hidden layer)
    hoặc average (output layer).
    
    Args:
        in_features:  số chiều input
        out_features: số chiều output PER HEAD
        num_heads:    số attention heads (K)
        concat:       True = concat (output = K * out_features)
                      False = average (output = out_features), dùng cho output layer
    """
    
    def __init__(self, in_features, out_features, num_heads=8, 
                 dropout=0.6, alpha=0.2, concat=True):
        super().__init__()
        self.num_heads = num_heads
        self.concat = concat
        
        # K attention heads độc lập
        self.heads = nn.ModuleList([
            GATLayer(in_features, out_features, dropout, alpha)
            for _ in range(num_heads)
        ])
    
    def forward(self, h, adj):
        """
        Args:
            h:   [N, in_features]
            adj: [N, N]
        
        Returns:
            Output:
                - Nếu concat=True: [N, num_heads * out_features]
                - Nếu concat=False: [N, out_features]
        """
        head_outputs = [head(h, adj) for head in self.heads]
        
        if self.concat:
            # Hidden layer: concat các head
            return torch.cat(head_outputs, dim=1)
        else:
            # Output layer: average các head
            return torch.mean(torch.stack(head_outputs), dim=0)


class GAT(nn.Module):
    """Full GAT network với 2 layer (theo paper gốc).
    
    Layer 1: Multi-head với concat → ELU activation
    Layer 2: Multi-head với average (output layer)
    
    Args:
        in_features:  số chiều input của mỗi node
        hidden_features: số chiều hidden PER HEAD
        out_features: số chiều output cuối cùng
        num_heads:    số attention heads ở layer 1
        num_heads_out: số attention heads ở output layer (thường = 1)
        dropout:      dropout rate
    """
    
    def __init__(self, in_features, hidden_features, out_features,
                 num_heads=8, num_heads_out=1, dropout=0.6, alpha=0.2):
        super().__init__()
        self.dropout = dropout
        
        # Layer 1: multi-head với concat
        self.layer1 = MultiHeadGATLayer(
            in_features, hidden_features, num_heads,
            dropout=dropout, alpha=alpha, concat=True
        )
        
        # Layer 2: output layer với averaging
        self.layer2 = MultiHeadGATLayer(
            hidden_features * num_heads, out_features, num_heads_out,
            dropout=dropout, alpha=alpha, concat=False
        )
    
    def forward(self, h, adj):
        """
        Args:
            h:   [N, in_features]
            adj: [N, N]
        
        Returns:
            output: [N, out_features]
        """
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.layer1(h, adj)
        h = F.elu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.layer2(h, adj)
        return h


# ============================================================
# Utility: Build adjacency matrix từ region proposals
# ============================================================

def build_knn_adjacency(node_features, k=8):
    """Xây adjacency matrix bằng k-Nearest Neighbors trong feature space.
    
    Args:
        node_features: [N, D] feature vector của N nodes
        k:             số láng giềng gần nhất
    
    Returns:
        adj: [N, N] adjacency matrix (binary, không có self-loop)
    """
    N = node_features.shape[0]
    
    # Tính ma trận khoảng cách Euclidean giữa mọi cặp
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    norm_sq = (node_features ** 2).sum(dim=1, keepdim=True)
    dist_sq = norm_sq + norm_sq.T - 2 * node_features @ node_features.T
    
    # Lấy k láng giềng gần nhất (bỏ qua chính nó - index 0 là self)
    _, knn_idx = dist_sq.topk(k=k+1, largest=False, dim=1)
    knn_idx = knn_idx[:, 1:]  # bỏ self
    
    # Xây adjacency matrix
    adj = torch.zeros(N, N, device=node_features.device)
    rows = torch.arange(N).unsqueeze(1).expand(-1, k)
    adj[rows.flatten(), knn_idx.flatten()] = 1.0
    
    # Symmetric: nếu i là láng giềng của j thì j cũng là láng giềng của i
    adj = torch.max(adj, adj.T)
    
    return adj


def build_iou_adjacency(boxes, threshold=0.1):
    """Xây adjacency matrix bằng IoU giữa các bounding box.
    
    Args:
        boxes:     [N, 4] (x1, y1, x2, y2)
        threshold: 2 box có IoU > threshold sẽ được nối cạnh
    
    Returns:
        adj: [N, N] adjacency matrix
    """
    N = boxes.shape[0]
    
    # Tính diện tích mỗi box
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    # Tính intersection cho mọi cặp
    lt = torch.max(boxes[:, None, :2], boxes[None, :, :2])  # [N, N, 2]
    rb = torch.min(boxes[:, None, 2:], boxes[None, :, 2:])  # [N, N, 2]
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]  # [N, N]
    
    # IoU
    union = area[:, None] + area[None, :] - inter
    iou = inter / (union + 1e-6)
    
    # Adjacency
    adj = (iou > threshold).float()
    adj.fill_diagonal_(0)  # bỏ self-loop
    
    return adj


# ============================================================
# Test khi chạy trực tiếp file này
# ============================================================
if __name__ == "__main__":
    print("Testing GAT module...")
    
    # Giả lập 10 region proposals, mỗi region có feature 1024-D
    N, D_in = 10, 1024
    node_features = torch.randn(N, D_in)
    
    # Xây adjacency matrix bằng k-NN
    adj = build_knn_adjacency(node_features, k=4)
    print(f"Adjacency matrix shape: {adj.shape}")
    print(f"Average degree: {adj.sum(dim=1).mean():.2f}")
    
    # Test single GAT layer
    layer = GATLayer(in_features=1024, out_features=256)
    out = layer(node_features, adj)
    print(f"Single GAT layer output: {out.shape}")
    
    # Test multi-head
    mh_layer = MultiHeadGATLayer(in_features=1024, out_features=64, num_heads=8)
    out = mh_layer(node_features, adj)
    print(f"Multi-head (8 heads, concat) output: {out.shape}")  # 8*64 = 512
    
    # Test full GAT
    gat = GAT(in_features=1024, hidden_features=64, out_features=256, num_heads=8)
    out = gat(node_features, adj)
    print(f"Full GAT output: {out.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in gat.parameters())
    print(f"\nTotal GAT parameters: {total_params:,}")
    
    print("\n✅ All tests passed!")
