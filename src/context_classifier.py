"""
Content Moderation GAT — Context-Aware Classification (Phase 2)
==============================================================
Mở rộng pipeline Faster R-CNN + GAT bằng một **Safety Head** (phân loại
safe/unsafe per region) và một **Context Explainer** (sinh lời giải thích
human-readable cho người kiểm duyệt dựa trên attention weights α_ij).

Kiến trúc:
    Image
      → Backbone (ResNet-50) → RPN → RoI Align → features [N, 1024]
      → Graph Builder → adj [N, N]
      → GAT (2 layers, 8 heads) → enriched features [N, 1024]
      → Safety Head      → safety_score [N, 1]      (THÊM MỚI ở Phase 2)
      → Context Explainer → reasoning string         (THÊM MỚI ở Phase 2)

Lưu ý trung thực: Safety Head ở đây khởi tạo ngẫu nhiên (CHƯA train) nên
safety_score per-region chỉ mang tính minh hoạ kiến trúc. Phần có ý nghĩa
ngay lập tức là Context Explainer — nó đọc cấu trúc attention/đồ thị để giải
thích "vì sao region bị flag" (vd: knife chú ý nhiều tới person → nghi ngờ
bạo lực). Việc train Safety Head trên dữ liệu nhãn để dành cho Phase 3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from gat_module import GAT, build_knn_adjacency, build_iou_adjacency


# ============================================================
# Taxonomy nội dung vi phạm (khớp chỉ số COCO của TorchVision)
# ============================================================

# Danh sách 91 class của COCO theo TorchVision (index 0 = background).
# Hardcode để module không phụ thuộc torchvision khi chỉ dùng Context Explainer.
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush',
]

PERSON_ID = 1

# Vũ khí: id -> severity cơ bản (0-1). Dao nguy hiểm hơn kéo.
WEAPON_SEVERITY = {
    49: 0.65,   # knife
    87: 0.35,   # scissors
    39: 0.50,   # baseball bat (vừa thể thao vừa có thể là hung khí)
}

# Đồ bếp -> tín hiệu context "nấu ăn" (giảm rủi ro).
KITCHEN_IDS = {
    44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 50: 'spoon',
    51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
    56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
    61: 'cake', 67: 'dining table', 78: 'microwave', 79: 'oven',
    80: 'toaster', 81: 'sink', 82: 'refrigerator',
}

# Đồ thể thao -> tín hiệu context "thể thao" (giảm rủi ro cho baseball bat).
SPORTS_IDS = {34: 'frisbee', 37: 'sports ball', 38: 'kite', 40: 'baseball glove'}


# ============================================================
# Base: Faster R-CNN tích hợp GAT (đưa từ Notebook 2 vào src)
# ============================================================

class FasterRCNNWithGAT(nn.Module):
    """Faster R-CNN tích hợp GAT module.

    Pipeline:
        Image → CNN → RPN → RoI Align → [GAT module] → Classifier + BBox Reg
                                            ↑ thêm context vào đây

    Args:
        num_classes:   số class (mặc định 91 = COCO + background)
        gat_hidden:    số chiều hidden PER HEAD của GAT
        gat_heads:     số attention heads (K)
        knn_k:         số láng giềng khi xây k-NN graph
        build_detector: True = build luôn backbone Faster R-CNN (cần torchvision).
                        False = chỉ dựng phần GAT (tiện unit test, không tải nặng).
    """

    def __init__(self, num_classes=91, gat_hidden=64, gat_heads=8, knn_k=8,
                 build_detector=True):
        super().__init__()

        self.detector = None
        if build_detector:
            # Lazy import để module dùng được Context Explainer mà không cần torchvision
            from torchvision.models.detection import fasterrcnn_resnet50_fpn
            self.detector = fasterrcnn_resnet50_fpn(
                weights=None, weights_backbone=None, num_classes=num_classes)

        # GAT module — input/output dim khớp RoI feature dim (1024 sau two_mlp_head)
        self.roi_dim = 1024
        self.gat = GAT(
            in_features=self.roi_dim,
            hidden_features=gat_hidden,
            out_features=self.roi_dim,  # giữ nguyên dim để plug-in được
            num_heads=gat_heads,
        )
        self.gat_heads = gat_heads
        self.knn_k = knn_k

    def build_adjacency(self, roi_features, boxes=None):
        """Xây adjacency matrix cho N region proposals.

        Mặc định dùng k-NN trong feature space (như Notebook 2). Nếu truyền
        boxes thì có thể đổi sang IoU graph qua build_iou_adjacency.

        Args:
            roi_features: [N, 1024]
            boxes:        [N, 4] hoặc None

        Returns:
            adj: [N, N]
        """
        N = roi_features.shape[0]
        k = max(1, min(self.knn_k, N - 1))
        return build_knn_adjacency(roi_features, k=k)

    def enrich_features_with_gat(self, roi_features, adj):
        """Áp GAT lên RoI features (có residual connection).

        Args:
            roi_features: [N, 1024]
            adj:          [N, N] adjacency matrix

        Returns:
            enriched: [N, 1024] feature đã thu thập context từ láng giềng
        """
        device = roi_features.device
        adj = adj.to(device)
        enriched = self.gat(roi_features, adj)          # [N, 1024]
        return roi_features + enriched                  # residual để ổn định


# ============================================================
# Safety Head — phân loại safe/unsafe per region
# ============================================================

class SafetyHead(nn.Module):
    """FC layers: enriched feature [N, D] → unsafe logit [N, 1].

    sigmoid(logit) = xác suất region "unsafe". Binary head (safe/unsafe).

    Args:
        in_features: số chiều feature đầu vào (= roi_dim, 1024)
        hidden:      số neuron lớp ẩn
        dropout:     dropout giữa các FC layer
    """

    def __init__(self, in_features=1024, hidden=256, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 1),                  # 1 logit (unsafe)
        )

    def forward(self, x):
        """
        Args:
            x: [N, in_features]

        Returns:
            logits: [N, 1] unsafe logit (chưa qua sigmoid)
        """
        return self.net(x)


# ============================================================
# Context Explainer — đọc α_ij để giải thích vì sao bị flag
# ============================================================

def gat_attention_matrix(gat, features, adj):
    """Trích attention weights α_ij từ layer 1 của GAT (trung bình trên các head).

    Args:
        gat:      instance của GAT (gat_module.GAT)
        features: [N, D] node features đưa vào GAT
        adj:      [N, N] adjacency matrix

    Returns:
        alpha: [N, N] attention weights, mỗi hàng tổng = 1 trên láng giềng
    """
    device = features.device
    adj = adj.to(device)
    alphas = []
    with torch.no_grad():
        for head in gat.layer1.heads:                   # mỗi head là 1 GATLayer
            Wh = torch.mm(features, head.W)             # [N, out_per_head]
            e = head._compute_attention_scores(Wh)      # [N, N]
            masked = torch.where(adj > 0, e, -9e15 * torch.ones_like(e))
            alphas.append(F.softmax(masked, dim=1))     # [N, N]
    return torch.stack(alphas).mean(dim=0)              # [N, N] avg over heads


def get_context_explanation(attention_matrix, labels, scores=None,
                            adj_matrix=None, class_names=None,
                            top_k=3, attn_eps=1e-3):
    """Nhìn vào α_ij để giải thích tại sao mỗi region vũ khí bị flag.

    Với mỗi region thuộc nhóm vũ khí (knife/scissors/bat), hàm xem region đó
    "chú ý" (attention cao) tới những láng giềng nào, từ đó suy ra context và
    đưa ra verdict + câu giải thích cho người kiểm duyệt.

    Args:
        attention_matrix: [N, N] α_ij (vd từ gat_attention_matrix)
        labels:           [N] COCO class id của mỗi region
        scores:           [N] confidence detector (tuỳ chọn, để hiển thị)
        adj_matrix:       [N, N] tuỳ chọn — nếu có, chỉ xét láng giềng thật
                          (mask attention bằng adjacency, tránh nhiễu trên
                          node cô lập có softmax đều)
        class_names:      list tên class theo index (mặc định COCO 91 names)
        top_k:            số láng giềng chú ý nhất để liệt kê
        attn_eps:         ngưỡng coi là "có chú ý"

    Returns:
        explanations (list[dict]): mỗi phần tử cho 1 region vũ khí gồm
            index, label, verdict, attends_to [(name, alpha), ...], explanation
    """
    if class_names is None:
        class_names = COCO_INSTANCE_CATEGORY_NAMES

    # đưa về numpy/tensor thuần để indexing thống nhất
    if torch.is_tensor(attention_matrix):
        alpha = attention_matrix.detach().cpu().clone()
    else:
        alpha = torch.as_tensor(attention_matrix, dtype=torch.float32)

    labels = [int(l) for l in labels]
    N = len(labels)

    if adj_matrix is not None:
        adj = (adj_matrix.detach().cpu() if torch.is_tensor(adj_matrix)
               else torch.as_tensor(adj_matrix, dtype=torch.float32))
        alpha = alpha * (adj > 0).float()               # chỉ giữ láng giềng thật

    explanations = []
    for i in range(N):
        if labels[i] not in WEAPON_SEVERITY:
            continue                                    # chỉ giải thích region vũ khí

        weapon_name = class_names[labels[i]]
        row = alpha[i].clone()
        row[i] = 0.0                                    # bỏ self-attention

        # top-k láng giềng được chú ý nhất
        attends_to = []
        if row.numel() > 0 and float(row.max()) > attn_eps:
            k = min(top_k, int((row > attn_eps).sum().item()))
            top_vals, top_idx = torch.topk(row, k=max(1, k))
            for v, j in zip(top_vals.tolist(), top_idx.tolist()):
                if v > attn_eps:
                    attends_to.append((class_names[labels[j]], round(float(v), 3)))

        # đếm context trong các láng giềng được chú ý
        attended_labels = [labels[j] for j in range(N)
                           if j != i and float(row[j]) > attn_eps]
        n_person = sum(1 for l in attended_labels if l == PERSON_ID)
        n_kitchen = sum(1 for l in attended_labels if l in KITCHEN_IDS)
        n_sports = sum(1 for l in attended_labels if l in SPORTS_IDS)

        # suy ra verdict từ context (giống logic Notebook 3, nhưng dẫn dắt bởi attention)
        if n_sports > 0 and labels[i] == 39:
            verdict = 'SAFE'
            why = (f"'{weapon_name}' chú ý chủ yếu tới đồ thể thao "
                   f"({n_sports} vật) → bối cảnh thể thao")
        elif n_kitchen > 0 and n_person <= 1:
            verdict = 'SAFE'
            why = (f"'{weapon_name}' chú ý chủ yếu tới đồ bếp "
                   f"({n_kitchen} vật) → bối cảnh nấu ăn")
        elif n_person >= 2:
            verdict = 'FLAG'
            why = (f"'{weapon_name}' chú ý mạnh tới {n_person} người "
                   f"(đám đông) → nghi ngờ bạo lực")
        elif n_person == 1:
            verdict = 'REVIEW'
            why = (f"'{weapon_name}' chú ý tới 1 người, không có context "
                   f"an toàn → cần xem xét")
        else:
            verdict = 'REVIEW'
            why = (f"'{weapon_name}' không có context rõ ràng từ láng giềng "
                   f"→ cần xem xét")

        focus = ", ".join(f"{name} (α={a})" for name, a in attends_to) or "—"
        conf = f", confidence={float(scores[i]):.2f}" if scores is not None else ""
        explanation = (f"Region {i} ['{weapon_name}'{conf}] {verdict}: {why}. "
                       f"Chú ý nhiều nhất tới: {focus}.")

        explanations.append({
            'index': i,
            'label': weapon_name,
            'verdict': verdict,
            'attends_to': attends_to,
            'explanation': explanation,
        })

    return explanations


# ============================================================
# ContentModerationGAT — kết hợp tất cả
# ============================================================

class ContentModerationGAT(FasterRCNNWithGAT):
    """Faster R-CNN + GAT + Safety Head + Context Explainer (Phase 2).

    Mở rộng FasterRCNNWithGAT bằng Safety Head (safe/unsafe per region) và
    tích hợp Context Explainer để sinh reasoning từ attention weights.

    Args: xem FasterRCNNWithGAT, thêm:
        safety_hidden: số neuron lớp ẩn của Safety Head
    """

    def __init__(self, num_classes=91, gat_hidden=64, gat_heads=8, knn_k=8,
                 safety_hidden=256, build_detector=True,
                 class_names=None):
        super().__init__(num_classes=num_classes, gat_hidden=gat_hidden,
                         gat_heads=gat_heads, knn_k=knn_k,
                         build_detector=build_detector)
        self.safety_head = SafetyHead(self.roi_dim, hidden=safety_hidden)
        self.class_names = class_names or COCO_INSTANCE_CATEGORY_NAMES

    def forward(self, roi_features, boxes=None, adj=None):
        """Chạy GAT enrichment + Safety Head.

        Args:
            roi_features: [N, 1024] feature của N region proposals
            boxes:        [N, 4] (tuỳ chọn, dùng nếu xây IoU graph)
            adj:          [N, N] (tuỳ chọn, nếu None sẽ tự xây k-NN)

        Returns:
            enriched:        [N, 1024]
            unsafe_prob:     [N]  xác suất unsafe per region (sigmoid)
            adj:             [N, N] adjacency đã dùng
        """
        if adj is None:
            adj = self.build_adjacency(roi_features, boxes)
        enriched = self.enrich_features_with_gat(roi_features, adj)   # [N, 1024]
        logits = self.safety_head(enriched)                          # [N, 1]
        unsafe_prob = torch.sigmoid(logits).squeeze(-1)              # [N]
        return enriched, unsafe_prob, adj

    def analyze(self, roi_features, labels, boxes=None, scores=None, adj=None):
        """Pipeline đầy đủ: enrich → safety score → attention → explanation.

        Args:
            roi_features: [N, 1024]
            labels:       [N] COCO class id
            boxes:        [N, 4] (tuỳ chọn)
            scores:       [N] confidence (tuỳ chọn)
            adj:          [N, N] (tuỳ chọn)

        Returns:
            result (dict):
                unsafe_prob:  [N] xác suất unsafe per region (Safety Head, CHƯA train)
                attention:    [N, N] α_ij từ GAT
                adj:          [N, N]
                explanations: list[dict] từ get_context_explanation
                decision:     'FLAG' | 'REVIEW' | 'SAFE' (tổng hợp từ explanations)
        """
        enriched, unsafe_prob, adj = self.forward(roi_features, boxes, adj)
        attention = gat_attention_matrix(self.gat, roi_features, adj)  # [N, N]
        explanations = get_context_explanation(
            attention, labels, scores=scores, adj_matrix=adj,
            class_names=self.class_names)

        verdicts = [e['verdict'] for e in explanations]
        if 'FLAG' in verdicts:
            decision = 'FLAG'
        elif 'REVIEW' in verdicts:
            decision = 'REVIEW'
        else:
            decision = 'SAFE'

        return {
            'unsafe_prob': unsafe_prob,
            'attention': attention,
            'adj': adj,
            'explanations': explanations,
            'decision': decision,
        }


# ============================================================
# Test khi chạy trực tiếp file này
# ============================================================
if __name__ == "__main__":
    import numpy as np

    print("Testing ContentModerationGAT (Phase 2)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # build_detector=False để test nhanh (không tải backbone nặng)
    model = ContentModerationGAT(build_detector=False).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    safety_params = sum(p.numel() for p in model.safety_head.parameters())
    print(f"Total params (GAT + SafetyHead): {n_params:,}")
    print(f"  SafetyHead: {safety_params:,}")

    # Giả lập 6 region proposals: knife + person + person + person + bowl + fork
    torch.manual_seed(0)
    labels = [49, 1, 1, 1, 51, 48]      # knife, person×3, bowl, fork
    N = len(labels)
    roi_features = torch.randn(N, 1024, device=device)
    scores = torch.tensor([0.91, 0.95, 0.93, 0.90, 0.85, 0.80])

    out = model.analyze(roi_features, labels, scores=scores)
    print(f"\nShapes:")
    print(f"  unsafe_prob: {tuple(out['unsafe_prob'].shape)}")
    print(f"  attention:   {tuple(out['attention'].shape)}")
    print(f"  adj:         {tuple(out['adj'].shape)}")
    assert out['unsafe_prob'].shape == (N,)
    assert out['attention'].shape == (N, N)

    print(f"\nDecision tổng thể: {out['decision']}")
    print("Explanations:")
    for e in out['explanations']:
        print(f"  - {e['explanation']}")

    # --- Test get_context_explanation độc lập với attention "đặt tay" ---
    # Cảnh A: knife (node 0) chú ý mạnh tới 2 person → mong đợi FLAG
    print("\n--- Unit test get_context_explanation (attention thủ công) ---")
    labels_a = [49, 1, 1]
    attn_a = torch.tensor([[0.0, 0.5, 0.5],
                           [0.5, 0.0, 0.5],
                           [0.5, 0.5, 0.0]])
    exp_a = get_context_explanation(attn_a, labels_a)
    print("Scene A (knife + 2 person):", exp_a[0]['verdict'])
    assert exp_a[0]['verdict'] == 'FLAG', exp_a

    # Cảnh B: knife (node 0) chú ý mạnh tới bowl + fork → mong đợi SAFE
    labels_b = [49, 51, 48]
    attn_b = torch.tensor([[0.0, 0.5, 0.5],
                           [0.5, 0.0, 0.5],
                           [0.5, 0.5, 0.0]])
    exp_b = get_context_explanation(attn_b, labels_b)
    print("Scene B (knife + bowl + fork):", exp_b[0]['verdict'])
    assert exp_b[0]['verdict'] == 'SAFE', exp_b

    print("\n✅ All tests passed!")
