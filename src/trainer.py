"""
Training utilities (Phase 3)
============================
Tiện ích training cho pipeline Faster R-CNN (+ GAT) trên COCO, thiết kế để
chạy được cả trên RTX 3050 (4GB) lẫn Google Colab T4/A100.

Bao gồm:
  - CocoDetectionDataset: bọc TorchVision CocoDetection → đúng format Faster R-CNN
  - collate_fn, convert_coco_target: chuẩn hoá batch/target
  - build_optimizer (Adam), build_scheduler (MultiStepLR: 1e-4 → 1e-5 @epoch8 → 1e-6 @epoch11)
  - freeze_backbone / unfreeze_backbone: frozen 2 epoch đầu
  - train_one_epoch, save_checkpoint / load_checkpoint
  - train: vòng lặp đầy đủ + checkpoint mỗi epoch + eval mAP mỗi epoch

Convention: mọi tensor chuyển .to(device); model detection ở train mode trả về
dict loss, ở eval mode trả về list predictions.
"""

import os
import time
import torch

# Convention: check device ở đầu module
DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# Data utilities
# ============================================================

def convert_coco_target(anns, image_id):
    """Chuyển annotation COCO (list dict) → dict target cho Faster R-CNN.

    Issue 4 (CONTEXT.md): bbox COCO ở dạng [x, y, w, h] → cần [x1, y1, x2, y2].

    Args:
        anns:     list[dict] annotation của 1 ảnh (mỗi dict có 'bbox', 'category_id')
        image_id: id ảnh (int) — cần cho evaluation

    Returns:
        target (dict): boxes [M,4], labels [M], image_id [1], area [M], iscrowd [M]
    """
    boxes, labels, areas, iscrowd = [], [], [], []
    for obj in anns:
        x, y, w, h = obj['bbox']
        if w <= 0 or h <= 0:
            continue  # bỏ box suy biến (tránh NaN loss)
        boxes.append([x, y, x + w, y + h])      # [x1, y1, x2, y2]
        labels.append(obj['category_id'])
        areas.append(obj.get('area', w * h))
        iscrowd.append(obj.get('iscrowd', 0))

    if boxes:
        boxes = torch.as_tensor(boxes, dtype=torch.float32)     # [M, 4]
        labels = torch.as_tensor(labels, dtype=torch.int64)     # [M]
    else:
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.zeros((0,), dtype=torch.int64)

    return {
        'boxes': boxes,
        'labels': labels,
        'image_id': torch.tensor([image_id]),
        'area': torch.as_tensor(areas, dtype=torch.float32),
        'iscrowd': torch.as_tensor(iscrowd, dtype=torch.int64),
    }


class CocoDetectionDataset(torch.utils.data.Dataset):
    """Bọc torchvision.datasets.CocoDetection → (image_tensor, target_dict).

    Trả image dạng tensor [3, H, W] trong [0,1] và target đúng format Faster
    R-CNN (kèm image_id để evaluator dùng). Thuộc tính `.coco` là COCO ground
    truth object — evaluator cần nó.

    Args:
        img_dir:  thư mục ảnh (vd val2017/)
        ann_file: file annotation JSON (vd annotations/instances_val2017.json)
    """

    def __init__(self, img_dir, ann_file):
        from torchvision.datasets import CocoDetection
        self._ds = CocoDetection(img_dir, ann_file)
        self.coco = self._ds.coco          # GT cho COCOeval
        self.ids = self._ds.ids

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        from torchvision.transforms import functional as TF
        img, anns = self._ds[idx]
        image_id = self.ids[idx]
        target = convert_coco_target(anns, image_id)
        img = TF.to_tensor(img.convert('RGB'))   # [3, H, W] in [0,1]
        return img, target


def collate_fn(batch):
    """Gom batch detection: list[(image, target)] → (tuple images, tuple targets).

    Detection model nhận list ảnh có kích thước khác nhau nên KHÔNG stack được.
    """
    return tuple(zip(*batch))


def make_loader(dataset, batch_size=2, shuffle=True, num_workers=2):
    """Tạo DataLoader với collate_fn phù hợp detection.

    Lưu ý batch size (CONTEXT.md): 2-4 cho RTX 3050 (4GB), 8 cho Colab A100.
    """
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, collate_fn=collate_fn)


# ============================================================
# Optimizer & scheduler
# ============================================================

def build_optimizer(model, lr=1e-4, weight_decay=1e-4):
    """Adam optimizer trên toàn bộ tham số (kể cả backbone đang frozen).

    Giữ tất cả param trong optimizer để khi unfreeze (epoch 3) không phải dựng
    lại optimizer — param frozen có grad None nên tự động không bị update.

    Args:
        model: detection model (hoặc model + GAT)
        lr:    learning rate khởi điểm (1e-4 theo CONTEXT.md)

    Returns:
        torch.optim.Adam
    """
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer, milestones=(8, 11), gamma=0.1):
    """LR schedule: 1e-4 → 1e-5 (@epoch 8) → 1e-6 (@epoch 11).

    MultiStepLR nhân LR với gamma=0.1 tại mỗi milestone (gọi .step() mỗi epoch).
    """
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(milestones), gamma=gamma)


# ============================================================
# Freeze / unfreeze backbone
# ============================================================

def _get_backbone(model):
    """Tìm backbone của detection model (hỗ trợ cả ContentModerationGAT)."""
    if hasattr(model, 'backbone'):
        return model.backbone                       # torchvision FasterRCNN
    if hasattr(model, 'detector') and model.detector is not None:
        return model.detector.backbone              # ContentModerationGAT
    return None


def freeze_backbone(model):
    """Đóng băng backbone (2 epoch đầu) — chỉ train RPN/heads/GAT."""
    bb = _get_backbone(model)
    if bb is None:
        return
    for p in bb.parameters():
        p.requires_grad = False


def unfreeze_backbone(model):
    """Mở băng backbone (từ epoch 3) để fine-tune toàn bộ."""
    bb = _get_backbone(model)
    if bb is None:
        return
    for p in bb.parameters():
        p.requires_grad = True


# ============================================================
# Train / checkpoint
# ============================================================

def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    print_freq=20, warmup=False):
    """Train 1 epoch. Detection model ở train mode trả về dict loss.

    Args:
        model:       detection model (train mode trả dict loss)
        optimizer:   optimizer
        data_loader: DataLoader (collate_fn detection)
        device:      cpu/cuda
        epoch:       chỉ số epoch (để log)
        print_freq:  in log mỗi N iteration
        warmup:      bật linear warmup LR cho epoch đầu (ổn định detection)

    Returns:
        avg_loss (float): loss trung bình trong epoch
    """
    model.train()
    base_lrs = [g['lr'] for g in optimizer.param_groups]
    warmup_iters = min(1000, len(data_loader) - 1) if warmup else 0

    running, n_iter = 0.0, 0
    for it, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # linear warmup LR trong epoch đầu
        if warmup and it < warmup_iters:
            scale = (it + 1) / warmup_iters
            for g, base in zip(optimizer.param_groups, base_lrs):
                g['lr'] = base * scale

        loss_dict = model(images, targets)          # dict các loss thành phần
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running += float(losses)
        n_iter += 1
        if it % print_freq == 0:
            parts = " | ".join(f"{k}={float(v):.3f}" for k, v in loss_dict.items())
            print(f"  [epoch {epoch}] iter {it}/{len(data_loader)} "
                  f"loss={float(losses):.3f} ({parts})")

        if device.type == 'cuda':
            torch.cuda.empty_cache()                # Issue 3: tránh OOM trên 4GB

    # khôi phục LR gốc sau warmup
    if warmup:
        for g, base in zip(optimizer.param_groups, base_lrs):
            g['lr'] = base

    return running / max(1, n_iter)


def save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_dir,
                    extra=None):
    """Lưu checkpoint mỗi epoch.

    Returns:
        path (str): đường dẫn file checkpoint đã lưu
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch:02d}.pth")
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)
    return path


def load_checkpoint(path, model, optimizer=None, scheduler=None,
                    map_location=None):
    """Nạp checkpoint để train tiếp / inference.

    Returns:
        epoch (int): epoch đã lưu trong checkpoint
    """
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer and ckpt.get('optimizer_state_dict'):
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler and ckpt.get('scheduler_state_dict'):
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    return ckpt.get('epoch', 0)


def train(model, train_loader, device=DEFAULT_DEVICE, num_epochs=12, lr=1e-4,
          milestones=(8, 11), gamma=0.1, freeze_backbone_epochs=2,
          checkpoint_dir='checkpoints', evaluate_fn=None, val_loader=None,
          print_freq=20):
    """Vòng lặp training đầy đủ cho Phase 3.

    Mỗi epoch: (un)freeze backbone → train → scheduler.step → checkpoint →
    eval mAP (nếu có evaluate_fn + val_loader).

    Args:
        model:        detection model (+ GAT) — train mode trả dict loss
        train_loader: DataLoader train
        device:       cpu/cuda
        num_epochs:   số epoch (mặc định 12 theo CONTEXT.md)
        lr:           LR khởi điểm (1e-4)
        milestones:   epoch giảm LR (8, 11) → 1e-5, 1e-6
        freeze_backbone_epochs: số epoch đầu đóng băng backbone (2)
        checkpoint_dir: thư mục lưu checkpoint
        evaluate_fn:  hàm eval(model, val_loader, device) -> dict có 'mAP'
        val_loader:   DataLoader validation

    Returns:
        history (dict): {'train_loss': [...], 'mAP': [...]}
    """
    model.to(device)
    optimizer = build_optimizer(model, lr=lr)
    scheduler = build_scheduler(optimizer, milestones=milestones, gamma=gamma)
    history = {'train_loss': [], 'mAP': []}

    for epoch in range(num_epochs):
        # Frozen backbone 2 epoch đầu, unfreeze từ epoch 3 (index 2)
        if epoch < freeze_backbone_epochs:
            freeze_backbone(model)
            bb_state = 'FROZEN'
        else:
            unfreeze_backbone(model)
            bb_state = 'trainable'

        cur_lr = optimizer.param_groups[0]['lr']
        print(f"\n{'='*60}\nEpoch {epoch}/{num_epochs-1} | LR={cur_lr:.1e} | "
              f"backbone={bb_state}\n{'='*60}")

        t0 = time.time()
        avg_loss = train_one_epoch(model, optimizer, train_loader, device,
                                   epoch, print_freq=print_freq,
                                   warmup=(epoch == 0))
        scheduler.step()
        history['train_loss'].append(avg_loss)

        ckpt_path = save_checkpoint(model, optimizer, scheduler, epoch,
                                    checkpoint_dir,
                                    extra={'train_loss': avg_loss})
        print(f"  → avg_loss={avg_loss:.4f} | {time.time()-t0:.0f}s | "
              f"saved {ckpt_path}")

        if evaluate_fn is not None and val_loader is not None:
            stats = evaluate_fn(model, val_loader, device)
            history['mAP'].append(stats.get('mAP', float('nan')))
            print(f"  → mAP={stats.get('mAP', float('nan')):.4f}")

    return history


# ============================================================
# Smoke test khi chạy trực tiếp (không cần tải COCO)
# ============================================================
if __name__ == "__main__":
    print("Smoke test trainer.py (1 ảnh giả lập, không tải COCO)...")
    from torchvision.models.detection import fasterrcnn_resnet50_fpn

    device = DEFAULT_DEVICE
    print(f"Device: {device}")

    # model nhẹ: random init, 91 class
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None,
                                    num_classes=91).to(device)

    # 1 ảnh giả + 2 box hợp lệ
    img = torch.rand(3, 256, 256, device=device)
    target = {
        'boxes': torch.tensor([[10., 10., 100., 120.], [50., 60., 200., 220.]],
                              device=device),
        'labels': torch.tensor([49, 1], device=device),   # knife, person
        'image_id': torch.tensor([0]),
    }

    # test freeze/unfreeze
    freeze_backbone(model)
    n_frozen = sum(1 for p in model.backbone.parameters() if not p.requires_grad)
    print(f"Backbone params frozen: {n_frozen}")
    unfreeze_backbone(model)
    n_train = sum(1 for p in model.backbone.parameters() if p.requires_grad)
    print(f"Backbone params trainable sau unfreeze: {n_train}")

    # test optimizer + scheduler LR schedule
    opt = build_optimizer(model, lr=1e-4)
    sch = build_scheduler(opt, milestones=(8, 11), gamma=0.1)
    lrs = []
    for e in range(13):
        lrs.append(opt.param_groups[0]['lr'])
        sch.step()
    print(f"LR @epoch 0/8/11/12: {lrs[0]:.1e} / {lrs[8]:.1e} / {lrs[11]:.1e} / {lrs[12]:.1e}")
    assert abs(lrs[0] - 1e-4) < 1e-9 and abs(lrs[8] - 1e-5) < 1e-9 and abs(lrs[11] - 1e-6) < 1e-9

    # test 1 training step (loss dict + backward)
    model.train()
    loss_dict = model([img], [target])
    losses = sum(loss_dict.values())
    opt.zero_grad(); losses.backward(); opt.step()
    print(f"1 training step OK | loss={float(losses):.3f} | "
          f"thành phần: {list(loss_dict.keys())}")

    print("\n✅ Trainer smoke test passed!")
