"""
COCO mAP Evaluation (Phase 3)
=============================
Đánh giá detection model bằng COCO mean Average Precision (mAP) qua pycocotools,
gọi sau mỗi epoch trong training loop (xem trainer.py).

Pipeline:
  model(images) → predictions [{boxes,labels,scores}]
    → chuyển sang định dạng COCO results [{image_id, category_id, bbox, score}]
    → COCOeval(coco_gt, coco_dt, 'bbox') → summarize → mAP

Lưu ý: pycocotools được import lazy bên trong hàm để module vẫn import được
trên môi trường chưa cài (vd để dùng riêng hàm convert).
"""

import torch


def predictions_to_coco_results(outputs, targets, score_threshold=0.0):
    """Chuyển output model + image_id → list kết quả định dạng COCO.

    COCO results yêu cầu bbox dạng [x, y, w, h] (Issue 4: ngược với detector
    cho ra [x1, y1, x2, y2]).

    Args:
        outputs: list[dict] mỗi ảnh có 'boxes' [N,4] (x1y1x2y2), 'labels', 'scores'
        targets: list[dict] tương ứng, mỗi dict có 'image_id'
        score_threshold: bỏ prediction có score thấp hơn ngưỡng

    Returns:
        results (list[dict]): {image_id, category_id, bbox [x,y,w,h], score}
    """
    results = []
    for output, target in zip(outputs, targets):
        image_id = int(target['image_id'].item()
                       if torch.is_tensor(target['image_id'])
                       else target['image_id'])
        boxes = output['boxes'].detach().cpu()      # [N, 4] x1y1x2y2
        labels = output['labels'].detach().cpu().tolist()
        scores = output['scores'].detach().cpu().tolist()

        for box, label, score in zip(boxes, labels, scores):
            if score < score_threshold:
                continue
            x1, y1, x2, y2 = box.tolist()
            results.append({
                'image_id': image_id,
                'category_id': int(label),
                'bbox': [x1, y1, x2 - x1, y2 - y1],   # → [x, y, w, h]
                'score': float(score),
            })
    return results


@torch.no_grad()
def evaluate_coco(model, data_loader, device, coco_gt=None, max_images=None):
    """Chạy inference trên val set + tính COCO mAP bằng pycocotools.

    Args:
        model:       detection model (eval mode trả list predictions)
        data_loader: DataLoader validation (collate_fn detection)
        device:      cpu/cuda
        coco_gt:     COCO ground truth object. Nếu None lấy từ
                     data_loader.dataset.coco
        max_images:  giới hạn số ảnh để eval nhanh (None = toàn bộ)

    Returns:
        stats (dict): mAP, mAP_50, mAP_75, mAP_small/medium/large, stats[12]
    """
    from pycocotools.cocoeval import COCOeval

    if coco_gt is None:
        coco_gt = data_loader.dataset.coco

    model.eval()
    results, seen = [], 0
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        results.extend(predictions_to_coco_results(outputs, targets))
        seen += len(images)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        if max_images is not None and seen >= max_images:
            break

    if len(results) == 0:
        print("  [eval] Cảnh báo: model không sinh prediction nào → mAP = 0")
        return {'mAP': 0.0, 'mAP_50': 0.0, 'stats': [0.0] * 12}

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    if max_images is not None:
        # chỉ eval trên các ảnh đã chạy
        coco_eval.params.imgIds = sorted({r['image_id'] for r in results})
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    s = coco_eval.stats  # 12 số liệu COCO chuẩn
    return {
        'mAP': float(s[0]),       # AP @ IoU=0.50:0.95
        'mAP_50': float(s[1]),    # AP @ IoU=0.50
        'mAP_75': float(s[2]),    # AP @ IoU=0.75
        'mAP_small': float(s[3]),
        'mAP_medium': float(s[4]),
        'mAP_large': float(s[5]),
        'stats': [float(x) for x in s],
    }


# ============================================================
# Smoke test khi chạy trực tiếp (không cần pycocotools/COCO)
# ============================================================
if __name__ == "__main__":
    print("Smoke test evaluator.py (convert results, không cần pycocotools)...")

    # giả lập output 1 ảnh: 2 box
    outputs = [{
        'boxes': torch.tensor([[10., 20., 110., 220.],   # x1y1x2y2
                               [50., 60., 90., 100.]]),
        'labels': torch.tensor([49, 1]),
        'scores': torch.tensor([0.91, 0.40]),
    }]
    targets = [{'image_id': torch.tensor([42])}]

    res = predictions_to_coco_results(outputs, targets, score_threshold=0.5)
    print("Số result sau ngưỡng 0.5:", len(res))
    print("Result[0]:", res[0])

    # box đầu: [x,y,w,h] = [10,20,100,200]
    assert len(res) == 1                      # chỉ box score>=0.5 được giữ
    assert res[0]['bbox'] == [10.0, 20.0, 100.0, 200.0]
    assert res[0]['category_id'] == 49 and res[0]['image_id'] == 42

    print("\n✅ Evaluator convert test passed!")
    print("(mAP đầy đủ cần pycocotools + COCO val set — chạy trong notebook 04)")
