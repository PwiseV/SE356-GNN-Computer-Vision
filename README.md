# Object Detection with Graph Attention Network

Đồ án môn **Mạng Xã Hội** — UIT (Bài 3: GNN cho Computer Vision)

---

## 📁 Cấu trúc project

```
project/
├── README.md                              ← file này
├── requirements.txt                       ← danh sách thư viện cần cài
├── notebooks/
│   ├── 01_demo_pretrained_detection.ipynb ← DEMO chạy được (mở đầu tiên)
│   └── 02_gat_module_implementation.ipynb ← Implement GAT + tích hợp
├── src/
│   └── gat_module.py                      ← GAT layer implement from scratch
└── outputs/                               ← thư mục lưu ảnh kết quả
```

---

## 🚀 HƯỚNG DẪN CÀI ĐẶT (Windows - ASUS TUF F15)

### Bước 1: Kiểm tra Python

Mở **PowerShell** (nhấn Windows + X → chọn "Windows PowerShell") và gõ:

```bash
python --version
```

Nếu thấy `Python 3.10.x` hoặc `3.11.x` → OK. Nếu chưa có hoặc < 3.10 → tải tại https://python.org

### Bước 2: Tạo môi trường ảo (khuyến nghị)

```bash
cd Desktop
python -m venv venv_gat
.\venv_gat\Scripts\activate
```

Khi thấy `(venv_gat)` ở đầu dòng lệnh → đã vào môi trường ảo.

### Bước 3: Cài thư viện

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install jupyter matplotlib pillow numpy
```

> **Giải thích:** dòng đầu cài PyTorch + CUDA 12.1 để dùng GPU NVIDIA. Mất ~2-3GB và 5-10 phút.

### Bước 4: Kiểm tra GPU

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

Output mong đợi:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 3050   (hoặc tên GPU của cậu)
```

Nếu báo `CUDA available: False` → vẫn chạy được trên CPU (chậm hơn ~5x nhưng vẫn ổn cho inference demo).

### Bước 5: Mở Jupyter Notebook

```bash
jupyter notebook
```

Trình duyệt sẽ tự mở. Vào folder `notebooks/` → click **`01_demo_pretrained_detection.ipynb`**.

---

## 📒 Cách chạy notebook

1. **Chạy từng cell** từ trên xuống bằng phím `Shift + Enter`
2. **Cell đầu tiên** sẽ tải pretrained model lần đầu (~160MB, mất 1-2 phút) — chỉ tải 1 lần
3. **Cell test images** sẽ tải 4 ảnh sample từ COCO val2017
4. **Cell visualize** sẽ hiện ảnh có bbox + class + confidence

### Thứ tự khuyên dùng:
1. `01_demo_pretrained_detection.ipynb` — chạy trước để verify môi trường OK + có kết quả thật
2. `02_gat_module_implementation.ipynb` — sau khi notebook 1 chạy ổn

---

## ❓ TROUBLESHOOTING

### "ModuleNotFoundError: No module named 'torch'"
→ Chưa activate venv. Chạy lại: `.\venv_gat\Scripts\activate`

### "CUDA out of memory"
→ Đóng các app khác (Chrome, game). GPU 4GB chỉ chạy được 1 ảnh/batch.

### "URLError when downloading model"
→ Kiểm tra mạng. Hoặc tải thủ công từ:
- https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
- Copy file `.pth` vào `C:\Users\<username>\.cache\torch\hub\checkpoints\`

### Notebook không hiện ảnh
→ Thêm `%matplotlib inline` vào cell đầu tiên.

---

## 🎤 ĐOẠN PHÁT BIỂU KHI DEMO LIVE

Nếu thầy yêu cầu demo trực tiếp:

> "Dạ nhóm em xin phép demo Notebook 1 — chạy Faster R-CNN baseline. Đây là model đã được train trên COCO 118K ảnh, đạt mAP 37 đúng như paper gốc.
>
> *(Mở notebook 01, chạy đến cell visualize)*
>
> Đây là kết quả phát hiện trên ảnh từ COCO val2017. Model phát hiện được [X] đối tượng với confidence > 0.7.
>
> Trong Notebook 2, nhóm em implement GAT module từ scratch theo paper Veličković 2018, và tích hợp vào pipeline này. Phần training trên full dataset nhóm em đang chạy, dự kiến hoàn thiện cho báo cáo cuối kỳ."

---

## 📚 Tài liệu tham khảo

- [GAT Paper (Veličković et al., 2018)](https://arxiv.org/abs/1710.10903)
- [Faster R-CNN Paper (Ren et al., 2015)](https://arxiv.org/abs/1506.01497)
- [COCO Dataset](https://cocodataset.org/)
- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)
- [TorchVision Models](https://pytorch.org/vision/stable/models.html)
