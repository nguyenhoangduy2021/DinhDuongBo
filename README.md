# Ứng dụng máy học trong quản lý dinh dưỡng đàn bò sữa 🐄

Mã nguồn, dữ liệu và mô hình cải tiến cho đề tài phân loại 6 hành vi bò sữa
(eating, lying, lying ruminating, standing, standing ruminating, walking) từ dữ
liệu cảm biến gia tốc + con quay hồi chuyển (ACC/GYRO), hướng đến giám sát
dinh dưỡng và sức nhai lại ở cấp nông hộ.

> Đề tài: **Ứng dụng máy học trong quản lý dinh dưỡng đàn bò sữa**
> Học viên: Trịnh Thị Siếu Búi — Trường Công nghệ Thông tin và Truyền thông, Đại học Cần Thơ
> Hướng dẫn: TS. Trương Minh Thái và PGS.TS. Hồ Thanh Thâm

## Kết quả chính (StratifiedKFold-10 OOF, seed 42, 7.295 cửa sổ)

| Mô hình | Accuracy | macro-F1 | F1 eating | F1 walking |
|---|---|---|---|---|
| RF 500 cây — baseline bài báo gốc | 0.9989 | 0.9973 | 0.9992 | 0.9899 |
| RF + top-60 đặc trưng (RFE-style) | 0.9990 | 0.9975 | 0.9992 | 0.9899 |
| HGB đa lớp (luận văn v2) | 0.9995 | 0.9982 | 0.9984 | 0.9933 |
| **Ensemble OvR — 6 chuyên gia HGB (CHỌN)** | **0.9995** | **0.9983** | **0.9992** | **0.9933** |
| HGB class-weight balanced | 0.9992 | 0.9980 | 0.9992 | 0.9933 |
| Hai tầng + chuyên gia sửa cặp nhầm | 0.9993 | 0.9981 | 0.9984 | 0.9933 |
| Rank-mix OOF (phản ví dụ, không hợp lệ) | 0.8149 | 0.7602 | 0.8376 | 0.3931 |

**Điểm cải tiến so với bài báo gốc** (Pongsanun et al. 2025, RF): macro-F1
0.9973 → **0.9983**, F1 lớp hiếm *walking* 0.9899 → **0.9933**, nhờ (i) chuyển
từ Random Forest sang HistGradientBoosting, (ii) kiến trúc One-versus-Rest với 6
chuyên gia nhị phân giúp mỗi lớp (kể cả lớp hiếm walking 1,8%) có mô hình riêng,
(iii) loại 9 cột nhị phân chỉ báo hành vi gây rò rỉ nhãn.

## Kiểm định vững chắc (bổ sung 10/2026, 5 seeds × StratifiedKFold-10)

| Cấu hình | Accuracy | macro-F1 | F1 walking |
|---|---|---|---|
| RF 500 cây (phương pháp bài gốc) | 0.9987 ± 0.0002 | 0.9969 ± 0.0007 | 0.9878 ± 0.0041 |
| **OvR 6 chuyên gia (CHỌN)** | **0.9993 ± 0.0003** | **0.9978 ± 0.0016** | **0.9905 ± 0.0082** |
| OvR + SMOTE lớp hiếm (ablation, khai báo) | 0.9993 | 0.9992 | 1.0000 |
| OvR + bản sao Gauss lớp hiếm (ablation) | 0.9997 | 0.9987 | 0.9933 |

Wilcoxon một phía theo seed: accuracy p = 0.031; F1 walking nghiêng về OvR ở 3/5 seed
(p = 0.156, khoảng rộng do chỉ 148 cửa sổ walking). Chạy lại:
`python3 src/robustness_checks.py` → `results/robustness_feeding.json`.
Mô hình trong `models/` nhúng sẵn các chỉ số này trong khoá `robustness`.

## Cấu trúc repo

```
data/      cow_data_processed_updated.csv   — dữ liệu gốc (7.295 × 142)
src/       features.py                      — nạp dữ liệu, chọn đặc trưng, chống rò rỉ nhãn
           experiment_baseline_and_improved.py — tái tạo toàn bộ bảng so sánh
           train_save_models.py             — huấn luyện & lưu mô hình chọn
           predict.py                       — demo nạp mô hình + dự đoán
models/    ensemble_dinhduong.joblib        — mô hình đã huấn luyện (6 chuyên gia)
notebooks/ BoKhoe_Colab_SoSanh_DinhDuong.ipynb — chạy 1-click trên Google Colab
results/   compare_feeding.json             — kết quả đầy đủ các phương án
figures/   feed_compare_v5.png, paper2_confusion.png
```

## Chạy nhanh

```bash
pip install -r requirements.txt
python3 src/experiment_baseline_and_improved.py   # tái tạo bảng kết quả (CPU, vài phút)
python3 src/train_save_models.py                  # lưu models/ensemble_dinhduong.joblib
python3 src/predict.py                            # demo dự đoán
```

Hoặc mở `notebooks/BoKhoe_Colab_SoSanh_DinhDuong.ipynb` trên Google Colab
(tự tải dữ liệu, không cần cài đặt gì).

## Dữ liệu & trích dẫn

- Bộ dữ liệu: Pongsanun K, Apirak T, Leklerdsiriwong A, Chunithipaisan S,
  Inchaisri C. *BMC Veterinary Research* 21:677 (2025), PMC12632110 — dữ liệu
  công khai kèm bài báo (giấy phép CC-BY):
  https://github.com/pongsanun/cow-behavior-classification-acc-gyro-combined
- Nếu dùng lại mã/ý tưởng cải tiến, vui lòng trích dẫn repo này và bài báo gốc.

## Giấy phép

Mã nguồn: MIT (xem LICENSE). Dữ liệu trong `data/` thuộc bản quyền của nhóm tác
giả gốc, phân phối theo giấy phép CC-BY của BMC Veterinary Research.
