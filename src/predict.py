# -*- coding: utf-8 -*-
"""Minh hoạ nạp mô hình đã lưu và dự đoán hành vi từ vector đặc trưng cảm biến.

Chạy: python3 src/predict.py
Lưu ý: demo chấm trên chính dữ liệu huấn luyện (in-sample) chỉ để kiểm tra
pipeline hoạt động; số đánh giá trung thực là kết quả out-of-fold
acc 0.9995 / macroF1 0.9983 trong results/compare_feeding.json.
"""
import os
import joblib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import sys; sys.path.insert(0, HERE)
from features import get_Xy

MODEL = os.path.join(HERE, "..", "models", "ensemble_dinhduong.joblib")
DATA = os.path.join(HERE, "..", "data", "cow_data_processed_updated.csv")


def predict(bundle, Xnew):
    """Xnew: DataFrame đúng thứ tự cột bundle['features'] → (nhãn, xác suất 6 lớp)."""
    P = np.column_stack([exp.predict_proba(Xnew)[:, 1] for exp in bundle["experts"]])
    idx = P.argmax(1)
    return [bundle["classes"][i] for i in idx], P


def main():
    bundle = joblib.load(MODEL)
    print("Mô hình:", bundle["model_type"], "| lớp:", bundle["classes"])

    X, y, feat, _ = get_Xy(DATA)
    Xn = X[bundle["features"]]
    labels, P = predict(bundle, Xn[:2000])

    correct = sum(1 for lab, yi in zip(labels, y[:2000])
                  if bundle["classes"].index(lab) == yi)
    print(f"Demo 2000 mẫu đầu (in-sample): đúng {correct}/2000 "
          f"= {correct/20:.1f}%")
    print("(in-sample chỉ kiểm tra pipeline; số chính thức = OOF: acc 0.9995)")


if __name__ == "__main__":
    main()
