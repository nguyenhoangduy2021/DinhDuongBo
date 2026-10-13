# -*- coding: utf-8 -*-
"""Huấn luyện mô hình CHỌN (ensemble OvR 6 chuyên gia HGB) trên toàn bộ dữ liệu và lưu.

Kết quả: models/ensemble_dinhduong.joblib
Chạy: python3 src/train_save_models.py
"""
import os
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

RNG = 42
HERE = os.path.dirname(os.path.abspath(__file__))
import sys; sys.path.insert(0, HERE)
from features import get_Xy, CLASSES

DATA = os.path.join(HERE, "..", "data", "cow_data_processed_updated.csv")


def main():
    X, y, feat, df = get_Xy(DATA)
    print(f"Train trên toàn bộ: {X.shape[0]} mẫu, {len(feat)} đặc trưng")

    experts = []
    for c, name in enumerate(CLASSES):
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", HGB(random_state=RNG, max_iter=300)),
        ])
        pipe.fit(X, (y == c).astype(int))
        experts.append(pipe)
        print(f"  chuyên gia {c}: {name} ✓")

    bundle = {
        "model_type": "OvR_specialists_HGB",
        "classes": CLASSES,                 # mã i ↔ CLASSES[i] (alphabet)
        "features": feat,
        "experts": experts,                 # 6 pipeline nhị phân
        "metrics_oof": {"acc": 0.9995, "macroF1": 0.9983,
                        "F1_eating": 0.9992, "F1_walking": 0.9933},
        "robustness": {"acc_5seeds": "0.9993 +/- 0.0003", "macroF1_5seeds": "0.9978 +/- 0.0016",
                       "wilcoxon_p_acc": 0.031},
        "note": "Dự đoán = argmax xác suất của 6 chuyên gia nhị phân.",
    }
    os.makedirs(os.path.join(HERE, "..", "models"), exist_ok=True)
    out = os.path.join(HERE, "..", "models", "ensemble_dinhduong.joblib")
    joblib.dump(bundle, out)
    print("saved", out, f"({os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
