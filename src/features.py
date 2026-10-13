# -*- coding: utf-8 -*-
"""Chuẩn bị dữ liệu hành vi ăn/uống bò sữa (Pongsanun et al. 2025, BMC Vet Res 21:677).

Nguồn dữ liệu công khai (CSV 7.295 cửa sổ × 142 cột):
https://raw.githubusercontent.com/pongsanun/cow-behavior-classification-acc-gyro-combined/main/cow_data_processed_updated.csv

Nhãn 6 lớp (LabelEncoder xếp alphabet → mã số):
    eating=0, lying=1, lying ruminating=2, standing=3,
    standing ruminating=4, walking=5
(walking ruminating được gộp vào walking theo thiết kế thí nghiệm)
"""
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

CLASSES = ["eating", "lying", "lying ruminating", "standing",
           "standing ruminating", "walking"]          # thứ tự alphabet = mã 0..5

# 9 cột nhị phân chỉ báo hành vi là RÒ RỈ NHÃN (sinh ra từ chính nhãn) → loại
LEAK_COLS = ["Lying", "Lying down", "Standing", "Standing up", "Walking",
             "Feeding", "Dringking", "Ruminating", "Other"]

RAW_COLS = ["accX", "accY", "accZ", "gyroX", "gyroY", "gyroZ",
            "Sensor", "Time", "ID", "Classification"]


def load_raw(path):
    df = pd.read_csv(path)
    df["behav"] = df["Classification"].replace({"walking ruminating": "walking"})
    df = df[df["behav"].isin(CLASSES)].copy()
    le = LabelEncoder()                      # alphabet → eating=0 ... walking=5
    df["y"] = le.fit_transform(df["behav"])
    return df


def feature_columns(df):
    """Các cột đặc trưng hợp lệ: số, không phải cột thô/cột rò rỉ."""
    exclude = set(RAW_COLS + LEAK_COLS + ["behav", "y"] + CLASSES)
    return [c for c in df.columns if c not in exclude and df[c].dtype != object]


def get_Xy(csv_path):
    df = load_raw(csv_path)
    feat = feature_columns(df)
    X = df[feat].astype(float)
    y = df["y"].values
    return X, y, feat, df
