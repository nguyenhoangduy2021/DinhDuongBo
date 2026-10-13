# -*- coding: utf-8 -*-
"""Tái tạo đầy đủ thí nghiệm: baseline bài báo (Random Forest) vs các mô hình cải tiến.

Đánh giá out-of-fold StratifiedKFold-10 (seed 42) trên 7.295 cửa sổ 6 lớp.
Kết quả mong đợi (results/compare_feeding.json):
    RF_paper_full      acc 0.9989  macroF1 0.9973
    RF_paper_RFE60     acc 0.9990  macroF1 0.9975
    HGB_ours           acc 0.9995  macroF1 0.9982
    OvR_specialists    acc 0.9995  macroF1 0.9983  ← CHỌN
    HGB_balanced       acc 0.9990  macroF1 0.9977
    TwoStage_pairfix   acc 0.9995  macroF1 0.9982
    RankMix_HGB_OvR    acc 0.8119  (minh hoạ phản ví dụ: rank-OOF toàn cục không hợp lệ)
Chạy: python3 src/experiment_baseline_and_improved.py  (mất vài phút, CPU)
"""
import json, os, warnings
import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier as HGB, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
RNG = 42
np.random.seed(RNG)
HERE = os.path.dirname(os.path.abspath(__file__))
import sys; sys.path.insert(0, HERE)
from features import get_Xy

DATA = os.path.join(HERE, "..", "data", "cow_data_processed_updated.csv")


def pipe(m, scaled=False):
    st = [("imp", SimpleImputer(strategy="median"))]
    if scaled:
        st.append(("sc", StandardScaler()))
    st.append(("clf", m))
    return Pipeline(st)


def metrics(y, pred):
    return {"acc": round(accuracy_score(y, pred), 4),
            "macroF1": round(f1_score(y, pred, average="macro"), 4),
            "F1_eating": round(f1_score(y == 0, pred == 0), 4),
            "F1_walking": round(f1_score(y == 5, pred == 5), 4)}


def main():
    X, y, feat, df = get_Xy(DATA)
    print(f"X: {X.shape} | 6 lớp | {len(feat)} đặc trưng cảm biến")

    skf = StratifiedKFold(10, shuffle=True, random_state=RNG)
    folds = list(skf.split(X, y))
    res = {}

    def oof_eval(name, fn):
        P = np.zeros((len(y), 6))
        for tr, te in folds:
            m = pipe(fn()); m.fit(X.iloc[tr], y[tr])
            P[te] = m.predict_proba(X.iloc[te])
        res[name] = metrics(y, P.argmax(1))
        print(name, res[name])
        return P

    # (1) Baseline bài báo: Random Forest 500 cây trên toàn bộ đặc trưng
    oof_eval("RF_paper_full", lambda: RandomForestClassifier(500, random_state=RNG, n_jobs=2))

    # (1b) RF + chọn top-60 đặc trưng theo importance (RFE-style)
    rf0 = RandomForestClassifier(500, random_state=RNG, n_jobs=2).fit(X, y)
    idx60 = np.argsort(rf0.feature_importances_)[::-1][:60]
    P_rf60 = np.zeros((len(y), 6))
    for tr, te in folds:
        m = RandomForestClassifier(500, random_state=RNG, n_jobs=2)
        m.fit(X.iloc[tr, idx60], y[tr]); P_rf60[te] = m.predict_proba(X.iloc[te, idx60])
    res["RF_paper_RFE60"] = metrics(y, P_rf60.argmax(1))
    print("RF_paper_RFE60", res["RF_paper_RFE60"])

    # (2) HGB đa lớp (mô hình luận văn v2)
    P_hgb = oof_eval("HGB_ours", lambda: HGB(random_state=RNG, max_iter=300))

    # (3) CẢI TIẾN CHỌN: ensemble OvR – 6 chuyên gia nhị phân HGB
    P_ovr = np.zeros((len(y), 6))
    for c in range(6):
        yc = (y == c).astype(int)
        for tr, te in folds:
            m = HGB(random_state=RNG, max_iter=300)
            m.fit(X.iloc[tr], yc[tr]); P_ovr[te, c] = m.predict_proba(X.iloc[te])[:, 1]
    res["OvR_specialists"] = metrics(y, P_ovr.argmax(1))
    print("OvR_specialists", res["OvR_specialists"])

    # (4) HGB balanced (đối trọng lớp hiếm walking/eating)
    oof_eval("HGB_balanced", lambda: HGB(random_state=RNG, max_iter=300, class_weight="balanced"))

    # (5) Hai tầng: HGB 6 lớp + chuyên gia sửa cặp dễ nhầm (standing vs lying ruminating)
    pair = np.isin(y, [2, 3])
    P_two = P_hgb.copy()
    for tr, te in folds:
        trp, tep = tr[pair[tr]], te[pair[te]]
        m = HGB(random_state=RNG, max_iter=300)
        m.fit(X.iloc[trp], (y[trp] == 3).astype(int))
        q = m.predict_proba(X.iloc[tep])[:, 1]
        s = P_two[tep, 2] + P_two[tep, 3]
        P_two[tep, 3] = s * q; P_two[tep, 2] = s * (1 - q)
    res["TwoStage_pairfix"] = metrics(y, P_two.argmax(1))
    print("TwoStage_pairfix", res["TwoStage_pairfix"])

    # (6) Phản ví dụ: rank-mix trên OOF toàn cục (không hợp lệ về mặt đánh giá)
    def rk(a): return rankdata(a) / (len(a) + 1)
    P_mix = 0.5 * np.array([rk(P_hgb[:, c]) for c in range(6)]).T \
          + 0.5 * np.array([rk(P_ovr[:, c]) for c in range(6)]).T
    res["RankMix_HGB_OvR"] = metrics(y, P_mix.argmax(1))
    print("RankMix_HGB_OvR", res["RankMix_HGB_OvR"])

    out = os.path.join(HERE, "..", "results", "compare_feeding.json")
    json.dump(res, open(out, "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
