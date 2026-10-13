# -*- coding: utf-8 -*-
"""Gói bằng chứng thống kê bổ sung (đề tài dinh dưỡng):
  (1) 5 seed: mean±SD (acc, macroF1, F1_walking) cho RF-baseline vs OvR-chọn; Wilcoxon
  (2) Bootstrap 95% CI cho OvR
  (3) Ablation tăng cường lớp hiếm (SMOTE-lite nội suy + nhiễu Gauss, khai báo minh bạch)
Kết quả: thesis/results/robustness_feeding.json
"""
import sys, json, warnings, time
import numpy as np
from scipy.stats import wilcoxon
from sklearn.ensemble import HistGradientBoostingClassifier as HGB, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import get_Xy

import os as _os
X, y, feat, df = get_Xy(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "data", "cow_data_processed_updated.csv"))
Xv = X.values.astype(float)
col_sd = np.nanstd(Xv, axis=0); col_sd[col_sd == 0] = 1
print("n=%d, feat=%d" % (len(y), Xv.shape[1]))


def smote_lite(Xc, yc_classes, rng, k=5):
    """Nội suy SMOTE: mỗi mẫu thiểu số + 1 bản tổng hợp với láng giềng gần."""
    if len(Xc) < k + 1: k = max(1, len(Xc) - 1)
    out = []
    for i in range(len(Xc)):
        d = np.linalg.norm(Xc - Xc[i], axis=1); d[i] = np.inf
        nn = np.argsort(d)[:k]
        j = nn[rng.randint(k)]
        t = rng.rand()
        out.append(Xc[i] + t * (Xc[j] - Xc[i]))
    return np.vstack(out)


def run(seed, mode="plain"):
    """mode: plain | smote | noise — trả về dict chỉ số của RF và OvR."""
    skf = StratifiedKFold(10, shuffle=True, random_state=seed)
    Prf, Povr = np.zeros((len(y), 6)), np.zeros((len(y), 6))
    for tr, te in skf.split(Xv, y):
        Xtr, ytr = Xv[tr], y[tr]
        rng = np.random.RandomState(seed)
        if mode in ("smote", "noise"):
            xs, ys = [Xtr], [ytr]
            for c in range(6):
                share = (ytr == c).mean()
                if share < 0.10:  # lớp hiếm: walking (1.8%), eating (8.4%)
                    Xc = Xtr[ytr == c]
                    if mode == "smote":
                        xs.append(smote_lite(Xc, c, rng))
                    else:
                        xs.append(Xc + rng.randn(*Xc.shape) * (0.05 * col_sd))
                    ys.append(np.full(len(xs[-1]), c))
            Xtr, ytr = np.vstack(xs), np.concatenate(ys)
        rf = Pipeline([("imp", SimpleImputer(strategy="median")),
                       ("clf", RandomForestClassifier(500, random_state=seed, n_jobs=2))])
        rf.fit(Xtr, ytr); Prf[te] = rf.predict_proba(Xv[te])
        for c in range(6):
            m = Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("clf", HGB(random_state=seed, max_iter=300))])
            m.fit(Xtr, (ytr == c).astype(int))
            Povr[te, c] = m.predict_proba(Xv[te])[:, 1]
    p_rf, p_ovr = Prf.argmax(1), Povr.argmax(1)
    return {"rf": [accuracy_score(y, p_rf), f1_score(y, p_rf, average="macro"),
                   f1_score(y == 5, p_rf == 5)],
            "ovr": [accuracy_score(y, p_ovr), f1_score(y, p_ovr, average="macro"),
                    f1_score(y == 5, p_ovr == 5)]}

t0 = time.time()
SEEDS = [42, 7, 123, 2024, 2026]
allr = {m: {"rf": [], "ovr": []} for m in ["acc", "mf1", "fw"]}
for s in SEEDS:
    r = run(s)
    for j, k in enumerate(["acc", "mf1", "fw"]):
        allr[k]["rf"].append(r["rf"][j]); allr[k]["ovr"].append(r["ovr"][j])
    print("seed %5d  RF acc %.4f mf1 %.4f fw %.4f | OvR acc %.4f mf1 %.4f fw %.4f"
          % (s, *r["rf"], *r["ovr"]))

ms = lambda v: [round(float(np.mean(v)), 4), round(float(np.std(v)), 4)]
p_acc = wilcoxon(allr["acc"]["ovr"], allr["acc"]["rf"], alternative="greater")[1]
p_fw = wilcoxon(allr["fw"]["ovr"], allr["fw"]["rf"], alternative="greater")[1]

# CI 95% ước lượng từ phân phối t trên 5 seed (trung thực, không dùng OOF kép)
from scipy.stats import t as tdist  # noqa: F401
ci95 = lambda v: [round(float(np.mean(v) - 2.776 * np.std(v, ddof=1) / np.sqrt(len(v))), 4),
                  round(float(np.mean(v) + 2.776 * np.std(v, ddof=1) / np.sqrt(len(v))), 4)]

ab_smote = run(42, "smote")
ab_noise = run(42, "noise")

res = {
 "seeds": {str(s): {"rf": [round(v, 4) for v in [allr[k]["rf"][i] for k in ["acc", "mf1", "fw"]]],
                    "ovr": [round(v, 4) for v in [allr[k]["ovr"][i] for k in ["acc", "mf1", "fw"]]]}
           for i, s in enumerate(SEEDS)},
 "mean_sd": {"rf": {"acc": ms(allr["acc"]["rf"]), "macroF1": ms(allr["mf1"]["rf"]), "F1_walking": ms(allr["fw"]["rf"])},
             "ovr": {"acc": ms(allr["acc"]["ovr"]), "macroF1": ms(allr["mf1"]["ovr"]), "F1_walking": ms(allr["fw"]["ovr"])}},
 "ci95_t_ovr": {"acc": ci95(allr["acc"]["ovr"]), "macroF1": ci95(allr["mf1"]["ovr"]), "F1_walking": ci95(allr["fw"]["ovr"])},
 "wilcoxon_p_one_sided": {"acc": float(p_acc), "F1_walking": float(p_fw)},
 "rare_class_augmentation_seed42": {"smote": {k: [round(v, 4) for v in vv] for k, vv in ab_smote.items()},
                                    "noise": {k: [round(v, 4) for v in vv] for k, vv in ab_noise.items()}},
}
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "robustness_feeding.json"), "w"), indent=1)
print("saved | %.0fs" % (time.time() - t0))
