"""
train_models.py  ->  Table S1, Table S2
=======================================
Trains the decision tree, random forest and MLP for the location (x, y, z) and force
(F) outputs on the same 80/20 split, with hyperparameters chosen by 5-fold grouped
cross-validation on the training set (see indent_models.py).

    python analysis/train_models.py

Writes to results/:
    table_s1.csv           test MAE of every model, all depths and depth >= 2 mm
    table_s2.csv           selected hyperparameters
    models/indent3d_<set>_<model>.joblib (+ _meta.json)
The MLP models in models/ are the ones used for Fig. 5 (trained with
scikit-learn 1.8.0); another scikit-learn version can give slightly different numbers.
The grid search takes a while (tens of minutes on a laptop).
"""
import warnings

import numpy as np
import pandas as pd

import extract_indent_data as eid
import indent_models as imd
from paths import INDENT_CSV, data_file, results_dir

N_POINTS = 5500          # indentations used (fixed-seed subset of the valid points)
SUBSAMPLE_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
SEED = 42
MIN_DEPTH_MM = 2.0       # second block of Table S1


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    pts = eid.load_dataset(data_file(INDENT_CSV), N_POINTS, SUBSAMPLE_SEED)
    train_ids, test_ids = imd.split_points(pts["point_id"], TEST_SIZE, SEED)
    print(f"[train] train points: {len(train_ids)}  test points: {len(test_ids)}")
    out = results_dir()
    (out / "models").mkdir(exist_ok=True)

    mae_rows, param_rows = [], []
    for set_name, targets in imd.TARGET_SETS.items():
        X, Y, ids, depth = imd.make_dataset(pts, targets)
        tr, te = np.isin(ids, train_ids), np.isin(ids, test_ids)
        deep = depth[te] >= MIN_DEPTH_MM
        for name in imd.MODEL_NAMES:
            print(f"[train] {set_name:3s} {name:3s} grid search ...", flush=True)
            model, params = imd.tune_and_fit(name, X[tr], Y[tr], ids[tr], CV_FOLDS, SEED)
            err = np.abs(Y[te] - imd.predict(model, X[te]))
            test_mae = {t: float(err[:, j].mean()) for j, t in enumerate(targets)}
            for subset, m in (("all depths", np.ones(len(err), bool)),
                              (f"depth >= {MIN_DEPTH_MM:g} mm", deep)):
                for j, t in enumerate(targets):
                    scale = imd.G_TO_N if t == "F" else 1.0
                    mae_rows.append({"model": name, "test points": subset, "n": int(m.sum()),
                                     "output": t, "MAE": err[m, j].mean() * scale,
                                     "unit": "N" if t == "F" else "mm"})
            param_rows.append({"model": name, "output": set_name,
                               **{k.split("__")[-1]: v for k, v in params.items()}})
            print(f"        {params}  " + "  ".join(f"{t} {v:.3f}" for t, v in test_mae.items()))
            imd.save_model(model, out / "models" / f"indent3d_{set_name}_{name}.joblib",
                           name, set_name, params, test_mae)

    s1 = (pd.DataFrame(mae_rows)
          .pivot_table(index=["test points", "n", "model"], columns="output", values="MAE",
                       sort=False)[["x", "y", "z", "F"]])
    s2 = pd.DataFrame(param_rows).set_index(["model", "output"])
    s1.to_csv(out / "table_s1.csv")
    s2.to_csv(out / "table_s2.csv")
    with pd.option_context("display.width", 150, "display.precision", 3):
        print("\nTable S1: test MAE (x, y, z in mm; F in N)\n", s1)
        print("\nTable S2: selected hyperparameters\n", s2)
    print(f"\nwrote {out}/table_s1.csv, table_s2.csv and models/")


if __name__ == "__main__":
    main()
