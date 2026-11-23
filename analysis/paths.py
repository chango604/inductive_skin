"""Repository paths shared by the analysis scripts.

    data/      raw CSVs from Zenodo (see the README)
    models/    trained MLP models used for Fig. 5
    results/   figures, tables and retrained models written by the scripts
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

INDENT_CSV = DATA_DIR / "3d_force_6000_v2.csv"
CYCLES_CSV = DATA_DIR / "cycles_1200.csv"
COIL_CSVS = [DATA_DIR / f"cycles_L{k}.csv" for k in range(1, 6)]


def data_file(path):
    """Return path, or exit with a hint if the data file has not been downloaded."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"Missing {path.relative_to(ROOT)}. Download the dataset from "
                         f"Zenodo and place the CSV files in data/ (see README.md).")
    return path


def results_dir():
    RESULTS_DIR.mkdir(exist_ok=True)
    return RESULTS_DIR
