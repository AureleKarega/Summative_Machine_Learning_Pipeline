"""Extract features for the ESC-10 subset -> data/base_features.csv
Usage: python scripts/build_features.py /path/to/ESC-50   (folder containing meta/ and audio/)"""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import load_audio, extract_features

root = Path(sys.argv[1])
meta = pd.read_csv(root / "meta" / "esc50.csv")
meta = meta[meta.esc10]
rows = []
for r in meta.itertuples():
    f = extract_features(load_audio(root / "audio" / r.filename))
    rows.append({"filename": r.filename, "label": r.category, "fold": r.fold, **f})
out = Path(__file__).resolve().parents[1] / "data" / "base_features.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print("saved", out, len(rows), "rows")
