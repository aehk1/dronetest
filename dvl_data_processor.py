import os
import json
import glob
import pandas as pd
from datetime import datetime, timedelta

# -------- User needs to modify here: Data Directory --------
DATA_DIR = r"PATH_TO_YOUR_DATA_DIRECTORY" 
OUT_DIR = os.path.join(DATA_DIR, "converted")
# ---------------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)  # Ensure output directory exists

def parse_start_dt_from_name(fname: str) -> datetime:
    """
    Extract JST start time from filename.
    Filename format: metafile_YYYYMMDD_HHMMSS.json
    """
    base = os.path.basename(fname)
    stem = base.replace("metafile_", "").replace(".json", "")
    return datetime.strptime(stem, "%Y%m%d_%H%M%S")

def json_to_rows(json_path: str):
    """Convert a single DVL json to a list of rows."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    start_dt = parse_start_dt_from_name(json_path)
    elapsed_us = 0
    rows = []
    for rec in data:
        if not rec:
            continue
        ms = rec.get("time", 0)
        try:
            us = int(round(float(ms) * 1000.0))
        except Exception:
            us = 0
        elapsed_us += us
        jst_dt = start_dt + timedelta(microseconds=elapsed_us)
        rows.append({
            "datetime(JST)": jst_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "vx(m/s)": rec.get("vx"),
            "vy(m/s)": rec.get("vy"),
            "vz(m/s)": rec.get("vz"),
            "altitude(m)": rec.get("altitude"),
            "fom": rec.get("fom"),
            "velocity_valid": rec.get("velocity_valid"),
        })
    return rows

def process_one(json_path: str) -> pd.DataFrame:
    """Process single json -> CSV, and return DataFrame."""
    rows = json_to_rows(json_path)
    df = pd.DataFrame(rows)
    df["datetime(JST)"] = df["datetime(JST)"].astype(str)

    # Output file path
    base = os.path.basename(json_path)
    csv_name = base.replace(".json", ".csv")
    csv_path = os.path.join(OUT_DIR, csv_name)

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Done: {base} -> {csv_path}")

    df.insert(0, "source_file", base)  # Add source file column
    return df

def main():
    json_files = sorted(glob.glob(os.path.join(DATA_DIR, "metafile_*.json")))
    if not json_files:
        print("No metafile_*.json files found in the directory")
        return

    all_dfs = []
    for jf in json_files:
        try:
            df = process_one(jf)
            all_dfs.append(df)
        except Exception as e:
            print(f"Processing failed: {os.path.basename(jf)} -- {e}")

    if all_dfs:
        big = pd.concat(all_dfs, ignore_index=True)
        big["_sort_dt"] = pd.to_datetime(big["datetime(JST)"], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
        big = big.sort_values("_sort_dt").drop(columns=["_sort_dt"]).reset_index(drop=True)

        combined_csv = os.path.join(OUT_DIR, "dvl.csv")
        big.to_csv(combined_csv, index=False, encoding="utf-8-sig")
        print(f"Summary table generated: {combined_csv}")

if __name__ == "__main__":
    main()
