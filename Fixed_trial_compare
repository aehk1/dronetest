import os
import pandas as pd
import matplotlib.pyplot as plt

# --- Configuration: File Paths ---
# User needs to modify these paths
FILE_DVL    = r"PATH_TO_YOUR_DVL_FILE.xlsx"
FILE_UQPS   = r"PATH_TO_YOUR_UQPS_FILE.xlsx"
FILE_FIFISH = r"PATH_TO_YOUR_FIFISH_FILE.xlsx"

OUT_DIR = r"PATH_TO_OUTPUT_DIRECTORY"
os.makedirs(OUT_DIR, exist_ok=True)

# Define statistical time range
START = pd.Timestamp("2023-09-11 00:00:00")
END   = pd.Timestamp("2023-09-15 23:59:59")

def load_excel_jst(path):
    """
    Specifically reads Excel and processes the time column.
    Returns a Series of timestamps floored to the second.
    """
    print(f"Reading Excel: {os.path.basename(path)}")
    # Use openpyxl engine for reading xlsx
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_excel(path, engine="openpyxl")
    
    # Get time column
    col_name = "datetime(JST)"
    if col_name not in df.columns:
        raise KeyError(f"Column name '{col_name}' not found in file: {path}")
    
    dt = pd.to_datetime(df[col_name], errors="coerce")
    
    # Unify to naive datetime (remove timezone info) for easier comparison
    # Assuming the input string contains timezone info like 'Z' or '+09:00'
    if getattr(dt.dtype, "tz", None) is not None:
         dt = dt.dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
    
    # Filter time range and align to seconds
    mask = (dt >= START) & (dt <= END)
    # Return valid timestamps floored to second precision
    return dt[mask].dropna().dt.floor("s")

def main():
    # ===== 1. Data Loading and Processing =====
    try:
        u_secs = load_excel_jst(FILE_UQPS)
        d_secs = load_excel_jst(FILE_DVL)
        f_secs = load_excel_jst(FILE_FIFISH)
    except Exception as e:
        print(f"Loading failed: {e}")
        return

    # Convert to set for faster lookup (O(1))
    set_u, set_d, set_f = set(u_secs), set(d_secs), set(f_secs)
    print(f"Data count stats (seconds) -> UQPS: {len(set_u)}, DVL: {len(set_d)}, FIFISH: {len(set_f)}")

    # ===== 2. Construct 0/1 Presence Matrix =====
    # Generate a complete time index with 1-second frequency
    rng = pd.date_range(start=START, end=END, freq="s")
    
    presence = pd.DataFrame({
        "UQPS":   rng.isin(set_u).astype(int),
        "DVL":    rng.isin(set_d).astype(int),
        "FIFISH": rng.isin(set_f).astype(int),
    }, index=rng)

    # Save CSV result for reference
    csv_path = os.path.join(OUT_DIR, "presence_per_second.csv")
    presence.to_csv(csv_path, index_label="datetime(JST)")
    print(f"Saved CSV report: {csv_path}")

    # ===== 3. Generate Visualization Charts =====
    dates = presence.index.normalize().unique()
    
    for day in dates:
        # Slice data for the specific day
        day_slice = presence.loc[presence.index.normalize() == day]
        
        if day_slice.empty: 
            continue
        
        # Skip plotting if there is absolutely no data for the whole day
        if day_slice.sum().sum() == 0:
            print(f"Skipping date with no data: {day.date()}")
            continue

        fig, ax = plt.subplots(figsize=(15, 3))
        
        # Use imshow to draw heatmap. 
        # Transpose (.T) so that sensors are on Y-axis and Time is on X-axis.
        # cmap="Greens" means present data (1) is green, missing (0) is white.
        Z = day_slice.T.values
        ax.imshow(Z, aspect="auto", interpolation="nearest", cmap="Greens", vmin=0, vmax=1)
        
        ax.set_title(f"Data Presence Heatmap - {day.date()}", fontsize=12)
        
        # Set Y-axis labels matching the DataFrame column order
        ax.set_yticks(range(3))
        ax.set_yticklabels(["UQPS", "DVL", "FIFISH"])
        
        # Set X-axis ticks (Time) - e.g., every 4 hours
        n = len(day_slice)
        # Create ~7 ticks distributed evenly
        xticks = [0, n//6, 2*n//6, 3*n//6, 4*n//6, 5*n//6, n-1]
        xticklabels = [day_slice.index[i].strftime("%H:%M") for i in xticks]
        
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        
        plt.tight_layout()
        save_path = os.path.join(OUT_DIR, f"presence_{day.date()}.png")
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"Saved plot: {save_path}")

    print(f"All processing complete! Results are in: {OUT_DIR}")

if __name__ == "__main__":
    main()
