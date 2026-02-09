import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import re

# --- Configuration: File Paths ---
# User needs to modify these paths
FILE_UQPS = r"PATH_TO_YOUR_UQPS_FILE.xlsx"
FILE_DVL  = r"PATH_TO_YOUR_DVL_FILE.xlsx"
OUT_DIR   = r"PATH_TO_OUTPUT_DIRECTORY"
# ---------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

# --- Time Range and Segmentation Logic ---
START     = pd.Timestamp("2023-09-11 00:00:00")
END       = pd.Timestamp("2023-09-15 23:59:59")
GAP_TOL_S = 2    # Tolerance for gap detection (seconds)
MIN_SEG_S = 120  # Minimum segment length (seconds)

# --- Analysis Parameters ---
WINDOW_SIZE_POINTS = 25  # Rolling window size for smoothing
DPI                = 160

# --- Helper Functions ---
def read_table(path, **kw):
    """Reads CSV or Excel files based on extension."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv": return pd.read_csv(path, **kw)
    if ext in (".xls", ".xlsx"): return pd.read_excel(path, **kw)
    raise ValueError(f"Unsupported file type: {ext}")

def to_jst_naive(series):
    """Converts time strings to JST (Japan Standard Time) naive datetime objects."""
    s  = series.astype(str).str.replace("Z", "", regex=False)
    dt = pd.to_datetime(s, format="%Y-m-%d %H:%M:%S.%f", errors="coerce", utc=False)
    
    # Handle missing milliseconds
    na = dt.isna()
    if na.any():
        s2  = s[na].str.replace(r"(\d{2}:\d{2}:\d{2}\.)(\d{3})(?!\d)", r"\1\2000", regex=True)
        dt2 = pd.to_datetime(s2, format="%Y-m-%d %H:%M:%S.%f", errors="coerce", utc=False)
        dt  = dt.where(~na, dt2)
    
    # Final fallback
    na2 = dt.isna()
    if na2.any():
        dt3 = pd.to_datetime(s[na2], errors="coerce", utc=False)
        dt  = dt.where(~na2, dt3)
        
    # Convert timezone if present
    if getattr(dt.dtype, "tz", None) is not None:
        dt = dt.dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
    return dt

def apply_valid_mask(df):
    """Filters data based on 'valid' columns if they exist."""
    vcols = [c for c in df.columns if "valid" in c.lower()]
    if not vcols: return df
    mask = None
    for c in vcols:
        v = df[c]
        cur = v if pd.api.types.is_bool_dtype(v) else v.astype(str).str.upper().isin(["1", "TRUE", "T", "YES"])
        mask = cur if mask is None else (mask & cur)
    return df[mask.fillna(False)]

def find_col(df, keys):
    """Fuzzy search for column names."""
    def norm(x): return "".join(ch for ch in str(x).lower() if ch.isalnum() or ch in "_.[] ")
    nm = {c: norm(c) for c in df.columns}
    for k in keys:
        nk = norm(k)
        for c, nc in nm.items():
            if nk == nc: return c
        for c, nc in nm.items():
            if nk in nc: return c
    return None

def safe_name(s: str) -> str:
    """Sanitize string for filename."""
    return re.sub(r'[\\/:*?"<>|]+', '-', str(s).replace(" ", "_"))

def load_raw_data(path, is_uqps=True):
    """Loads raw data and standardizes column names."""
    df   = read_table(path)
    if not is_uqps:
        df = apply_valid_mask(df)
        
    tcol = find_col(df, ["datetime(JST)", "datetime", "time", "timestamp"])
    if tcol is None: raise ValueError("Time column not found.")
    dt = to_jst_naive(df[tcol])
    
    if is_uqps:
        cx, cy, cz = find_col(df, ["x"]), find_col(df, ["y"]), find_col(df, ["z"])
        rename_dict = {cx: "x", cy: "y", cz: "z"}
    else:
        cx, cy, cz = find_col(df, ["vx"]), find_col(df, ["vy"]), find_col(df, ["vz"])
        rename_dict = {cx: "Vx", cy: "Vy", cz: "Vz"}
        
    if not all([cx, cy, cz]): raise ValueError("Missing coordinate columns.")
    
    raw = (df.assign(dt=dt).dropna(subset=["dt"]).set_index("dt")
           [[cx, cy, cz]].rename(columns=rename_dict)
           .apply(pd.to_numeric, errors="coerce").sort_index())
    raw = raw[(raw.index >= START) & (raw.index <= END)]
    return raw

def find_overlap_segments(raw_u, raw_v, gap_tol_s, min_len_s):
    """Finds overlapping time segments between two data sources."""
    presence_u = raw_u.dropna(how='all').index.floor("s").unique()
    presence_v = raw_v.dropna(how='all').index.floor("s").unique()
    common = presence_u.intersection(presence_v)
    if common.empty: return []
    common = common.sort_values()
    dif = common.to_series().diff().dt.total_seconds().fillna(0)
    grp = (dif > gap_tol_s).cumsum()
    segs = []
    for _, g in common.to_series().groupby(grp):
        if len(g) >= min_len_s:
            segs.append((g.iloc[0], g.iloc[-1]))
    return segs

# --- Main Execution ---
print("[INFO] Loading raw UQPS and DVL data...")
raw_uqps = load_raw_data(FILE_UQPS, is_uqps=True)
raw_dvl = load_raw_data(FILE_DVL, is_uqps=False)

print("[INFO] Finding overlapping segments...")
all_segments = find_overlap_segments(raw_uqps, raw_dvl, GAP_TOL_S, MIN_SEG_S)

if not all_segments:
    print("[ERROR] No overlapping segments found.")
else:
    print(f"[INFO] Found {len(all_segments)} segments. Starting pattern analysis loop...")
    
    for i, (seg_start, seg_end) in enumerate(all_segments, 1):
        print(f"\n--- Analyzing Segment {i}: from {seg_start} to {seg_end} ---")
        
        u_segment = raw_uqps[(raw_uqps.index >= seg_start) & (raw_uqps.index <= seg_end)]
        v_segment = raw_dvl[(raw_dvl.index >= seg_start) & (raw_dvl.index <= seg_end)].copy()

        if u_segment.empty or v_segment.empty or len(v_segment) < 2:
            print(f"Segment {i} has insufficient data. Skipping.")
            continue
        
        # --- Core Calculations ---
        # 1. Calculate Combined Speed (Magnitude)
        v_segment['Speed'] = np.sqrt(v_segment['Vx']**2 + v_segment['Vy']**2 + v_segment['Vz']**2)
        smooth_speed = v_segment['Speed'].rolling(window=WINDOW_SIZE_POINTS, center=True, min_periods=1).mean()
        
        # 2. Calculate Acceleration (Derivative of velocity)
        time_diff = v_segment.index.to_series().diff().dt.total_seconds()
        vx_diff, vy_diff, vz_diff = v_segment['Vx'].diff(), v_segment['Vy'].diff(), v_segment['Vz'].diff()
        ax, ay, az = vx_diff / time_diff, vy_diff / time_diff, vz_diff / time_diff
        v_segment['Acceleration'] = np.sqrt(ax**2 + ay**2 + az**2)
        smooth_acceleration = v_segment['Acceleration'].rolling(window=WINDOW_SIZE_POINTS, center=True, min_periods=1).mean()
        
        # 3. Calculate Detrended Volatility (Noise/Jitter analysis)
        detrended_speed = v_segment['Speed'] - smooth_speed
        speed_volatility = detrended_speed.rolling(window=WINDOW_SIZE_POINTS, center=True, min_periods=1).std()
        
        # --- Plotting ---
        fig, axs = plt.subplots(4, 1, figsize=(15, 16), sharex=True)
        
        # Subplot 1: UQPS Displacement
        ax1 = axs[0]
        ax1.plot(u_segment.index, u_segment['x'], label='UQPS X')
        ax1.plot(u_segment.index, u_segment['y'], label='UQPS Y')
        ax1.plot(u_segment.index, u_segment['z'], label='UQPS Z')
        ax1.set_ylabel("Position (m)")
        ax1.set_title(f"UQPS Displacement - Segment {i}")
        ax1.legend(loc="best")
        ax1.grid(True, linestyle="--", alpha=0.6)
        
        # Subplot 2: DVL Velocity
        ax2 = axs[1]
        ax2.scatter(v_segment.index, v_segment['Vx'], s=2, alpha=0.5, label="DVL Vx")
        ax2.scatter(v_segment.index, v_segment['Vy'], s=2, alpha=0.5, label="DVL Vy")
        ax2.scatter(v_segment.index, v_segment['Vz'], s=2, alpha=0.5, label="DVL Vz")
        ax2.plot(smooth_speed.index, smooth_speed, color='black', lw=1.5, label='Overall Speed (Smoothed)')
        ax2.set_ylabel("Velocity (m/s)")
        ax2.set_title("DVL Velocity Profile")
        ax2.legend(loc="best", markerscale=4)
        ax2.grid(True, linestyle="--", alpha=0.6)

        # Subplot 3: DVL Acceleration
        ax3 = axs[2]
        ax3.plot(smooth_acceleration.index, smooth_acceleration, color='green', lw=1.5, label="Overall Acceleration (Smoothed)")
        ax3.set_ylabel("Acceleration (m/s^2)")
        ax3.set_title("DVL Acceleration Profile")
        ax3.legend(loc="best")
        ax3.grid(True, linestyle="--", alpha=0.6)

        # Subplot 4: DVL Volatility
        ax4 = axs[3]
        ax4.plot(speed_volatility.index, speed_volatility, color='r', lw=1.5, label="De-trended Speed Volatility")
        ax4.set_ylabel("Volatility (Std Dev)")
        ax4.set_title(f"DVL Speed Volatility (Window={WINDOW_SIZE_POINTS} points)")
        ax4.legend(loc="best")
        ax4.grid(True, linestyle="--", alpha=0.6)
        
        axs[-1].set_xlabel("Time (JST)")
        fig.suptitle(f"Pattern Analysis - Segment {i}", fontsize=16)
        fig.autofmt_xdate()
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        time_str = seg_start.strftime('%Y-%m-%d_%H-%M-%S')
        png_filename = safe_name(f"Pattern_Analysis_Seg{i}_{time_str}.png")
        png_path = os.path.join(OUT_DIR, png_filename)

        fig.savefig(png_path, dpi=DPI)
        plt.close(fig)
        
        print(f"[OK] Pattern analysis plot for Segment {i} saved to: {png_path}")

    print(f"\n[DONE] All {len(all_segments)} segments have been analyzed.")
