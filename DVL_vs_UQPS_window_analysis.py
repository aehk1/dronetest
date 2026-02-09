import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ====== Raw Data Paths ======
# User needs to modify these paths
FILE_UQPS   = r"PATH_TO_YOUR_UQPS_FILE.xlsx"
FILE_DVL    = r"PATH_TO_YOUR_DVL_FILE.xlsx"
OUT_DIR     = r"PATH_TO_OUTPUT_DIRECTORY"
# ----------------------------

os.makedirs(OUT_DIR, exist_ok=True)

# ====== Plotting Parameters ======
GAP_SEC      = 2.0    # DVL time gap > threshold considered as segment break; integration resets to 0
VEL_JUMP_TH  = 0.30   # Velocity jump threshold (m/s), set to NaN if exceeded
UQPS_MS      = 9      # UQPS scatter dot size
DVL_LW       = 1.5    # DVL line width
DPI          = 160    # Output DPI

# ====== Time Windows ======
# Manually defined time ranges for analysis
WINDOWS = [
    ("2023-09-12 14:30:00", "2023-09-12 16:45:00"),
    ("2023-09-13 13:30:00", "2023-09-13 15:30:00"),
    ("2023-09-14 13:20:00", "2023-09-14 14:50:00"),
    ("2023-09-14 17:00:00", "2023-09-14 17:45:00"),
    ("2023-09-15 14:30:00", "2023-09-15 16:30:00"),
]

def read_table(path):
    """Reads Excel or CSV files based on extension."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xls", ".xlsx"): return pd.read_excel(path)
    if ext == ".csv":            return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {ext}")

def to_jst(series):
    """Converts time strings to JST (Asia/Tokyo) naive datetime objects."""
    s = series.astype(str).str.replace("Z", "", regex=False)
    dt = pd.to_datetime(s, errors="coerce", utc=False)
    if getattr(dt.dtype, "tz", None) is not None:
        dt = dt.dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
    return dt

def fcol(df, keys):
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

def integrate_with_gaps(tindex, v, gap_sec=GAP_SEC):
    """
    Trapezoidal integration with gap handling.
    If a time gap > gap_sec is detected, the position resets to 0.0.
    """
    if len(tindex) == 0:
        return pd.Series([], dtype=float)

    t = tindex.view(np.int64) / 1e9
    out = np.full(len(v), np.nan, float)
    out[0] = 0.0
    for i in range(1, len(v)):
        dt = t[i] - t[i-1]
        if np.isnan(v[i]) or np.isnan(v[i-1]) or dt <= 0 or dt > gap_sec:
            # Reset integration to 0 upon gap detection
            out[i] = 0.0
        else:
            out[i] = out[i-1] + 0.5 * (v[i] + v[i-1]) * dt
    return pd.Series(out, index=tindex)

def safe_name(s):
    """Sanitize string for filename."""
    return re.sub(r'[\\/:*?"<>|]+', '-', str(s).replace(" ", "_"))

# ====== Read UQPS ======
has_uqps = False
try:
    if os.path.exists(FILE_UQPS):
        uq = read_table(FILE_UQPS)
        tc = fcol(uq, ["datetime(JST)", "datetime", "time", "timestamp"])
        rx = fcol(uq, ["relative x", "x"]); ry = fcol(uq, ["relative y", "y"]); rz = fcol(uq, ["relative z", "z"])
        if tc and rx and ry and rz:
            dt = to_jst(uq[tc])
            uq = uq.assign(dt=dt).dropna(subset=["dt"]).set_index("dt").sort_index()
            uq = uq.rename(columns={rx: "x", ry: "y", rz: "z"})[["x", "y", "z"]].apply(pd.to_numeric, errors="coerce")
            has_uqps = True
except Exception as e:
    print(f"UQPS Load Warning: {e}")
    uq = None

# ====== Read DVL ======
dvl = read_table(FILE_DVL)
tc = fcol(dvl, ["datetime(JST)", "datetime", "time"])
cx = fcol(dvl, ["vx(m/s)", "vx"]); cy = fcol(dvl, ["vy(m/s)", "vy"]); cz = fcol(dvl, ["vz(m/s)", "vz"])
dt = to_jst(dvl[tc])
dvl = dvl.assign(dt=dt).dropna(subset=["dt"]).set_index("dt").sort_index()
dvl = dvl.rename(columns={cx: "Vx", cy: "Vy", cz: "Vz"})[["Vx", "Vy", "Vz"]].apply(pd.to_numeric, errors="coerce")

# Remove velocity jumps
for c in ["Vx", "Vy", "Vz"]:
    jump = dvl[c].diff().abs()
    dvl.loc[jump > VEL_JUMP_TH, c] = np.nan
dvl = dvl.dropna(how="all")

# ====== Plot by Time Windows ======
for (ts, te) in WINDOWS:
    start, end = pd.Timestamp(ts), pd.Timestamp(te)

    dvl_seg = dvl.loc[(dvl.index >= start) & (dvl.index <= end)]
    if dvl_seg.empty:
        continue

    # Perform integration (resets on gaps)
    X = integrate_with_gaps(dvl_seg.index, dvl_seg["Vx"].to_numpy())
    Y = integrate_with_gaps(dvl_seg.index, dvl_seg["Vy"].to_numpy())
    Z = integrate_with_gaps(dvl_seg.index, dvl_seg["Vz"].to_numpy())

    # Mask the plot lines where gaps occur (so they don't connect across big jumps)
    dif = dvl_seg.index.to_series().diff().dt.total_seconds().fillna(0)
    mask_break = dif > GAP_SEC
    for S in (X, Y, Z):
        S[mask_break] = np.nan

    # Process UQPS segment if available
    uq_disp = None
    if has_uqps:
        uq_seg = uq.loc[(uq.index >= start) & (uq.index <= end)]
        if not uq_seg.empty:
            dif_u = uq_seg.index.to_series().diff().dt.total_seconds().fillna(0)
            seg_id = dif_u.gt(GAP_SEC).cumsum()
            uq_disp = uq_seg.copy()
            # Reset UQPS to relative 0 for each continuous segment to match DVL behavior
            for _, sub in uq_seg.groupby(seg_id):
                uq_disp.loc[sub.index, ["x", "y", "z"]] = sub[["x", "y", "z"]].values - sub.iloc[0][["x","y","z"]].values
        
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

    # Plot X
    axes[0].plot(X.index, X.values, lw=DVL_LW, color="red", label="DVL∫ x", alpha=0.95)
    if uq_disp is not None and not uq_disp.empty:
        axes[0].scatter(uq_disp.index, uq_disp["x"], s=UQPS_MS, color="blue", alpha=0.7, label="UQPS x")
    axes[0].set_ylabel("x (m)"); axes[0].legend(loc="best", fontsize=9)
    axes[0].grid(True, which="both", axis="both", linestyle="--", alpha=0.6); axes[0].minorticks_on()

    # Plot Y
    axes[1].plot(Y.index, Y.values, lw=DVL_LW, color="red", label="DVL∫ y", alpha=0.95)
    if uq_disp is not None and not uq_disp.empty:
        axes[1].scatter(uq_disp.index, uq_disp["y"], s=UQPS_MS, color="blue", alpha=0.7, label="UQPS y")
    axes[1].set_ylabel("y (m)"); axes[1].legend(loc="best", fontsize=9)
    axes[1].grid(True, which="both", axis="both", linestyle="--", alpha=0.6); axes[1].minorticks_on()

    # Plot Z
    axes[2].plot(Z.index, Z.values, lw=DVL_LW, color="red", label="DVL∫ z", alpha=0.95)
    if uq_disp is not None and not uq_disp.empty:
        axes[2].scatter(uq_disp.index, uq_disp["z"], s=UQPS_MS, color="blue", alpha=0.7, label="UQPS z")
    axes[2].set_ylabel("z (m)"); axes[2].set_xlabel("Time (JST)")
    axes[2].legend(loc="best", fontsize=9)
    axes[2].grid(True, which="both", axis="both", linestyle="--", alpha=0.6); axes[2].minorticks_on()

    fig.suptitle(f"UQPS vs DVL — {start:%Y-%m-%d}  [{start:%H:%M:%S} ~ {end:%H:%M:%S}]")
    fig.autofmt_xdate()
    fig.tight_layout(rect=[0,0,1,0.95])

    outfile = os.path.join(OUT_DIR, safe_name(f"Segment_{start:%Y-%m-%d_%H%M%S}-{end:%H%M%S}.png"))
    fig.savefig(outfile, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {outfile}")

print(f"[DONE] Segment plots saved to: {OUT_DIR}")
