import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re

# --- Configuration: File Paths ---
# User needs to modify these paths
FILE_DVL_ENHANCED = r"PATH_TO_YOUR_ENHANCED_DVL_CSV_FILE.csv"
FILE_UQPS         = r"PATH_TO_YOUR_UQPS_EXCEL_FILE.xlsx"
OUT_DIR           = r"PATH_TO_OUTPUT_DIRECTORY"
# ---------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

START     = pd.Timestamp("2023-09-11 00:00:00")
END       = pd.Timestamp("2023-09-15 23:59:59")
GAP_TOL_S = 2      # Maximum allowed gap tolerance in seconds
MIN_SEG_S = 120    # Minimum segment length in seconds
DPI       = 150
COLORS_ID = {0: 'tab:red', 1: 'tab:green', 2: 'tab:blue', 3: 'tab:orange'}

def load_data():
    """Loads and preprocesses Enhanced DVL and UQPS data."""
    print("[INFO] Loading Enhanced DVL Data (CSV)...")
    if not os.path.exists(FILE_DVL_ENHANCED):
        print(f"[ERROR] File not found: {FILE_DVL_ENHANCED}")
        return pd.DataFrame(), pd.DataFrame()

    df_dvl = pd.read_csv(FILE_DVL_ENHANCED)
    
    # Parse timestamps
    df_dvl['dt'] = pd.to_datetime(df_dvl['datetime(JST)'])
    
    # Filter by time range
    df_dvl = df_dvl[(df_dvl['dt'] >= START) & (df_dvl['dt'] <= END)].sort_values('dt')
    
    print("[INFO] Loading UQPS Data (Excel)...")
    df_uqps = pd.read_excel(FILE_UQPS)
    
    # Handle UQPS time column
    tcol = next((c for c in df_uqps.columns if "time" in str(c).lower()), None)
    if tcol:
        t_str = df_uqps[tcol].astype(str).str.replace("Z", "")
        df_uqps['dt'] = pd.to_datetime(t_str, format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
    
    # Handle UQPS coordinate columns
    cx = next((c for c in df_uqps.columns if str(c).lower() in ['relative x', 'rel_x', 'x']), None)
    cy = next((c for c in df_uqps.columns if str(c).lower() in ['relative y', 'rel_y', 'y']), None)
    cz = next((c for c in df_uqps.columns if str(c).lower() in ['relative z', 'rel_z', 'z']), None)
    
    clean_uqps = df_uqps[['dt', cx, cy, cz]].dropna(subset=['dt']).rename(columns={cx:'x', cy:'y', cz:'z'})
    clean_uqps = clean_uqps[(clean_uqps['dt'] >= START) & (clean_uqps['dt'] <= END)].sort_values('dt')
    
    # Remove timezone info for alignment
    if df_dvl['dt'].dt.tz is not None: df_dvl['dt'] = df_dvl['dt'].dt.tz_localize(None)
    if clean_uqps['dt'].dt.tz is not None: clean_uqps['dt'] = clean_uqps['dt'].dt.tz_localize(None)
    
    return df_dvl.set_index('dt'), clean_uqps.set_index('dt')

def find_overlap_segments(raw_u, raw_v):
    """Finds overlapping time segments between two datasets."""
    print("[INFO] Calculating overlapping segments...")
    presence_u = raw_u.index.floor("s").unique()
    presence_v = raw_v.index.floor("s").unique()
    common = presence_u.intersection(presence_v).sort_values()
    
    if common.empty: return []

    dif = common.to_series().diff().dt.total_seconds().fillna(0)
    grp = (dif > GAP_TOL_S).cumsum()
    
    segs = []
    for _, g in common.to_series().groupby(grp):
        if len(g) >= MIN_SEG_S:
            segs.append((g.iloc[0], g.iloc[-1]))
    return segs

def plot_deep_dive_segment(dvl_seg, uqps_seg, seg_idx):
    """Generates a detailed multi-panel plot for a single segment."""
    if dvl_seg.empty: return

    time_str = dvl_seg.index[0].strftime("%Y%m%d_%H%M%S")
    
    # Create subplots (8 rows)
    fig, axs = plt.subplots(8, 1, figsize=(16, 32), sharex=True)
    t = dvl_seg.index
    
    # --- Row 1: UQPS Position (Reference) ---
    ax = axs[0]
    if not uqps_seg.empty:
        ax.plot(uqps_seg.index, uqps_seg['x'], label='X', color='tab:blue', lw=1.5)
        ax.plot(uqps_seg.index, uqps_seg['y'], label='Y', color='tab:orange', lw=1.5)
        ax.plot(uqps_seg.index, uqps_seg['z'], label='Z', color='tab:green', lw=1.5)
        ax.legend(loc='upper right', ncol=3)
    else:
        ax.text(0.5, 0.5, "No UQPS Data", ha='center', va='center')
    ax.set_ylabel("Pos (m)")
    ax.set_title("1. UQPS Position", loc='left', fontweight='bold', fontsize=12)
    ax.grid(True, linestyle='--')

    # --- Row 2: DVL Velocity ---
    ax = axs[1]
    ax.plot(t, dvl_seg['vx(m/s)'], 'r', label='Vx', alpha=0.8)
    ax.plot(t, dvl_seg['vy(m/s)'], 'g', label='Vy', alpha=0.8)
    ax.plot(t, dvl_seg['vz(m/s)'], 'b', label='Vz', alpha=0.8)
    ax.set_ylabel("Vel (m/s)")
    ax.set_title("2. DVL Velocity", loc='left', fontweight='bold', fontsize=12)
    ax.legend(loc='upper right', ncol=3)
    ax.grid(True, linestyle='--')

    # --- Row 3: Altitude ---
    ax = axs[2]
    ax.plot(t, dvl_seg['altitude(m)'], 'k', label='Alt', alpha=0.8)
    ax.set_ylabel("Alt (m)")
    ax.set_title("3. DVL Altitude", loc='left', fontweight='bold', fontsize=12)
    ax.grid(True, linestyle='--')

    # --- Row 4: FOM ---
    ax = axs[3]
    ax.plot(t, dvl_seg['fom'], color='purple', label='FOM', alpha=0.8)
    ax.set_ylabel("FOM")
    ax.set_ylim(0, 0.001)  # Requirement: 0 - 0.001 limit
    ax.set_title("4. Figure of Merit (FOM)", loc='left', fontweight='bold', fontsize=12)
    ax.grid(True, linestyle='--')

    # --- Row 5: Raw Beam Velocity ---
    ax = axs[4]
    for i in range(4):
        col = f'id{i}_vel'
        if col in dvl_seg: ax.plot(t, dvl_seg[col], color=COLORS_ID[i], label=f'Beam {i}', alpha=0.6)
    ax.set_ylabel("Beam Vel (m/s)")
    ax.set_title("5. Raw Beam Velocity", loc='left', fontweight='bold', fontsize=12)
    ax.legend(loc='upper right', ncol=4)
    ax.grid(True, linestyle='--')

    # --- Row 6: Raw Beam Distance ---
    ax = axs[5]
    for i in range(4):
        col = f'id{i}_dist'
        if col in dvl_seg: ax.plot(t, dvl_seg[col], color=COLORS_ID[i], label=f'Beam {i}', alpha=0.6)
    ax.set_ylabel("Dist (m)")
    ax.set_title("6. Raw Beam Distance", loc='left', fontweight='bold', fontsize=12)
    ax.grid(True, linestyle='--')

    # --- Row 7: RSSI ---
    ax = axs[6]
    for i in range(4):
        col = f'id{i}_rssi'
        if col in dvl_seg: ax.plot(t, dvl_seg[col], color=COLORS_ID[i], label=f'Beam {i}', alpha=0.6)
    ax.set_ylabel("RSSI (dB)")
    ax.set_title("7. Signal Strength (RSSI)", loc='left', fontweight='bold', fontsize=12)
    ax.grid(True, linestyle='--')

    # --- Row 8: NSD ---
    ax = axs[7]
    for i in range(4):
        col = f'id{i}_nsd'
        if col in dvl_seg: ax.plot(t, dvl_seg[col], color=COLORS_ID[i], label=f'Beam {i}', alpha=0.6)
    ax.set_ylabel("NSD (dB)")
    ax.set_title("8. Noise Spectral Density (NSD)", loc='left', fontweight='bold', fontsize=12)
    ax.grid(True, linestyle='--')
    
    axs[-1].set_xlabel("Time (JST)")
    fig.autofmt_xdate()
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    out_path = os.path.join(OUT_DIR, f"Transducer_Analysis_Seg{seg_idx}_{time_str}.png")
    plt.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"[OK] Saved Segment {seg_idx} plot to: {out_path}")

# --- Main Program ---
if __name__ == "__main__":
    df_dvl_loaded, df_uqps_loaded = load_data()
    
    if not df_dvl_loaded.empty and not df_uqps_loaded.empty:
        segments = find_overlap_segments(df_uqps_loaded, df_dvl_loaded)
        print(f"[INFO] Found {len(segments)} segments.")
        
        for i, (s, e) in enumerate(segments, 1):
            print(f"Processing segment {i}...")
            # Slice data
            d_slice = df_dvl_loaded.loc[s:e]
            u_slice = df_uqps_loaded.loc[s:e]
            plot_deep_dive_segment(d_slice, u_slice, i)
            
        print("[DONE] All plots generated.")
    else:
        print("[ERROR] Data loading failed. Check inputs.")
