import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import minimize

matplotlib.use("Agg")

# ========= Configuration: File Paths =========
# User needs to modify these paths
FILE_UQPS = r"PATH_TO_YOUR_UQPS_FILE.xlsx"
FILE_DVL  = r"PATH_TO_YOUR_DVL_FILE.xlsx"
OUT_DIR   = r"PATH_TO_OUTPUT_DIRECTORY"
# ---------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

# ========= Data Processing Parameters =========
START     = pd.Timestamp("2023-09-11 00:00:00")
END       = pd.Timestamp("2023-09-15 23:59:59")
RESAMPLE  = "1s"
GAP_TOL_S = 2
MIN_SEG_S = 120

# --- Plotting Parameters ---
UQ_DOT_S  = 9
DVL_LW    = 1.2
DPI       = 160

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
    
    # Handle missing milliseconds if any
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

# ---------- Data Loading ----------
def uqps_load_resample_and_presence(path):
    df   = read_table(path)
    tcol = find_col(df, ["datetime(JST)", "datetime", "time", "timestamp"])
    if tcol is None: raise ValueError("UQPS data: Time column not found.")
    dt   = to_jst_naive(df[tcol])
    cx   = find_col(df, ["Relative x", "rel_x", "pos_x", "x"])
    cy   = find_col(df, ["Relative y", "rel_y", "pos_y", "y"])
    cz   = find_col(df, ["Relative z", "rel_z", "pos_z", "z"])
    if not all([cx, cy, cz]): raise ValueError("UQPS data: Missing x/y/z columns.")
    
    raw = (df.assign(dt=dt).dropna(subset=["dt"]).set_index("dt")
           [[cx, cy, cz]].rename(columns={cx: "x", cy: "y", cz: "z"})
           .astype(float, errors="ignore").sort_index())
    raw = raw[(raw.index >= START) & (raw.index <= END)]
    
    present_sec = raw.dropna(how="all").index.floor("s")
    if len(present_sec): present_sec = present_sec.sort_values().unique()
    
    resampled = raw.resample(RESAMPLE).last()
    return resampled, pd.DatetimeIndex(present_sec)

def dvl_load_resample_and_presence(path):
    df   = read_table(path)
    df   = apply_valid_mask(df)
    tcol = find_col(df, ["datetime(JST)", "datetime", "time"])
    if tcol is None: raise ValueError("DVL data: Time column not found.")
    dt   = to_jst_naive(df[tcol])
    cx   = find_col(df, ["vx(m/s)", "vx"])
    cy   = find_col(df, ["vy(m/s)", "vy"])
    cz   = find_col(df, ["vz(m/s)", "vz"])
    if not all([cx, cy, cz]): raise ValueError("DVL data: Missing Vx/Vy/Vz columns.")
    
    raw = (df.assign(dt=dt).dropna(subset=["dt"]).set_index("dt")
           [[cx, cy, cz]].rename(columns={cx: "Vx", cy: "Vy", cz: "Vz"})
           .apply(pd.to_numeric, errors="coerce").sort_index())
    raw = raw[(raw.index >= START) & (raw.index <= END)]
    
    present_sec = raw.dropna(how="all").index.floor("s")
    if len(present_sec): present_sec = present_sec.sort_values().unique()
    
    return raw, pd.DatetimeIndex(present_sec)

# ---------- Segmentation ----------
def find_overlap_segments_by_presence(present_u, present_v, gap_tol_s=2, min_len_s=120):
    common = present_u.intersection(present_v)
    common = common[(common >= START) & (common <= END)]
    if common.empty: return []
    common = common.sort_values()
    dif = common.to_series().diff().dt.total_seconds().fillna(0)
    grp = (dif > gap_tol_s).cumsum()
    segs = []
    for _, g in common.to_series().groupby(grp):
        if len(g) >= min_len_s:
            segs.append((g.iloc[0], g.iloc[-1]))
    return segs

def get_rotation_matrix(roll, pitch, yaw):
    roll_rad, pitch_rad, yaw_rad = np.radians(roll), np.radians(pitch), np.radians(yaw)
    cr, sr = np.cos(roll_rad), np.sin(roll_rad)
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def apply_rotation_and_integrate(angles, v_data, u_data, start_pos_override=None):
    R = get_rotation_matrix(angles['roll'], angles['pitch'], angles['yaw'])
    v_vectors = v_data[['Vx', 'Vy', 'Vz']].to_numpy()
    v_rotated_vectors = (R @ v_vectors.T).T
    v_rotated = pd.DataFrame(v_rotated_vectors, index=v_data.index, columns=['Vx', 'Vy', 'Vz'])
    
    start_pos = start_pos_override
    if start_pos is None:
        start_pos = u_data.iloc[0] if not u_data.empty and u_data.iloc[0].notna().all() else pd.Series({'x': 0, 'y': 0, 'z': 0})

    disp_aligned = v_rotated.copy()
    for col_v, col_u in [("Vx", "x"), ("Vy", "y"), ("Vz", "z")]:
        vv = v_rotated[col_v].to_numpy(dtype=float)
        ts_seconds = v_rotated.index.to_numpy().astype('datetime64[ns]').astype(np.int64) / 1e9
        if len(ts_seconds) < 2:
            disp_aligned[col_v] = start_pos[col_u]
            continue
        dt = np.diff(ts_seconds)
        trapezoids = (vv[1:] + vv[:-1]) * 0.5 * dt
        integ = np.zeros_like(vv, dtype=float)
        integ[1:] = np.cumsum(trapezoids)
        valid_mask = ~np.isnan(integ)
        integ = np.interp(ts_seconds, ts_seconds[valid_mask], integ[valid_mask])
        disp_aligned[col_v] = integ + start_pos[col_u]
        
    return disp_aligned, v_rotated

def objective_vz_minimization(angles_rp, v_data_debiased):
    """Minimize vertical velocity component (assuming mostly flat motion)."""
    roll, pitch = angles_rp; R = get_rotation_matrix(roll, pitch, 0)
    v_vectors = v_data_debiased[['Vx', 'Vy', 'Vz']].to_numpy()
    v_rotated_vectors = (R @ v_vectors.T).T
    return np.sum(v_rotated_vectors[:, 2]**2)

def objective_x_rmse(angle_y, u_data, v_data_debiased, fixed_roll, fixed_pitch):
    """Minimize RMSE between UQPS X-position and DVL integrated X-position."""
    yaw, = angle_y; angles = {'roll': fixed_roll, 'pitch': fixed_pitch, 'yaw': yaw}
    d_disp_aligned, _ = apply_rotation_and_integrate(angles, v_data_debiased, u_data)
    d_resampled = d_disp_aligned.reindex(u_data.index).interpolate()
    valid_data = pd.concat([u_data['x'], d_resampled['Vx']], axis=1).dropna()
    if len(valid_data) < 2: return 1e9
    return np.sqrt(np.mean((valid_data['x'] - valid_data['Vx'])**2))

# ========= Main Process (Sequential Correction) =========
print("[INFO] Starting data loading...")
uqps_res, present_u = uqps_load_resample_and_presence(FILE_UQPS)
dvl_res,  present_v = dvl_load_resample_and_presence(FILE_DVL)
seg_ts_list = find_overlap_segments_by_presence(present_u, present_v, GAP_TOL_S, MIN_SEG_S)

# --- Parameters ---
TARGET_SEGMENT_INDEX = 7 
CHUNK_SECONDS = 15      # Size of chunks for independent correction
# ----------------

if len(seg_ts_list) > TARGET_SEGMENT_INDEX:
    seg_start, seg_end = seg_ts_list[TARGET_SEGMENT_INDEX]
    u_full_segment = uqps_res.reindex(pd.date_range(start=seg_start, end=seg_end, freq="1s")).interpolate(limit=GAP_TOL_S)
    v_full_segment = dvl_res[seg_start:seg_end].interpolate(limit_method='time', limit_area='inside')

    if not u_full_segment.empty and not v_full_segment.empty:
        print(f"[INFO] Starting final sequential correction for Segment {TARGET_SEGMENT_INDEX + 1}...")
        print(f"[INFO] Chunk size: {CHUNK_SECONDS}s.")

        final_trajectory = pd.DataFrame()
        final_velocities = pd.DataFrame()
        time_axis_angles = []
        collected_angles = []

        # Create time chunks
        time_chunks = pd.date_range(start=seg_start, end=seg_end, freq=f'{CHUNK_SECONDS}s')
        if seg_end not in time_chunks:
            time_chunks = time_chunks.append(pd.DatetimeIndex([seg_end]))

        for i in range(len(time_chunks) - 1):
            chunk_start = time_chunks[i]
            chunk_end = time_chunks[i+1]
            chunk_center = chunk_start + (chunk_end - chunk_start) / 2
            
            print(f"\n--- Processing chunk {i+1}/{len(time_chunks)-1}: {chunk_start.time()} to {chunk_end.time()} ---")

            u_chunk = u_full_segment[chunk_start:chunk_end]
            v_chunk = v_full_segment[chunk_start:chunk_end]
            if u_chunk.empty or v_chunk.empty or len(v_chunk) < 5: 
                print("Skipping chunk due to insufficient data.")
                continue

            # Step 1: Remove Vz bias only
            v_pre_corrected = v_chunk.copy()
            vz_bias = v_chunk['Vz'].mean()
            v_pre_corrected['Vz'] = v_chunk['Vz'] - vz_bias
            
            # Step 2a: Leveling (Optimize Roll/Pitch)
            initial_rp = [180.0, 0.0]
            res_rp = minimize(objective_vz_minimization, initial_rp, args=(v_pre_corrected,), method='Nelder-Mead', options={'xatol':1e-3, 'fatol':1e-3})
            optimal_roll, optimal_pitch = res_rp.x
            
            # Step 2b: Heading Alignment (Optimize Yaw)
            initial_yaw = [0.0]
            # args=(..., optimal_roll, optimal_pitch) passes the leveled angles to the yaw optimizer
            res_y = minimize(objective_x_rmse, initial_yaw, args=(u_chunk, v_pre_corrected, optimal_roll, optimal_pitch), method='Nelder-Mead', options={'xatol':1e-3, 'fatol':1e-3})
            optimal_yaw, = res_y.x
            
            print(f"Found Angles: Roll={optimal_roll:.2f}, Pitch={optimal_pitch:.2f}, Yaw={optimal_yaw:.2f}")
            optimal_angles = {'roll': optimal_roll, 'pitch': optimal_pitch, 'yaw': optimal_yaw}
            time_axis_angles.append(chunk_center)
            collected_angles.append([optimal_roll, optimal_pitch, optimal_yaw])
            
            # Determine start position for this chunk (continuous trajectory)
            if final_trajectory.empty:
                start_pos_override = None
            else:
                start_pos_override = final_trajectory.iloc[-1]
                start_pos_override = start_pos_override.rename({'Vx': 'x', 'Vy': 'y', 'Vz': 'z'})

            # Final integration for this chunk
            corrected_traj_chunk, corrected_vel_chunk = apply_rotation_and_integrate(optimal_angles, v_pre_corrected, u_chunk, start_pos_override)

            # Append to final results
            if final_trajectory.empty:
                final_trajectory = corrected_traj_chunk
                final_velocities = corrected_vel_chunk
            else:
                # Concatenate (drop duplicate boundary point to ensure smoothness)
                final_trajectory = pd.concat([final_trajectory.iloc[:-1], corrected_traj_chunk])
                final_velocities = pd.concat([final_velocities.iloc[:-1], corrected_vel_chunk])
        
        print("\n[INFO] Sequential correction finished. Plotting...")
        
        day = seg_start.date()
        fig, axs = plt.subplots(5, 1, figsize=(15, 18), sharex=True)
        
        # Plot Position X, Y, Z
        for ax, axis in zip(axs[:3], ["x", "y", "z"]):
            ax.scatter(u_full_segment.index, u_full_segment[axis], s=UQ_DOT_S, color="tab:blue", alpha=0.7, label=f"UQPS {axis}")
            vcol = {"x": "Vx", "y": "Vy", "z": "Vz"}[axis]
            ax.plot(final_trajectory.index, final_trajectory[vcol], color="tab:red", lw=DVL_LW, alpha=0.9, label=f"DVL∫")
            ax.set_ylabel(f"Position {axis} (m)")
            ax.legend(fontsize=9, loc="best")
            ax.grid(True, linestyle="--", alpha=0.6)
            
        # Plot Velocities
        ax_vel = axs[3]
        ax_vel.scatter(final_velocities.index, final_velocities['Vx'], s=1, alpha=0.8, label="DVL Vx (Corrected)")
        ax_vel.scatter(final_velocities.index, final_velocities['Vy'], s=1, alpha=0.8, label="DVL Vy (Corrected)")
        ax_vel.scatter(final_velocities.index, final_velocities['Vz'], s=1, alpha=0.8, label="DVL Vz (Corrected)")
        ax_vel.set_ylabel("Velocity (m/s)")
        ax_vel.legend(fontsize=9, loc="best", markerscale=5)
        ax_vel.grid(True, linestyle="--", alpha=0.6)
        
        # Plot Estimated Angles
        ax_ang = axs[4]
        if time_axis_angles:
            angles_df = pd.DataFrame(collected_angles, index=time_axis_angles, columns=['Roll', 'Pitch', 'Yaw'])
            ax_ang.plot(angles_df.index, angles_df['Roll'], 'o-', label='Roll Angle')
            ax_ang.plot(angles_df.index, angles_df['Pitch'], 's-', label='Pitch Angle')
            ax_ang.plot(angles_df.index, angles_df['Yaw'], '^-', label='Yaw Angle')
        ax_ang.set_ylabel("Angle (deg)")
        ax_ang.legend(fontsize=9, loc="best")
        ax_ang.grid(True, linestyle="--", alpha=0.6)

        axs[-1].set_xlabel("Time (JST)")
        fig.suptitle(f"UQPS vs DVL (Final Corrected Model) - Segment {TARGET_SEGMENT_INDEX + 1}")
        fig.autofmt_xdate()
        
        png = os.path.join(OUT_DIR, safe_name(f"UQPS_vs_DVL_Seg{TARGET_SEGMENT_INDEX + 1}.png"))
        fig.savefig(png, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        
        print(f"[OK] Final plot saved: {png}")
        print("[DONE] Analysis finished.")
    else:
        print("[ERROR] Not enough data in the selected segment.")
else:
    print("[ERROR] Cannot run analysis. Not enough segments found.")
