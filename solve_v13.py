import warnings; warnings.filterwarnings('ignore')
import re
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize as _opt_minimize
from sklearn.metrics import mean_squared_error

# ── Configuration ──────────────────────────────────────────────────────────────
INPUT_CSV        = 'dataset.csv'
OUTPUT_FILLED    = 'filled_dataset.csv'
OUTPUT_SUBMISSION = 'submission.csv'
OPTION_EXPIRY          = pd.Timestamp('2026-01-27 15:30')
MIN_PER_TRADING_YEAR   = 252 * 390   # 98,280 trading minutes / year
BUCKET_SPLIT           = 5 * 390     # 1950 min  — normal vs mid-expiry boundary
BUCKET_FINAL           = 60          # 60 min    — final-stretch boundary
OPT_SEEDS              = [42, 0]     # 2-seed averaging
NUM_RESTARTS           = 25

# ── Load Data ──────────────────────────────────────────────────────────────────
raw_df = pd.read_csv(INPUT_CSV)
raw_df['datetime'] = pd.to_datetime(raw_df['datetime'], format='%d-%m-%Y %H:%M')

all_option_cols = [c for c in raw_df.columns if re.match(r'NIFTY', c)]
call_cols       = sorted([c for c in all_option_cols if 'CE' in c])
put_cols        = sorted([c for c in all_option_cols if 'PE' in c])
call_strikes    = [int(re.search(r'(\d{5})CE', c).group(1)) for c in call_cols]
put_strikes     = [int(re.search(r'(\d{5})PE', c).group(1)) for c in put_cols]

raw_df['tte_minutes'] = (OPTION_EXPIRY - raw_df['datetime']).dt.total_seconds() / 60

print(f'Dataset shape : {raw_df.shape}')
print(f'Option columns: {len(all_option_cols)} ({len(call_cols)} CE, {len(put_cols)} PE)')
print(f'Missing values: {raw_df[all_option_cols].isna().sum().sum()} / {raw_df[all_option_cols].size}')

# ── Smile Fill Methods ─────────────────────────────────────────────────────────

def smile_poly2(iv_row, strike_list):
    """Quadratic polynomial fit: IV ~ a + b*K + c*K²"""
    observed = [(k, v) for k, v in zip(strike_list, iv_row.values) if not np.isnan(v)]
    if len(observed) < 2:
        return iv_row
    ks, vs = zip(*observed)
    coeffs = np.polyfit(ks, vs, min(2, len(observed) - 1))
    if len(coeffs) == 2:
        coeffs = [0] + list(coeffs)
    result = iv_row.copy()
    for k in strike_list:
        if np.isnan(iv_row[k]):
            result[k] = max(np.polyval(coeffs, k), 1e-4)
    return result


def smile_poly2_var(iv_row, strike_list):
    """Quadratic fit in variance space: IV² ~ a + b*K + c*K²"""
    observed = [(k, v) for k, v in zip(strike_list, iv_row.values) if not np.isnan(v)]
    if len(observed) < 2:
        return smile_poly2(iv_row, strike_list)
    ks, vs = zip(*observed)
    variances = [v ** 2 for v in vs]
    coeffs = np.polyfit(ks, variances, min(2, len(observed) - 1))
    if len(coeffs) == 2:
        coeffs = [0] + list(coeffs)
    result = iv_row.copy()
    for k in strike_list:
        if np.isnan(iv_row[k]):
            result[k] = max(np.sqrt(max(np.polyval(coeffs, k), 1e-8)), 1e-4)
    return result


def smile_pchip(iv_row, strike_list):
    """Monotone cubic spline (PCHIP) with extrapolation"""
    observed = [(k, v) for k, v in zip(strike_list, iv_row.values) if not np.isnan(v)]
    if len(observed) < 2:
        return smile_poly2(iv_row, strike_list)
    ks, vs = zip(*observed)
    spline_fn = PchipInterpolator(ks, vs, extrapolate=True) if len(observed) >= 3 \
        else lambda x: float(np.interp(x, ks, vs))
    result = iv_row.copy()
    for k in strike_list:
        if np.isnan(iv_row[k]):
            result[k] = max(float(spline_fn(k)), 1e-4)
    return result


def smile_adaptive(iv_row, strike_list):
    """PCHIP for interior missing strikes; poly2 for wing extrapolation"""
    obs_strikes = [k for k, v in zip(strike_list, iv_row.values) if not np.isnan(v)]
    if not obs_strikes:
        return iv_row
    lo_strike, hi_strike = min(obs_strikes), max(obs_strikes)
    result = iv_row.copy()
    interior_missing = [k for k in strike_list if np.isnan(iv_row[k]) and lo_strike <= k <= hi_strike]
    wing_missing     = [k for k in strike_list if np.isnan(iv_row[k]) and (k < lo_strike or k > hi_strike)]
    if interior_missing:
        filled_pchip = smile_pchip(iv_row, strike_list)
        for k in interior_missing:
            result[k] = filled_pchip[k]
    if wing_missing:
        filled_poly = smile_poly2(iv_row, strike_list)
        for k in wing_missing:
            result[k] = filled_poly[k]
    return result


def smile_logwing(iv_row, strike_list):
    """PCHIP interior; log-linear (geometric) wing extrapolation"""
    observed = [(k, v) for k, v in zip(strike_list, iv_row.values) if not np.isnan(v) and v > 0]
    if len(observed) < 2:
        return smile_poly2(iv_row, strike_list)
    obs_strikes = [k for k, v in observed]
    lo_strike, hi_strike = min(obs_strikes), max(obs_strikes)
    result = smile_pchip(iv_row, strike_list)
    lo_iv = iv_row[lo_strike]
    hi_iv = iv_row[hi_strike]
    near_lo = sorted([(k, v) for k, v in observed if lo_strike <= k <= lo_strike + 100])
    near_hi = sorted([(k, v) for k, v in observed if hi_strike - 100 <= k <= hi_strike])
    slope_lo = np.clip(
        (np.log(near_lo[-1][1]) - np.log(near_lo[0][1])) /
        (near_lo[-1][0] - near_lo[0][0] + 1e-9), -0.02, 0.02
    ) if len(near_lo) >= 2 else 0.0
    slope_hi = np.clip(
        (np.log(near_hi[-1][1]) - np.log(near_hi[0][1])) /
        (near_hi[-1][0] - near_hi[0][0] + 1e-9), -0.02, 0.02
    ) if len(near_hi) >= 2 else 0.0
    for k in strike_list:
        if np.isnan(iv_row[k]):
            if k < lo_strike:
                result[k] = max(lo_iv * np.exp(slope_lo * (k - lo_strike)), 1e-4)
            elif k > hi_strike:
                result[k] = max(hi_iv * np.exp(slope_hi * (k - hi_strike)), 1e-4)
    return result


def smile_totvar(iv_row, strike_list, tte_min):
    """PCHIP in total-variance space: w = IV² × TTE. Stable near expiry."""
    observed = [(k, v) for k, v in zip(strike_list, iv_row.values) if not np.isnan(v) and v > 0]
    if len(observed) < 2:
        return smile_poly2(iv_row, strike_list)
    T_years = max(tte_min / MIN_PER_TRADING_YEAR, 1e-7)
    ks = [k for k, v in observed]
    tot_vars = [v ** 2 * T_years for k, v in observed]
    spline_fn = PchipInterpolator(ks, tot_vars, extrapolate=True) if len(observed) >= 3 \
        else lambda x: float(np.interp(x, ks, tot_vars))
    result = iv_row.copy()
    for k in strike_list:
        if np.isnan(iv_row[k]):
            result[k] = max(np.sqrt(max(float(spline_fn(k)), 1e-10) / T_years), 1e-4)
    return result


def run_fill(data_df, fill_fn):
    """Apply a cross-sectional fill function across all rows and both option types."""
    out_df = data_df.copy()
    for cols, strikes in [(call_cols, call_strikes), (put_cols, put_strikes)]:
        pivot = out_df[cols].copy()
        pivot.columns = strikes
        for row_idx in pivot.index:
            pivot.loc[row_idx] = fill_fn(pivot.loc[row_idx], strikes)
        out_df[cols] = pivot.values
    return out_df


def run_fill_with_tte(data_df, fill_fn):
    """Apply a TTE-aware fill function across all rows and both option types."""
    out_df = data_df.copy()
    for cols, strikes in [(call_cols, call_strikes), (put_cols, put_strikes)]:
        pivot = out_df[cols].copy()
        pivot.columns = strikes
        for row_idx in pivot.index:
            pivot.loc[row_idx] = fill_fn(pivot.loc[row_idx], strikes, data_df.loc[row_idx, 'tte_minutes'])
        out_df[cols] = pivot.values
    return out_df


METHOD_KEYS = ['p2', 'var', 'ph', 'adp', 'lgw', 'tv']

# ── Stratified CV Mask (seed=42, fixed) ────────────────────────────────────────
np.random.seed(42)
col_missing_rate = {col: raw_df[col].isna().mean() for col in all_option_cols}

masked_df = raw_df.copy()
cv_positions = []
for col in all_option_cols:
    present_idx = raw_df.index[~raw_df[col].isna()].tolist()
    mask_rate   = np.clip(col_missing_rate[col], 0.05, 0.50)
    n_to_mask   = max(1, int(len(present_idx) * mask_rate))
    selected    = np.random.choice(present_idx, size=n_to_mask, replace=False)
    for row_idx in selected:
        masked_df.loc[row_idx, col] = np.nan
        cv_positions.append((row_idx, col))

cv_true_ivs  = np.array([raw_df.loc[i, c] for i, c in cv_positions])
cv_tte_vals  = np.array([raw_df.loc[i, 'tte_minutes'] for i, c in cv_positions])

final_mask  = np.where(cv_tte_vals <= BUCKET_FINAL)[0]
mid_mask    = np.where((cv_tte_vals > BUCKET_FINAL) & (cv_tte_vals <= BUCKET_SPLIT))[0]
normal_mask = np.where(cv_tte_vals > BUCKET_SPLIT)[0]

print(f'\nStratified CV: {len(cv_positions):,} positions')
print(f'  normal        (TTE > 1950) : {len(normal_mask)}')
print(f'  mid-expiry    (60<TTE<=1950): {len(mid_mask)}')
print(f'  final-stretch (TTE <= 60)  : {len(final_mask)}')

# ── Run All 6 CV Fills ─────────────────────────────────────────────────────────
print('\nRunning CV fills (6 methods)...')

def extract_cv_preds(filled_df):
    return np.array([filled_df.loc[i, c] for i, c in cv_positions])

cv_p2  = extract_cv_preds(run_fill(masked_df.copy(), smile_poly2))
cv_var = extract_cv_preds(run_fill(masked_df.copy(), smile_poly2_var))
cv_ph  = extract_cv_preds(run_fill(masked_df.copy(), smile_pchip))
cv_adp = extract_cv_preds(run_fill(masked_df.copy(), smile_adaptive))
cv_lgw = extract_cv_preds(run_fill(masked_df.copy(), smile_logwing))
cv_tv  = extract_cv_preds(run_fill_with_tte(masked_df.copy(), smile_totvar))
print('Done.')

method_preds = {'p2': cv_p2, 'var': cv_var, 'ph': cv_ph, 'adp': cv_adp, 'lgw': cv_lgw, 'tv': cv_tv}

# ── QP Weight Optimisation with 3-Seed Averaging ──────────────────────────────

def optimise_single_seed(preds_by_method, ground_truth, seed, n_restarts=NUM_RESTARTS):
    """SLSQP with Dirichlet restarts; returns best weight dict and MSE."""
    method_list = list(preds_by_method.keys())
    pred_matrix = np.stack([preds_by_method[m] for m in method_list], axis=1)

    def objective(w):
        return mean_squared_error(ground_truth, pred_matrix @ w)

    eq_constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1}]
    weight_bounds  = [(0, 1)] * len(method_list)
    best_result    = None
    np.random.seed(seed)
    for _ in range(n_restarts):
        w_init  = np.random.dirichlet(np.ones(len(method_list)))
        opt_res = _opt_minimize(objective, w_init, method='SLSQP',
                                bounds=weight_bounds, constraints=eq_constraints,
                                options={'ftol': 1e-12, 'maxiter': 2000})
        if best_result is None or opt_res.fun < best_result.fun:
            best_result = opt_res
    return dict(zip(method_list, best_result.x)), best_result.fun


def optimise_averaged(preds_by_method, ground_truth, seeds=OPT_SEEDS):
    """Run QP with each seed and return the averaged weight vector (v13)."""
    seed_weights = []
    for seed in seeds:
        w_dict, mse_val = optimise_single_seed(preds_by_method, ground_truth, seed)
        print(f'    seed={seed}: MSE={mse_val:.8f}  weights={" ".join(f"{m}={v:.3f}" for m, v in w_dict.items())}')
        seed_weights.append(np.array([w_dict[m] for m in preds_by_method.keys()]))
    avg_weights = np.mean(seed_weights, axis=0)
    avg_weights = avg_weights / avg_weights.sum()   # normalise for floating-point safety
    return dict(zip(preds_by_method.keys(), avg_weights))


print('\nOptimising QP weights per bucket (3-seed average)...')

print('  [Normal bucket]')
W_normal = optimise_averaged({m: method_preds[m][normal_mask] for m in METHOD_KEYS}, cv_true_ivs[normal_mask])

print('  [Mid-expiry bucket]')
W_mid = optimise_averaged({m: method_preds[m][mid_mask] for m in METHOD_KEYS}, cv_true_ivs[mid_mask])

print('  [Final-stretch bucket]')
if len(final_mask) >= 20:
    W_final = optimise_averaged({m: method_preds[m][final_mask] for m in METHOD_KEYS}, cv_true_ivs[final_mask])
else:
    print('    (too few samples — copying mid-expiry weights)')
    W_final = W_mid.copy()

print('\nFinal averaged weights:')
for bucket_label, W_bucket in [('Normal       ', W_normal), ('Mid-expiry   ', W_mid), ('Final-stretch', W_final)]:
    print(f'  {bucket_label}: {" ".join(f"{m}={v:.3f}" for m, v in W_bucket.items())}')

# ── CV Blend MSE ───────────────────────────────────────────────────────────────
blend_preds = np.empty_like(cv_true_ivs)
blend_preds[normal_mask] = np.stack([method_preds[m][normal_mask] for m in METHOD_KEYS], axis=1) @ np.array([W_normal[m] for m in METHOD_KEYS])
blend_preds[mid_mask]    = np.stack([method_preds[m][mid_mask]    for m in METHOD_KEYS], axis=1) @ np.array([W_mid[m]    for m in METHOD_KEYS])
blend_preds[final_mask]  = np.stack([method_preds[m][final_mask]  for m in METHOD_KEYS], axis=1) @ np.array([W_final[m]  for m in METHOD_KEYS])
print(f'\nBlend CV MSE: {mean_squared_error(cv_true_ivs, blend_preds):.8f}')

# ── Full-Dataset Fills ─────────────────────────────────────────────────────────
print('\nBuilding 6 fills on full dataset...')
full_fills = {
    'p2' : run_fill(raw_df.copy(), smile_poly2),
    'var' : run_fill(raw_df.copy(), smile_poly2_var),
    'ph'  : run_fill(raw_df.copy(), smile_pchip),
    'adp' : run_fill(raw_df.copy(), smile_adaptive),
    'lgw' : run_fill(raw_df.copy(), smile_logwing),
    'tv'  : run_fill_with_tte(raw_df.copy(), smile_totvar),
}
print('Done.')

# ── Apply 3-Bucket Blend ───────────────────────────────────────────────────────
filled_df = raw_df.copy()
for col in all_option_cols:
    is_missing = raw_df[col].isna()
    if not is_missing.any():
        continue
    for row_idx in raw_df.index[is_missing]:
        tte_val      = raw_df.loc[row_idx, 'tte_minutes']
        W_active     = W_final if tte_val <= BUCKET_FINAL else (W_mid if tte_val <= BUCKET_SPLIT else W_normal)
        method_vals  = np.array([full_fills[m].loc[row_idx, col] for m in METHOD_KEYS])
        filled_df.loc[row_idx, col] = max(float(np.dot(method_vals, [W_active[m] for m in METHOD_KEYS])), 1e-4)

assert filled_df[all_option_cols].isna().sum().sum() == 0, 'Still missing values!'
print('\nAll missing values filled')

filled_df['datetime'] = filled_df['datetime'].dt.strftime('%d-%m-%Y %H:%M')
filled_df.to_csv(OUTPUT_FILLED, index=False)
print(f'Saved {OUTPUT_FILLED}')

# ── Generate submission.csv ────────────────────────────────────────────────────
filled_read = pd.read_csv(OUTPUT_FILLED)
filled_read['datetime'] = pd.to_datetime(filled_read['datetime'], format='%d-%m-%Y %H:%M')
original_read = pd.read_csv(INPUT_CSV)

submission_rows = []
for col in all_option_cols:
    for row_idx in original_read.index[original_read[col].isna()]:
        dt_str = filled_read.loc[row_idx, 'datetime'].strftime('%d-%m-%Y %H:%M')
        submission_rows.append({'id': f'{dt_str}||{col}', 'value': filled_read.loc[row_idx, col]})

submission_df = pd.DataFrame(submission_rows, columns=['id', 'value'])
submission_df = submission_df.sort_values('id').reset_index(drop=True)
submission_df.to_csv(OUTPUT_SUBMISSION, index=False)

print(f'\nsubmission.csv saved  ({len(submission_df)} rows)')
print(f'   Range:      [{submission_df.value.min():.6f}, {submission_df.value.max():.6f}]')
print(f'   Negatives:  {(submission_df.value < 0).sum()}')
print(f'   Floor hits: {(submission_df.value <= 1e-4).sum()}')
print(submission_df.head(8).to_string(index=False))
