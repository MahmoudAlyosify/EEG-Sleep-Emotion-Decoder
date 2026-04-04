
# ============================================================================
# EEG Emotional Memory — Unified Runner (v10.1-hybrid)
# ============================================================================
# Purpose:
#   Run THREE trial-level architectures in one script using shared preprocessing:
#   1) CORE   : Claude-style trial-level ensemble (LDA + LGBM + LR [+ MDM])
#   2) MOE    : EEG-specific Mixture-of-Experts (ERP + SPEC + RIEM + Sentinel)
#   3) HYBRID : Claude core architecture shift + EEG-specific expert routing
#
# Output:
#   - Separate LOSO results per model
#   - Separate submission CSV per model
#   - Shared preprocessing/cache to reduce redundant work
#
# Notes:
#   - Trial-level probability is tiled across all 200 timepoints.
#   - No GRU / no per-timepoint TRE / no ITPC tiling.
#   - Fold-safe ERP template (no LOSO leakage).
# ============================================================================

import os, re, time, gc, warnings, subprocess, sys, logging
from pathlib import Path
import multiprocessing

os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['TORCHINDUCTOR_DISABLE'] = '1'
logging.getLogger("torch._dynamo").setLevel(logging.CRITICAL)
logging.getLogger("torch._inductor").setLevel(logging.CRITICAL)
warnings.filterwarnings('ignore')

for pkg in ['lightgbm', 'tqdm', 'pyriemann']:
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', pkg, '-q'], check=False)
    except Exception:
        pass

try:
    import torch
    HAVE_TORCH = True
    TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
except Exception:
    HAVE_TORCH = False
    TORCH_DEVICE = None

import numpy as np
import pandas as pd
import h5py
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import butter, sosfiltfilt, hilbert, detrend as sp_detrend
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold

try:
    from pyriemann.classification import MDM
    HAVE_PYRIEMANN = True
except Exception:
    HAVE_PYRIEMANN = False

GPU_AVAILABLE = torch.cuda.is_available() if HAVE_TORCH else False
N_JOBS = multiprocessing.cpu_count()
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
FS = 200
N_TP = 200
N_CH = 16
ERP_WIN_START = int(0.100 * FS)   # 20
ERP_WIN_END   = int(0.350 * FS)   # 70
BASE_WIN_START = 0
BASE_WIN_END   = int(0.100 * FS)  # 20
LATE_WIN_START = ERP_WIN_END      # 70
LATE_WIN_END   = int(0.700 * FS)  # 140

BANDS = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'sigma': (12.0, 16.0),
    'beta':  (13.0, 30.0),
}
BAND_NAMES = list(BANDS.keys())

CHANNELS = ['c3','c4','o1','o2','cp3','f3','f4','cp4',
            'c5','cz','c6','cp5','p7','pz','p8','cp6']
CH = {c: i for i, c in enumerate(CHANNELS)}
ASYM_PAIRS = [('c3','c4'), ('f3','f4'), ('cp3','cp4'), ('p7','p8')]
ASYM_BANDS = ['theta', 'alpha', 'beta']
CONN_PAIRS = [
    ('f3','pz'), ('f4','pz'), ('f3','cz'), ('f4','cz'),
    ('c3','c4'), ('cp3','cp4'), ('f3','f4'), ('cz','pz'),
    ('f3','cp4'), ('f4','cp3')
]
CONN_BANDS = ['theta', 'alpha', 'sigma']

# Paths
BASE = r'D:\EEG Project\Project Overview and Specifications\eeg_competition'
EMO_DIR = os.path.join(BASE, 'training', 'sleep_emo')
NEU_DIR = os.path.join(BASE, 'training', 'sleep_neu')
TEST_DIR = os.path.join(BASE, 'testing')
OUT_DIR = os.path.join(BASE, 'hybrid_outputs_v10_1')
os.makedirs(OUT_DIR, exist_ok=True)

# Run control
RUN_LOSO = True
GENERATE_SUBMISSIONS = True
MODELS_TO_RUN = ['core', 'moe', 'hybrid']   # subset if needed

# Model knobs
K_BEST_LDA = 150
K_BEST_SPEC = 180
K_BEST_SENT = 220
LGBM_PARAMS = dict(
    n_estimators=350 if GPU_AVAILABLE else 250,
    num_leaves=31,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.80,
    colsample_bytree=0.70,
    min_child_samples=5,
    class_weight='balanced',
    reg_alpha=0.1,
    reg_lambda=1.0,
    n_jobs=4,
    random_state=42,
    verbose=-1,
)
if GPU_AVAILABLE:
    LGBM_PARAMS.update({'device': 'gpu', 'gpu_use_dp': False, 'max_bin': 255})
else:
    LGBM_PARAMS.update({'device': 'cpu'})

print(f"GPU_AVAILABLE={GPU_AVAILABLE} HAVE_TORCH={HAVE_TORCH} TORCH_DEVICE={TORCH_DEVICE}")
print(f"PyRiemann={'YES' if HAVE_PYRIEMANN else 'NO'} | MODELS_TO_RUN={MODELS_TO_RUN}")
print(f"Outputs → {OUT_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_field(f, grp, key):
    field = grp[key]
    if isinstance(field, h5py.Dataset):
        val = field[()]
        if isinstance(val, h5py.Reference):
            return np.array(f[val])
        if hasattr(val, 'shape') and val.shape == (1, 1):
            ref = val.item()
            if isinstance(ref, h5py.Reference):
                return np.array(f[ref])
        return np.array(val)
    return np.array(field)


def load_mat(path: str, label_override: int = None) -> dict:
    with h5py.File(str(path), 'r') as f:
        grp = None
        if 'data' in f:
            grp = f['data']
        else:
            for k in f.keys():
                if hasattr(f[k], 'keys') and 'trial' in f[k]:
                    grp = f[k]
                    break
        if grp is None:
            raise ValueError(f"Cannot find 'data' struct in {path}")
        trial_raw = _resolve_field(f, grp, 'trial')
        if trial_raw.ndim == 3:
            sh = trial_raw.shape
            if sh[2] == N_CH and sh[1] == N_TP:
                trial_raw = trial_raw.transpose(0,2,1)
            elif sh[0] == N_CH and sh[1] == N_TP:
                trial_raw = trial_raw.transpose(2,0,1)
            elif sh[0] == N_TP and sh[1] == N_CH:
                trial_raw = trial_raw.transpose(2,1,0)
        elif trial_raw.ndim == 2:
            trial_raw = trial_raw.T[np.newaxis]
        eeg = trial_raw.astype(np.float32)
        if label_override is not None:
            labels = np.full(eeg.shape[0], label_override, dtype=int)
        else:
            try:
                ti = _resolve_field(f, grp, 'trialinfo')
                if ti.ndim == 2 and ti.shape[0] == eeg.shape[0]:
                    labels = ti[:,0].astype(int)
                elif ti.ndim == 2 and ti.shape[1] == eeg.shape[0]:
                    labels = ti[0,:].astype(int)
                else:
                    labels = ti.flatten().astype(int)
            except Exception:
                labels = np.ones(eeg.shape[0], dtype=int)
        try:
            tv = _resolve_field(f, grp, 'time').flatten()
        except Exception:
            tv = np.arange(N_TP) / FS
        t_mask = tv >= -1e-6
        if np.any(~t_mask):
            tv = tv[t_mask]
            eeg = eeg[:, :, t_mask]
        if len(tv) != N_TP:
            tv = np.arange(N_TP) / FS
    return {'eeg': eeg, 'labels': labels, 'time': tv}


def load_all_training(emo_dir, neu_dir):
    emo_data, neu_data = {}, {}
    for fpath in sorted(Path(emo_dir).glob('*.mat')):
        d = load_mat(str(fpath), label_override=2)
        d['id'] = fpath.stem
        emo_data[fpath.stem] = d
    for fpath in sorted(Path(neu_dir).glob('*.mat')):
        d = load_mat(str(fpath), label_override=1)
        d['id'] = fpath.stem
        neu_data[fpath.stem] = d
    subjects = []
    for stem in sorted(set(emo_data.keys()) | set(neu_data.keys())):
        parts_eeg, parts_lbl = [], []
        if stem in emo_data:
            parts_eeg.append(emo_data[stem]['eeg'])
            parts_lbl.append(emo_data[stem]['labels'])
        if stem in neu_data:
            parts_eeg.append(neu_data[stem]['eeg'])
            parts_lbl.append(neu_data[stem]['labels'])
        merged = {
            'eeg': np.concatenate(parts_eeg, axis=0),
            'labels': np.concatenate(parts_lbl, axis=0),
            'time': (emo_data.get(stem) or neu_data.get(stem))['time'],
            'id': stem,
        }
        print(f" → {stem}: {merged['eeg'].shape[0]} trials (emo={(merged['labels']==2).sum()}, neu={(merged['labels']==1).sum()})")
        subjects.append(merged)
    print(f"\n✓ Training: {len(subjects)} subjects loaded")
    return subjects


def load_all_test(test_dir):
    subjects = []
    for fpath in sorted(Path(test_dir).glob('*.mat')):
        d = load_mat(str(fpath))
        nums = re.findall(r'\d+', fpath.stem)
        d['id'] = fpath.stem
        d['subj_id'] = int(nums[-1]) if nums else len(subjects)+1
        subjects.append(d)
    print(f"\n✓ Test: {len(subjects)} subjects | IDs: {[s['subj_id'] for s in subjects]}")
    return subjects

# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def bandpass(data, lo, hi, fs=FS, order=4):
    nyq = fs / 2.0
    sos = butter(order, [max(lo/nyq, 1e-4), min(hi/nyq, 0.9999)], btype='band', output='sos')
    return sosfiltfilt(sos, data, axis=-1).astype(np.float32)


def preprocess_trial(raw_trial):
    x = sp_detrend(raw_trial.astype(np.float64), axis=-1).astype(np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    x = bandpass(x, 0.5, 40.0, FS, 4)
    return x


def zscore_subject(eeg):
    mu = eeg.mean(axis=(0,2), keepdims=True)
    sig = eeg.std(axis=(0,2), keepdims=True) + 1e-8
    return ((eeg - mu) / sig).astype(np.float32), mu, sig


def euclidean_alignment(eeg):
    n_tr, n_ch, n_tp = eeg.shape
    covs = np.zeros((n_tr, n_ch, n_ch), dtype=np.float64)
    for i in range(n_tr):
        x = eeg[i].astype(np.float64)
        x = x - x.mean(axis=1, keepdims=True)
        covs[i] = (x @ x.T) / max(n_tp-1, 1)
    R_mean = covs.mean(axis=0)
    eps = 1e-6 * np.trace(R_mean) / n_ch
    R_mean += eps * np.eye(n_ch)
    try:
        vals, vecs = np.linalg.eigh(R_mean)
        vals = np.maximum(vals, 1e-10)
        R_inv_sqrt = vecs @ np.diag(vals**(-0.5)) @ vecs.T
    except np.linalg.LinAlgError:
        return eeg.astype(np.float32)
    out = np.zeros_like(eeg, dtype=np.float32)
    for i in range(n_tr):
        out[i] = (R_inv_sqrt @ eeg[i].astype(np.float64)).astype(np.float32)
    return out


def log_euclidean_cov_features(seg):
    n_ch = seg.shape[0]
    n_feat = n_ch * (n_ch + 1) // 2
    if seg.shape[1] < 3:
        return np.zeros(n_feat, dtype=np.float32)
    seg_c = seg.astype(np.float64)
    seg_c = seg_c - seg_c.mean(axis=1, keepdims=True)
    C = (seg_c @ seg_c.T) / max(seg_c.shape[1]-1, 1)
    eps = 1e-6 * (np.trace(C) / n_ch + 1e-12)
    C += eps * np.eye(n_ch)
    try:
        vals, vecs = np.linalg.eigh(C)
        vals = np.maximum(vals, 1e-10)
        logC = vecs @ np.diag(np.log(vals)) @ vecs.T
        return logC[np.triu_indices(n_ch)].astype(np.float32)
    except Exception:
        return np.zeros(n_feat, dtype=np.float32)


def compute_covariance_matrices(eeg_proc, t0=ERP_WIN_START, t1=ERP_WIN_END):
    n_tr = eeg_proc.shape[0]
    covs = np.zeros((n_tr, N_CH, N_CH), dtype=np.float64)
    for i in range(n_tr):
        seg = eeg_proc[i, :, t0:t1].astype(np.float64)
        seg -= seg.mean(axis=1, keepdims=True)
        C = (seg @ seg.T) / max(seg.shape[1]-1, 1)
        eps = 1e-6 * (np.trace(C) / N_CH + 1e-12)
        covs[i] = C + eps*np.eye(N_CH)
    return covs


def frontal_theta_asymmetry(theta_power, t0, t1):
    f3t = np.mean(theta_power[CH['f3'], t0:t1]) + 1e-12
    f4t = np.mean(theta_power[CH['f4'], t0:t1]) + 1e-12
    pzt = np.mean(theta_power[CH['pz'], t0:t1]) + 1e-12
    czt = np.mean(theta_power[CH['cz'], t0:t1]) + 1e-12
    return [float(np.log(f4t)-np.log(f3t)), float(np.log((f3t+f4t)/2.0)-np.log(pzt)), float(np.log(czt))]

# ─────────────────────────────────────────────────────────────────────────────
# Shared cache + ERP template
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_subject_for_cache(subj):
    eeg = subj['eeg'].copy()
    labels = subj['labels']
    eeg, _, _ = zscore_subject(eeg)
    eeg = euclidean_alignment(eeg)
    eeg_proc = np.array([preprocess_trial(eeg[i]) for i in range(eeg.shape[0])], dtype=np.float32)
    emo_mask = labels == 2
    neu_mask = labels == 1
    return {
        'id': subj['id'],
        'eeg_proc': eeg_proc,
        'labels': labels.copy(),
        'erp_emo': eeg_proc[emo_mask].mean(axis=0) if emo_mask.sum() > 1 else None,
        'erp_neu': eeg_proc[neu_mask].mean(axis=0) if neu_mask.sum() > 1 else None,
        'covs': compute_covariance_matrices(eeg_proc),
    }


def build_cache(subjects):
    print("Building shared cache (preprocessing each subject once)...")
    out = []
    for i, s in enumerate(subjects):
        t0 = time.time()
        c = preprocess_subject_for_cache(s)
        out.append(c)
        print(f" [{i+1:2d}/{len(subjects)}] {s['id']}: eeg_proc={c['eeg_proc'].shape} ({time.time()-t0:.1f}s)")
    return out


def compute_fold_erp_template(cache, exclude_idx, t0=ERP_WIN_START, t1=ERP_WIN_END):
    emo_list, neu_list = [], []
    for j, c in enumerate(cache):
        if j == exclude_idx:
            continue
        if c['erp_emo'] is not None:
            emo_list.append(c['erp_emo'])
        if c['erp_neu'] is not None:
            neu_list.append(c['erp_neu'])
    if not emo_list or not neu_list:
        return np.zeros((N_CH, t1-t0), dtype=np.float32)
    grand_emo = np.mean(emo_list, axis=0)
    grand_neu = np.mean(neu_list, axis=0)
    return (grand_emo - grand_neu)[:, t0:t1].astype(np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# CSP
# ─────────────────────────────────────────────────────────────────────────────
class CSP:
    def __init__(self, n_components=6):
        self.n_components = n_components
        self.filters_ = None
    def fit(self, eeg, labels):
        mask1 = labels == 1
        mask2 = labels == 2
        if mask1.sum() < 2 or mask2.sum() < 2:
            self.filters_ = np.eye(N_CH, dtype=np.float32)[:self.n_components]
            return self
        def cov(X):
            Xc = X - X.mean(axis=2, keepdims=True)
            return np.mean([x @ x.T / max(x.shape[1]-1,1) for x in Xc], axis=0) + 1e-8*np.eye(N_CH)
        C1 = cov(eeg[mask1].astype(np.float64))
        C2 = cov(eeg[mask2].astype(np.float64))
        try:
            eigvals, eigvecs = eigh(C1, C1 + C2)
            idx = np.argsort(np.abs(eigvals - 0.5))[::-1]
            self.filters_ = eigvecs[:, idx[:self.n_components]].T.astype(np.float32)
        except Exception:
            self.filters_ = np.eye(N_CH, dtype=np.float32)[:self.n_components]
        return self
    def log_var_features(self, eeg):
        if self.filters_ is None:
            return np.zeros((eeg.shape[0], self.n_components), dtype=np.float32)
        out = []
        for i in range(eeg.shape[0]):
            proj = self.filters_ @ eeg[i].astype(np.float64)
            out.append(np.log(np.var(proj, axis=1) + 1e-12).astype(np.float32))
        return np.array(out, dtype=np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────
def _plv(x, y):
    if len(x) < 4 or np.var(x) < 1e-14 or np.var(y) < 1e-14:
        return 0.0
    try:
        return float(np.abs(np.mean(np.exp(1j*(np.angle(hilbert(x)) - np.angle(hilbert(y)))))))
    except Exception:
        return 0.0


def extract_trial_feature_blocks(trial, erp_template=None):
    a0, a1 = ERP_WIN_START, ERP_WIN_END
    b0, b1 = LATE_WIN_START, LATE_WIN_END
    s0, s1 = BASE_WIN_START, BASE_WIN_END

    bp_signals, bp_power = {}, {}
    for bname, (lo, hi) in BANDS.items():
        sig = bandpass(trial, lo, hi, FS)
        pwr = (np.abs(hilbert(sig.astype(np.float64), axis=-1))**2).astype(np.float32)
        bp_signals[bname] = sig
        bp_power[bname] = pwr

    # Core/full feature vector (Claude-style)
    core = []
    # A: log band power in ERP window
    for bname in BAND_NAMES:
        core.extend(np.log1p(np.mean(bp_power[bname][:, a0:a1], axis=1)).tolist())
    # B: baseline-corrected power
    for bname in BAND_NAMES:
        p_erp = np.mean(bp_power[bname][:, a0:a1], axis=1) + 1e-12
        p_bas = np.mean(bp_power[bname][:, s0:s1], axis=1) + 1e-12
        core.extend((np.log(p_erp) - np.log(p_bas)).tolist())
    # C: late window power
    for bname in BAND_NAMES:
        core.extend(np.log1p(np.mean(bp_power[bname][:, b0:b1], axis=1)).tolist())
    # D/E: ERP morphology + template corr
    erp_win = trial[:, a0:a1]
    core.extend(erp_win.max(axis=1).tolist())
    core.extend(erp_win.min(axis=1).tolist())
    core.extend((erp_win.max(axis=1) - erp_win.min(axis=1)).tolist())
    core.extend(np.sqrt(np.mean(erp_win**2, axis=1)).tolist())
    if erp_template is not None:
        for c in range(N_CH):
            x = erp_win[c]; tm = erp_template[c]
            if np.std(x) > 1e-8 and np.std(tm) > 1e-8:
                core.append(float(np.corrcoef(x, tm)[0,1]))
            else:
                core.append(0.0)
    else:
        core.extend([0.0]*N_CH)
    # F: frontal theta asymmetry
    core.extend(frontal_theta_asymmetry(bp_power['theta'], a0, a1))
    # G: asymmetry
    asym = []
    for ch1, ch2 in ASYM_PAIRS:
        for bn in ASYM_BANDS:
            p1 = np.mean(bp_power[bn][CH[ch1], a0:a1]) + 1e-12
            p2 = np.mean(bp_power[bn][CH[ch2], a0:a1]) + 1e-12
            asym.append(float(np.log(p2) - np.log(p1)))
    core.extend(asym)
    # H: PLV
    plv = []
    for bn in CONN_BANDS:
        sig = bp_signals[bn]
        for ch1, ch2 in CONN_PAIRS:
            plv.append(_plv(sig[CH[ch1], a0:a1], sig[CH[ch2], a0:a1]))
    core.extend(plv)
    # I: Riemannian proxy
    riem = log_euclidean_cov_features(bp_power['theta'][:, a0:a1])
    core.extend(riem.tolist())
    # J: sigma spindle 100-700ms
    sp0, sp1 = int(0.100*FS), int(0.700*FS)
    spindle = np.log1p(np.mean(bp_power['sigma'][:, sp0:sp1], axis=1))
    core.extend(spindle.tolist())

    # Expert-specific blocks (V9.3 style)
    erp_feat = []
    erp_feat.extend(erp_win.max(axis=1).tolist())
    erp_feat.extend(erp_win.min(axis=1).tolist())
    erp_feat.extend(erp_win.std(axis=1).tolist())
    erp_feat.extend((erp_win.max(axis=1) - erp_win.min(axis=1)).tolist())
    if erp_template is not None:
        for c in range(N_CH):
            x = erp_win[c]; tm = erp_template[c]
            if np.std(x) > 1e-8 and np.std(tm) > 1e-8:
                erp_feat.append(float(np.corrcoef(x, tm)[0,1]))
            else:
                erp_feat.append(0.0)
    else:
        erp_feat.extend([0.0]*N_CH)

    spec_feat = []
    # ERP band powers + baseline-corrected + late + asym + theta/alpha + sigma + FAI
    for bname in BAND_NAMES:
        spec_feat.extend(np.log1p(np.mean(bp_power[bname][:, a0:a1], axis=1)).tolist())
    for bname in BAND_NAMES:
        p_erp = np.mean(bp_power[bname][:, a0:a1], axis=1) + 1e-12
        p_bas = np.mean(bp_power[bname][:, s0:s1], axis=1) + 1e-12
        spec_feat.extend((np.log(p_erp) - np.log(p_bas)).tolist())
    for bname in BAND_NAMES:
        spec_feat.extend(np.log1p(np.mean(bp_power[bname][:, b0:b1], axis=1)).tolist())
    spec_feat.extend(asym)
    theta_pow = np.mean(bp_power['theta'][:, a0:a1], axis=1)
    alpha_pow = np.mean(bp_power['alpha'][:, a0:a1], axis=1)
    spec_feat.extend((np.log1p(theta_pow) - np.log1p(alpha_pow + 1e-12)).tolist())
    spec_feat.extend(np.log1p(np.mean(bp_power['sigma'][:, b0:b1], axis=1)).tolist())
    spec_feat.extend(frontal_theta_asymmetry(bp_power['theta'], a0, a1))

    sent_feat = np.concatenate([np.array(erp_feat, dtype=np.float32), np.array(spec_feat, dtype=np.float32), riem.astype(np.float32)], axis=0)

    return {
        'core': np.array(core, dtype=np.float32),
        'erp': np.array(erp_feat, dtype=np.float32),
        'spec': np.array(spec_feat, dtype=np.float32),
        'riem': np.array(riem, dtype=np.float32),
        'sent': np.array(sent_feat, dtype=np.float32),
    }


def extract_subject_blocks(eeg_proc, erp_template=None):
    out = {'core': [], 'erp': [], 'spec': [], 'riem': [], 'sent': []}
    for i in range(eeg_proc.shape[0]):
        fb = extract_trial_feature_blocks(eeg_proc[i], erp_template)
        for k in out:
            out[k].append(fb[k])
    for k in out:
        out[k] = np.vstack(out[k]).astype(np.float32)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────
class TabularBinaryModel:
    def __init__(self, model_type='lda', k_best=None):
        self.model_type = model_type
        self.k_best = k_best
        self.sel = None
        self.sc = RobustScaler()
        self.model = None
    def _make(self):
        if self.model_type == 'lda':
            return LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        if self.model_type == 'lr':
            return LogisticRegression(max_iter=2000, class_weight='balanced', C=0.5)
        if self.model_type == 'lgbm':
            return lgb.LGBMClassifier(**LGBM_PARAMS)
        raise ValueError(self.model_type)
    def fit(self, X, y):
        if self.k_best is not None and self.k_best < X.shape[1]:
            self.sel = SelectKBest(f_classif, k=self.k_best)
            Xs = self.sel.fit_transform(X, y)
        else:
            Xs = X
        Xs = self.sc.fit_transform(Xs)
        self.model = self._make()
        try:
            self.model.fit(Xs, y)
        except Exception:
            # lgbm fallback without GPU early stopping complexity
            self.model.fit(Xs, y)
        return self
    def predict_proba(self, X):
        Xs = self.sel.transform(X) if self.sel is not None else X
        Xs = self.sc.transform(Xs)
        try:
            return self.model.predict_proba(Xs)[:,1]
        except Exception:
            s = self.model.decision_function(Xs)
            return 1/(1+np.exp(-s))


class ClaudeCoreTrialClassifier:
    def __init__(self, use_mdm=HAVE_PYRIEMANN):
        self.lda = TabularBinaryModel('lda', k_best=K_BEST_LDA)
        self.lgbm = TabularBinaryModel('lgbm', k_best=None)
        self.lr = TabularBinaryModel('lr', k_best=None)
        self.use_mdm = use_mdm and HAVE_PYRIEMANN
        self.mdm = None
        self.weights = {'lda': 0.30, 'lgbm': 0.40, 'lr': 0.15, 'mdm': 0.15 if self.use_mdm else 0.0}
    def fit(self, X_core, y, covs=None):
        yb = (y==2).astype(int)
        self.lda.fit(X_core, yb)
        self.lgbm.fit(X_core, yb)
        self.lr.fit(X_core, yb)
        if self.use_mdm and covs is not None:
            self.mdm = MDM(metric='riemann')
            self.mdm.fit(covs, yb)
        return self
    def predict(self, X_core, covs=None):
        p_lda = self.lda.predict_proba(X_core)
        p_lgb = self.lgbm.predict_proba(X_core)
        p_lr = self.lr.predict_proba(X_core)
        out = {'p_lda': p_lda, 'p_lgbm': p_lgb, 'p_lr': p_lr}
        if self.use_mdm and self.mdm is not None and covs is not None:
            p = self.mdm.predict_proba(covs)
            classes = np.unique(self.mdm.classes_ if hasattr(self.mdm, 'classes_') else np.array([0,1]))
            idx1 = np.where(classes == 1)[0]
            p_mdm = p[:, idx1[0]] if len(idx1)>0 else p[:, -1]
            out['p_mdm'] = p_mdm
        else:
            out['p_mdm'] = None
        w = self.weights
        if out['p_mdm'] is not None:
            tot = w['lda']+w['lgbm']+w['lr']+w['mdm']
            p_final = (w['lda']*p_lda + w['lgbm']*p_lgb + w['lr']*p_lr + w['mdm']*out['p_mdm'])/tot
        else:
            tot = w['lda']+w['lgbm']+w['lr']
            p_final = (w['lda']*p_lda + w['lgbm']*p_lgb + w['lr']*p_lr)/tot
        out['p_final'] = p_final.astype(np.float32)
        return out


class MoETrialClassifier:
    def __init__(self):
        self.erp_expert = TabularBinaryModel('lda', k_best=96)
        self.spec_expert = TabularBinaryModel('lgbm', k_best=K_BEST_SPEC)
        self.riem_expert = TabularBinaryModel('lda', k_best=None)
        self.sentinel = TabularBinaryModel('lr', k_best=K_BEST_SENT)
        self.meta = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
        self.meta_sc = RobustScaler()
    @staticmethod
    def meta_features(p_sent, p_erp, p_spec, p_riem):
        P = np.vstack([p_sent, p_erp, p_spec, p_riem]).T
        mean_p = P.mean(axis=1)
        std_p = P.std(axis=1)
        mx = P.max(axis=1)
        mn = P.min(axis=1)
        disagree = np.abs(p_erp-p_spec)+np.abs(p_erp-p_riem)+np.abs(p_spec-p_riem)
        return np.column_stack([P, mean_p, std_p, mx, mn, disagree]).astype(np.float32)
    def fit(self, X_blocks, y):
        yb = (y==2).astype(int)
        self.erp_expert.fit(X_blocks['erp'], yb)
        self.spec_expert.fit(X_blocks['spec'], yb)
        self.riem_expert.fit(X_blocks['riem'], yb)
        self.sentinel.fit(X_blocks['sent'], yb)
        # internal OOF meta-train
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        oof_sent = np.zeros(len(yb), dtype=np.float32)
        oof_erp = np.zeros(len(yb), dtype=np.float32)
        oof_spec = np.zeros(len(yb), dtype=np.float32)
        oof_riem = np.zeros(len(yb), dtype=np.float32)
        for tr_idx, va_idx in skf.split(X_blocks['sent'], yb):
            e1 = TabularBinaryModel('lda', k_best=min(96, X_blocks['erp'].shape[1]))
            e2 = TabularBinaryModel('lgbm', k_best=min(K_BEST_SPEC, X_blocks['spec'].shape[1]))
            e3 = TabularBinaryModel('lda', k_best=None)
            s1 = TabularBinaryModel('lr', k_best=min(K_BEST_SENT, X_blocks['sent'].shape[1]))
            e1.fit(X_blocks['erp'][tr_idx], yb[tr_idx])
            e2.fit(X_blocks['spec'][tr_idx], yb[tr_idx])
            e3.fit(X_blocks['riem'][tr_idx], yb[tr_idx])
            s1.fit(X_blocks['sent'][tr_idx], yb[tr_idx])
            oof_erp[va_idx] = e1.predict_proba(X_blocks['erp'][va_idx])
            oof_spec[va_idx] = e2.predict_proba(X_blocks['spec'][va_idx])
            oof_riem[va_idx] = e3.predict_proba(X_blocks['riem'][va_idx])
            oof_sent[va_idx] = s1.predict_proba(X_blocks['sent'][va_idx])
        meta_X = self.meta_features(oof_sent, oof_erp, oof_spec, oof_riem)
        self.meta.fit(self.meta_sc.fit_transform(meta_X), yb)
        return self
    def predict(self, X_blocks):
        p_erp = self.erp_expert.predict_proba(X_blocks['erp'])
        p_spec = self.spec_expert.predict_proba(X_blocks['spec'])
        p_riem = self.riem_expert.predict_proba(X_blocks['riem'])
        p_sent = self.sentinel.predict_proba(X_blocks['sent'])
        meta_X = self.meta_sc.transform(self.meta_features(p_sent, p_erp, p_spec, p_riem))
        p_final = self.meta.predict_proba(meta_X)[:,1]
        return {'p_erp': p_erp, 'p_spec': p_spec, 'p_riem': p_riem, 'p_sent': p_sent, 'p_final': p_final.astype(np.float32)}


class HybridTrialClassifier:
    """Claude core probabilities + EEG expert routing."""
    def __init__(self, use_mdm=HAVE_PYRIEMANN):
        self.core = ClaudeCoreTrialClassifier(use_mdm=use_mdm)
        self.moe = MoETrialClassifier()
        self.meta = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
        self.meta_sc = RobustScaler()
    @staticmethod
    def hybrid_features(core_out, moe_out):
        p_core = core_out['p_final']
        p_moe = moe_out['p_final']
        p_lda = core_out['p_lda']
        p_lgbm = core_out['p_lgbm']
        p_lr = core_out['p_lr']
        p_mdm = core_out['p_mdm'] if core_out['p_mdm'] is not None else np.full_like(p_core, 0.5)
        p_erp = moe_out['p_erp']
        p_spec = moe_out['p_spec']
        p_riem = moe_out['p_riem']
        p_sent = moe_out['p_sent']
        diff = np.abs(p_core - p_moe)
        expert_std = np.std(np.vstack([p_erp, p_spec, p_riem, p_sent]).T, axis=1)
        core_std = np.std(np.vstack([p_lda, p_lgbm, p_lr, p_mdm]).T, axis=1)
        return np.column_stack([p_core, p_moe, p_lda, p_lgbm, p_lr, p_mdm, p_erp, p_spec, p_riem, p_sent, diff, expert_std, core_std]).astype(np.float32)
    def fit(self, X_blocks, y, covs=None):
        yb = (y==2).astype(int)
        self.core.fit(X_blocks['core'], y, covs=covs)
        self.moe.fit(X_blocks, y)
        # internal OOF for meta layer
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        meta_parts = []
        meta_y = []
        for tr_idx, va_idx in skf.split(X_blocks['core'], yb):
            core_i = ClaudeCoreTrialClassifier(use_mdm=(covs is not None and HAVE_PYRIEMANN))
            moe_i = MoETrialClassifier()
            tr_blocks = {k: X_blocks[k][tr_idx] for k in X_blocks}
            va_blocks = {k: X_blocks[k][va_idx] for k in X_blocks}
            core_i.fit(tr_blocks['core'], y[tr_idx], covs=covs[tr_idx] if covs is not None else None)
            moe_i.fit(tr_blocks, y[tr_idx])
            co = core_i.predict(va_blocks['core'], covs=covs[va_idx] if covs is not None else None)
            mo = moe_i.predict(va_blocks)
            meta_parts.append(self.hybrid_features(co, mo))
            meta_y.append(yb[va_idx])
        meta_X = np.vstack(meta_parts)
        meta_y = np.concatenate(meta_y)
        self.meta.fit(self.meta_sc.fit_transform(meta_X), meta_y)
        return self
    def predict(self, X_blocks, covs=None):
        co = self.core.predict(X_blocks['core'], covs=covs)
        mo = self.moe.predict(X_blocks)
        meta_X = self.meta_sc.transform(self.hybrid_features(co, mo))
        p_final = self.meta.predict_proba(meta_X)[:,1]
        return {
            'p_core': co['p_final'], 'p_moe': mo['p_final'],
            'p_lda': co['p_lda'], 'p_lgbm': co['p_lgbm'], 'p_lr': co['p_lr'], 'p_mdm': co['p_mdm'],
            'p_erp': mo['p_erp'], 'p_spec': mo['p_spec'], 'p_riem': mo['p_riem'], 'p_sent': mo['p_sent'],
            'p_final': p_final.astype(np.float32)
        }

# ─────────────────────────────────────────────────────────────────────────────
# Metric / plots / output
# ─────────────────────────────────────────────────────────────────────────────
def tile_trial_probs(p_trial):
    return np.clip(np.tile(p_trial[:, None], (1, N_TP)).astype(np.float32), 0.01, 0.99)


def window_auc_score(probs, y_bin, win_tp=10, min_ms=50):
    n_tp = probs.shape[1]
    aucs = []
    for s in range(n_tp - win_tp + 1):
        wp = probs[:, s:s+win_tp].mean(axis=1)
        try:
            a = roc_auc_score(y_bin, wp)
        except Exception:
            a = 0.5
        aucs.append(a)
    aucs = np.array(aucs)
    min_w = max(1, int(min_ms * FS / 1000))
    best_len, best_auc, run_l = 0, 0.5, 0
    run_s = 0
    for i, above in enumerate(aucs > 0.5):
        if above:
            if run_l == 0:
                run_s = i
            run_l += 1
        else:
            if run_l >= min_w and run_l > best_len:
                best_len = run_l
                best_auc = aucs[run_s:run_s+run_l].mean()
            run_l = 0
    if run_l >= min_w and run_l > best_len:
        best_auc = aucs[run_s:run_s+run_l].mean()
    return {'window_auc': float(best_auc), 'aucs': aucs, 'mean_auc': float(aucs.mean())}


def plot_results(model_name, fold_results, save_path, subject_ids=None):
    try:
        n = len(fold_results)
        win_aucs = [r['window_auc'] for r in fold_results]
        trial_aucs = [r['trial_auc'] for r in fold_results]
        mean_aucs = [r['mean_auc'] for r in fold_results]
        fig = plt.figure(figsize=(20, 8))
        gs = gridspec.GridSpec(1,3,figure=fig)
        ax1 = fig.add_subplot(gs[0,0])
        colors = ['#2ecc71' if a > 0.5 else '#e74c3c' for a in win_aucs]
        ax1.bar(range(1,n+1), win_aucs, color=colors, edgecolor='black', lw=0.5)
        ax1.axhline(0.5, color='k', ls='--', lw=1.5)
        ax1.axhline(np.mean(win_aucs), color='blue', lw=2)
        ax1.set_title(f'{model_name}: Window-AUC per Fold', fontweight='bold')
        ax1.set_ylim(0.4, 1.0)
        ax2 = fig.add_subplot(gs[0,1])
        colors2 = ['#2ecc71' if a > 0.5 else '#e74c3c' for a in trial_aucs]
        ax2.bar(range(1,n+1), trial_aucs, color=colors2, edgecolor='black', lw=0.5)
        ax2.axhline(0.5, color='k', ls='--', lw=1.5)
        ax2.axhline(np.mean(trial_aucs), color='orange', lw=2)
        if subject_ids is not None:
            for idx, (sid, a) in enumerate(zip(subject_ids, trial_aucs)):
                ax2.text(idx+1, a+0.002, sid[:4], ha='center', fontsize=6, rotation=45)
        ax2.set_title(f'{model_name}: Trial-AUC per Fold', fontweight='bold')
        ax2.set_ylim(0.4, 1.0)
        ax3 = fig.add_subplot(gs[0,2])
        vals = [np.mean(win_aucs), np.mean(trial_aucs), np.mean(mean_aucs)]
        errs = [np.std(win_aucs), np.std(trial_aucs), np.std(mean_aucs)]
        ax3.bar(['Window\nAUC','Trial\nAUC','Mean\nAUC'], vals, yerr=errs, color=['steelblue','orange','coral'], edgecolor='black', capsize=5)
        ax3.axhline(0.5, color='k', ls='--', lw=1.5)
        plt.suptitle(f'{model_name} — mean wAUC={np.mean(win_aucs):.4f} | trial AUC={np.mean(trial_aucs):.4f}', fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f' Plot saved → {save_path}')
    except Exception as e:
        print(f' Plot error ({model_name}): {e}')

# ─────────────────────────────────────────────────────────────────────────────
# Unified LOSO runner
# ─────────────────────────────────────────────────────────────────────────────
def build_blocks_for_fold(subjects, cache, exclude_idx, csp_global):
    tmpl = compute_fold_erp_template(cache, exclude_idx)
    train_parts = {'core': [], 'erp': [], 'spec': [], 'riem': [], 'sent': []}
    y_parts = []
    cov_parts = []
    for j, s in enumerate(subjects):
        if j == exclude_idx:
            continue
        blocks = extract_subject_blocks(cache[j]['eeg_proc'], tmpl)
        csp_f = csp_global.log_var_features(s['eeg'])
        blocks['core'] = np.hstack([blocks['core'], csp_f])
        blocks['sent'] = np.hstack([blocks['sent'], csp_f])
        for k in train_parts:
            train_parts[k].append(blocks[k])
        y_parts.append(s['labels'])
        cov_parts.append(cache[j]['covs'])
    X_tr = {k: np.vstack(v).astype(np.float32) for k,v in train_parts.items()}
    y_tr = np.concatenate(y_parts)
    cov_tr = np.vstack(cov_parts)

    blocks_val = extract_subject_blocks(cache[exclude_idx]['eeg_proc'], tmpl)
    csp_v = csp_global.log_var_features(subjects[exclude_idx]['eeg'])
    blocks_val['core'] = np.hstack([blocks_val['core'], csp_v])
    blocks_val['sent'] = np.hstack([blocks_val['sent'], csp_v])
    y_val = subjects[exclude_idx]['labels']
    cov_val = cache[exclude_idx]['covs']
    return X_tr, y_tr, cov_tr, blocks_val, y_val, cov_val


def run_loso_model(model_name, subjects, cache, csp_global):
    print('\n' + '='*70)
    print(f' LOSO — {model_name}')
    print('='*70)
    fold_results = []
    oof_trial_probs = []
    oof_labels = []
    for i in range(len(subjects)):
        t0 = time.time()
        X_tr, y_tr, cov_tr, X_val, y_val, cov_val = build_blocks_for_fold(subjects, cache, i, csp_global)
        y_val_bin = (y_val == 2).astype(int)
        if model_name == 'core':
            clf = ClaudeCoreTrialClassifier(use_mdm=HAVE_PYRIEMANN)
            clf.fit(X_tr['core'], y_tr, covs=cov_tr)
            pred = clf.predict(X_val['core'], covs=cov_val)
        elif model_name == 'moe':
            clf = MoETrialClassifier()
            clf.fit(X_tr, y_tr)
            pred = clf.predict(X_val)
        elif model_name == 'hybrid':
            clf = HybridTrialClassifier(use_mdm=HAVE_PYRIEMANN)
            clf.fit(X_tr, y_tr, covs=cov_tr)
            pred = clf.predict(X_val, covs=cov_val)
        else:
            raise ValueError(model_name)
        p_trial = pred['p_final']
        try:
            trial_auc = roc_auc_score(y_val_bin, p_trial)
        except Exception:
            trial_auc = 0.5
        probs_2d = tile_trial_probs(p_trial)
        metric = window_auc_score(probs_2d, y_val_bin)
        fold_results.append({**metric, 'trial_auc': float(trial_auc)})
        oof_trial_probs.append(p_trial.astype(np.float32))
        oof_labels.append(y_val.copy())
        print(f" Fold {i+1:2d} | {subjects[i]['id'][:22]:22s} | trial_AUC={trial_auc:.4f} | wAUC={metric['window_auc']:.4f} | mAUC={metric['mean_auc']:.4f} ({time.time()-t0:.0f}s)")
        gc.collect()
    return fold_results, oof_trial_probs, oof_labels

# ─────────────────────────────────────────────────────────────────────────────
# Final training + submission
# ─────────────────────────────────────────────────────────────────────────────
def fit_trial_platt(oof_probs, oof_labels):
    p = np.concatenate(oof_probs)[:, None]
    y = (np.concatenate(oof_labels) == 2).astype(int)
    cal = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
    cal.fit(p, y)
    return cal


def train_final_model(model_name, train_subjects, train_cache, csp_global):
    tmpl = compute_fold_erp_template(train_cache, exclude_idx=-1)
    tr_parts = {'core': [], 'erp': [], 'spec': [], 'riem': [], 'sent': []}
    y_parts = []
    cov_parts = []
    for s, c in zip(train_subjects, train_cache):
        blocks = extract_subject_blocks(c['eeg_proc'], tmpl)
        csp_f = csp_global.log_var_features(s['eeg'])
        blocks['core'] = np.hstack([blocks['core'], csp_f])
        blocks['sent'] = np.hstack([blocks['sent'], csp_f])
        for k in tr_parts:
            tr_parts[k].append(blocks[k])
        y_parts.append(s['labels'])
        cov_parts.append(c['covs'])
    X_tr = {k: np.vstack(v).astype(np.float32) for k,v in tr_parts.items()}
    y_tr = np.concatenate(y_parts)
    cov_tr = np.vstack(cov_parts)
    if model_name == 'core':
        clf = ClaudeCoreTrialClassifier(use_mdm=HAVE_PYRIEMANN)
        clf.fit(X_tr['core'], y_tr, covs=cov_tr)
    elif model_name == 'moe':
        clf = MoETrialClassifier()
        clf.fit(X_tr, y_tr)
    elif model_name == 'hybrid':
        clf = HybridTrialClassifier(use_mdm=HAVE_PYRIEMANN)
        clf.fit(X_tr, y_tr, covs=cov_tr)
    else:
        raise ValueError(model_name)
    return clf, tmpl


def generate_submission_for_model(model_name, clf, tmpl, train_subjects, test_subjects, csp_global, trial_cal=None):
    rows = []
    out_csv = os.path.join(OUT_DIR, f'submission_{model_name}.csv')
    for s in test_subjects:
        subj_id = s['subj_id']
        eeg = s['eeg'].copy()
        eeg, _, _ = zscore_subject(eeg)
        eeg = euclidean_alignment(eeg)
        eeg_proc = np.array([preprocess_trial(eeg[i]) for i in range(eeg.shape[0])], dtype=np.float32)
        blocks = extract_subject_blocks(eeg_proc, tmpl)
        csp_f = csp_global.log_var_features(s['eeg'])
        blocks['core'] = np.hstack([blocks['core'], csp_f])
        blocks['sent'] = np.hstack([blocks['sent'], csp_f])
        covs = compute_covariance_matrices(eeg_proc)
        if model_name == 'core':
            pred = clf.predict(blocks['core'], covs=covs)
        elif model_name == 'moe':
            pred = clf.predict(blocks)
        else:
            pred = clf.predict(blocks, covs=covs)
        p_trial = pred['p_final']
        if trial_cal is not None:
            p_trial = trial_cal.predict_proba(p_trial[:, None])[:,1]
        probs_2d = tile_trial_probs(p_trial)
        print(f" Subject {subj_id} ({s['id']}): trials={probs_2d.shape[0]} range=[{probs_2d.min():.3f}, {probs_2d.max():.3f}] mean={probs_2d.mean():.4f}")
        for tr in range(probs_2d.shape[0]):
            for tp in range(N_TP):
                rows.append({'id': f'{subj_id}_{tr}_{tp}', 'prediction': float(probs_2d[tr, tp])})
    df = pd.DataFrame(rows)
    assert df['prediction'].between(0,1).all()
    assert len(str(df.iloc[0]['id']).split('_')) == 3
    df.to_csv(out_csv, index=False)
    print(f' ✓ Saved submission → {out_csv}')
    return out_csv

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║ EEG Unified Runner v10.1-hybrid                                        ║
╠══════════════════════════════════════════════════════════════════════════╣
║ Runs in ONE script:                                                     ║
║   1) CORE   = Claude trial-level architecture                           ║
║   2) MOE    = EEG-specific expert routing                               ║
║   3) HYBRID = Claude core shift + EEG-specific routing                  ║
║ Shared cache / preprocessing / fold-safe ERP template                   ║
╚══════════════════════════════════════════════════════════════════════════╝
""")
    total_start = time.time()
    print('STEP 1: Loading training data...')
    train_subjects = load_all_training(EMO_DIR, NEU_DIR)
    print('\nSTEP 2: Loading test data...')
    test_subjects = load_all_test(TEST_DIR)
    print('\nSTEP 3: Building shared cache...')
    train_cache = build_cache(train_subjects)
    print('\nSTEP 4: Fitting global CSP...')
    eeg_all = np.vstack([s['eeg'] for s in train_subjects])
    lbl_all = np.concatenate([s['labels'] for s in train_subjects])
    csp_global = CSP(n_components=6).fit(eeg_all, lbl_all)
    del eeg_all, lbl_all
    gc.collect()

    summary = {}
    for model_name in MODELS_TO_RUN:
        if RUN_LOSO:
            fold_results, oof_probs, oof_labels = run_loso_model(model_name, train_subjects, train_cache, csp_global)
            plot_path = os.path.join(OUT_DIR, f'loso_{model_name}.png')
            plot_results(model_name.upper(), fold_results, plot_path, [s['id'] for s in train_subjects])
            trial_cal = fit_trial_platt(oof_probs, oof_labels)
            mean_wauc = float(np.mean([r['window_auc'] for r in fold_results]))
            summary[model_name] = mean_wauc
        else:
            fold_results, oof_probs, oof_labels, trial_cal = [], [], [], None
            summary[model_name] = None
        if GENERATE_SUBMISSIONS:
            clf, tmpl = train_final_model(model_name, train_subjects, train_cache, csp_global)
            out_csv = generate_submission_for_model(model_name, clf, tmpl, train_subjects, test_subjects, csp_global, trial_cal)
        print('\n' + '-'*70)

    print('\nFinal summary:')
    for k, v in summary.items():
        print(f' {k:>6s} : {v:.4f}' if v is not None else f' {k:>6s} : LOSO skipped')
    print(f"\n✓ DONE ({(time.time()-total_start)/60:.1f} min total)")
