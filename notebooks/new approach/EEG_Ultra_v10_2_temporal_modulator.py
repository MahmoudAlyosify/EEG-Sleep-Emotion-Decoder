
# ============================================================================
# EEG Emotional Memory — Ultra Pipeline v10.2 (CORE + Temporal Modulator)
# ============================================================================
# Goal:
#   Preserve the successful trial-level backbone (Claude core shift), but avoid
#   the failure mode of fully flat tiling by adding a LIGHT temporal modulator
#   with only 3 zones:
#       Zone A: 100–350 ms  (ERP / auditory-evoked response)
#       Zone B: 350–700 ms  (late processing / spindle-related)
#       Zone C: rest        (baseline / weak context)
#
# Strategy:
#   1) Train a TRIAL-LEVEL backbone: LDA + LGBM + LR (+ optional MDM)
#   2) Train 3 lean zone models on zone-specific features with the SAME trial label
#   3) Tune blending weights (alpha_A, alpha_B, alpha_C) on pooled LOSO OOF
#   4) Build a time-course by mixing p_trial with p_zoneA / p_zoneB / p_zoneC
#
# Output:
#   One submission CSV with time-varying (but low-DOF) probabilities.
# ============================================================================

import os, re, gc, time, warnings, subprocess, sys, logging
from pathlib import Path
import multiprocessing

warnings.filterwarnings('ignore')
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['TORCHINDUCTOR_DISABLE'] = '1'
logging.getLogger("torch._dynamo").setLevel(logging.CRITICAL)
logging.getLogger("torch._inductor").setLevel(logging.CRITICAL)

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

try:
    from pyriemann.classification import MDM
    HAVE_PYRIEMANN = True
except Exception:
    HAVE_PYRIEMANN = False

np.random.seed(42)
GPU_AVAILABLE = torch.cuda.is_available() if HAVE_TORCH else False
N_JOBS = multiprocessing.cpu_count()

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
FS = 200
N_TP = 200
N_CH = 16
ERP_A_START = int(0.100 * FS)   # 20
ERP_A_END   = int(0.350 * FS)   # 70
ERP_B_START = int(0.350 * FS)   # 70
ERP_B_END   = int(0.700 * FS)   # 140
BASE_START  = 0
BASE_END    = int(0.100 * FS)   # 20

ZONE_A = (ERP_A_START, ERP_A_END)
ZONE_B = (ERP_B_START, ERP_B_END)
# Zone C is the union of [0,20) U [140,200)

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

# Paths — same environment paths you already use
BASE = r'D:\EEG Project\Project Overview and Specifications\eeg_competition'
EMO_DIR = os.path.join(BASE, 'training', 'sleep_emo')
NEU_DIR = os.path.join(BASE, 'training', 'sleep_neu')
TEST_DIR = os.path.join(BASE, 'testing')
OUTPUT = os.path.join(BASE, 'submission_v10_2_temporal_modulator.csv')
RUN_LOSO = True

# Backbone model settings
K_BEST_LDA = 150
LGBM_PARAMS = dict(
    n_estimators=400 if GPU_AVAILABLE else 250,
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

# Temporal modulator settings
K_BEST_ZONE = 80
ALPHA_A_GRID = [0.25, 0.40, 0.55, 0.70]
ALPHA_B_GRID = [0.00, 0.10, 0.20, 0.30]
ALPHA_C_GRID = [0.00, 0.05, 0.10, 0.15]
BEST_ALPHAS = (0.55, 0.15, 0.05)  # overwritten by LOSO tuning

print(f"GPU_AVAILABLE={GPU_AVAILABLE} HAVE_TORCH={HAVE_TORCH} TORCH_DEVICE={TORCH_DEVICE}")
print(f"PyRiemann={'YES' if HAVE_PYRIEMANN else 'NO'}")
print("✓ Config loaded — v10.2 (trial backbone + temporal modulator)")
print(f" ERP Zone A: {ERP_A_START*1000//FS}-{ERP_A_END*1000//FS} ms")
print(f" Late Zone B: {ERP_B_START*1000//FS}-{ERP_B_END*1000//FS} ms")
print(f" Output CSV: {OUTPUT}")

# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_field(f, grp, key):
    field = grp[key]
    if isinstance(field, h5py.Dataset):
        val = field[()]
        if isinstance(val, h5py.Reference):
            return np.array(f[val])
        if hasattr(val, 'shape') and val.shape == (1,1):
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
            tv = np.arange(N_TP)/FS
        t_mask = tv >= -1e-6
        if np.any(~t_mask):
            tv = tv[t_mask]
            eeg = eeg[:, :, t_mask]
        if len(tv) != N_TP:
            tv = np.arange(N_TP)/FS
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


def compute_covariance_matrices(eeg_proc, t0=ERP_A_START, t1=ERP_A_END):
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
    print("Building cache (preprocess each subject once)...")
    out = []
    for i, s in enumerate(subjects):
        t0 = time.time()
        c = preprocess_subject_for_cache(s)
        out.append(c)
        print(f" [{i+1:2d}/{len(subjects)}] {s['id']}: eeg_proc={c['eeg_proc'].shape} ({time.time()-t0:.1f}s)")
    return out


def compute_fold_erp_template(cache, exclude_idx, t0=ERP_A_START, t1=ERP_A_END):
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
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────
def _plv(x, y):
    if len(x) < 4 or np.var(x) < 1e-14 or np.var(y) < 1e-14:
        return 0.0
    try:
        return float(np.abs(np.mean(np.exp(1j*(np.angle(hilbert(x)) - np.angle(hilbert(y)))))))
    except Exception:
        return 0.0


def _compute_band_cache(trial):
    bp_signals, bp_power = {}, {}
    for bname, (lo, hi) in BANDS.items():
        sig = bandpass(trial, lo, hi, FS)
        pwr = (np.abs(hilbert(sig.astype(np.float64), axis=-1))**2).astype(np.float32)
        bp_signals[bname] = sig
        bp_power[bname] = pwr
    return bp_signals, bp_power


def extract_backbone_features(trial, erp_template=None):
    a0, a1 = ERP_A_START, ERP_A_END
    b0, b1 = ERP_B_START, ERP_B_END
    s0, s1 = BASE_START, BASE_END
    bp_signals, bp_power = _compute_band_cache(trial)
    feat = []
    # A: log band power in ERP window
    for bname in BAND_NAMES:
        feat.extend(np.log1p(np.mean(bp_power[bname][:, a0:a1], axis=1)).tolist())
    # B: baseline corrected power
    for bname in BAND_NAMES:
        p_erp = np.mean(bp_power[bname][:, a0:a1], axis=1) + 1e-12
        p_bas = np.mean(bp_power[bname][:, s0:s1], axis=1) + 1e-12
        feat.extend((np.log(p_erp) - np.log(p_bas)).tolist())
    # C: late window power
    for bname in BAND_NAMES:
        feat.extend(np.log1p(np.mean(bp_power[bname][:, b0:b1], axis=1)).tolist())
    # D/E: ERP morphology + template corr
    erp_win = trial[:, a0:a1]
    feat.extend(erp_win.max(axis=1).tolist())
    feat.extend(erp_win.min(axis=1).tolist())
    feat.extend((erp_win.max(axis=1) - erp_win.min(axis=1)).tolist())
    feat.extend(np.sqrt(np.mean(erp_win**2, axis=1)).tolist())
    if erp_template is not None:
        for c in range(N_CH):
            x = erp_win[c]; tm = erp_template[c]
            if np.std(x) > 1e-8 and np.std(tm) > 1e-8:
                feat.append(float(np.corrcoef(x, tm)[0,1]))
            else:
                feat.append(0.0)
    else:
        feat.extend([0.0] * N_CH)
    # F: frontal theta asymmetry
    feat.extend(frontal_theta_asymmetry(bp_power['theta'], a0, a1))
    # G: asymmetry
    for ch1, ch2 in ASYM_PAIRS:
        for bn in ASYM_BANDS:
            p1 = np.mean(bp_power[bn][CH[ch1], a0:a1]) + 1e-12
            p2 = np.mean(bp_power[bn][CH[ch2], a0:a1]) + 1e-12
            feat.append(float(np.log(p2) - np.log(p1)))
    # H: PLV
    for bn in CONN_BANDS:
        sig = bp_signals[bn]
        for ch1, ch2 in CONN_PAIRS:
            feat.append(_plv(sig[CH[ch1], a0:a1], sig[CH[ch2], a0:a1]))
    # I: Riemannian proxy
    feat.extend(log_euclidean_cov_features(bp_power['theta'][:, a0:a1]).tolist())
    # J: sigma spindle 100–700ms
    sp0, sp1 = ERP_A_START, ERP_B_END
    feat.extend(np.log1p(np.mean(bp_power['sigma'][:, sp0:sp1], axis=1)).tolist())
    arr = np.array(feat, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def extract_zone_features(trial, zone='A', erp_template=None):
    bp_signals, bp_power = _compute_band_cache(trial)
    if zone == 'A':
        t0, t1 = ERP_A_START, ERP_A_END
    elif zone == 'B':
        t0, t1 = ERP_B_START, ERP_B_END
    else:
        # zone C summary built from two weak-context chunks: [0,20) and [140,200)
        t0, t1 = None, None

    feat = []
    if zone in ['A', 'B']:
        # power summaries in zone
        for bname in BAND_NAMES:
            feat.extend(np.log1p(np.mean(bp_power[bname][:, t0:t1], axis=1)).tolist())
        # theta/alpha and sigma
        th = np.mean(bp_power['theta'][:, t0:t1], axis=1) + 1e-12
        al = np.mean(bp_power['alpha'][:, t0:t1], axis=1) + 1e-12
        sg = np.mean(bp_power['sigma'][:, t0:t1], axis=1) + 1e-12
        feat.extend((np.log(th) - np.log(al)).tolist())
        feat.extend(np.log1p(sg).tolist())
        # morphology and template corr (zone A only strongly informative; zone B still allowed)
        seg = trial[:, t0:t1]
        feat.extend(seg.max(axis=1).tolist())
        feat.extend(seg.min(axis=1).tolist())
        feat.extend(np.sqrt(np.mean(seg**2, axis=1)).tolist())
        feat.extend((seg.max(axis=1) - seg.min(axis=1)).tolist())
        if erp_template is not None and zone == 'A':
            for c in range(N_CH):
                x = seg[c]; tm = erp_template[c]
                if np.std(x) > 1e-8 and np.std(tm) > 1e-8:
                    feat.append(float(np.corrcoef(x, tm)[0,1]))
                else:
                    feat.append(0.0)
        else:
            feat.extend([0.0] * N_CH)
        feat.extend(frontal_theta_asymmetry(bp_power['theta'], t0, t1))
        for ch1, ch2 in [('f3','f4'), ('c3','c4')]:
            for bn in ['theta', 'alpha', 'beta']:
                p1 = np.mean(bp_power[bn][CH[ch1], t0:t1]) + 1e-12
                p2 = np.mean(bp_power[bn][CH[ch2], t0:t1]) + 1e-12
                feat.append(float(np.log(p2) - np.log(p1)))
        # compact covariance block
        feat.extend(log_euclidean_cov_features(bp_power['theta'][:, t0:t1]).tolist())
    else:
        # Zone C: weak-context baseline/tail summaries
        # early baseline + late tail aggregated
        chunks = [(0, BASE_END), (ERP_B_END, N_TP)]
        for bname in BAND_NAMES:
            vals = []
            for s, e in chunks:
                vals.append(np.mean(bp_power[bname][:, s:e], axis=1))
            vals = np.mean(vals, axis=0)
            feat.extend(np.log1p(vals).tolist())
        th = np.mean([np.mean(bp_power['theta'][:, s:e], axis=1) for s,e in chunks], axis=0) + 1e-12
        al = np.mean([np.mean(bp_power['alpha'][:, s:e], axis=1) for s,e in chunks], axis=0) + 1e-12
        feat.extend((np.log(th) - np.log(al)).tolist())
        # global variance and DC drift proxy
        feat.extend(np.log1p(np.var(trial, axis=1)).tolist())
        feat.extend(np.mean(trial[:, :BASE_END], axis=1).tolist())
        feat.extend(np.mean(trial[:, ERP_B_END:], axis=1).tolist())
        # light asymmetry
        for ch1, ch2 in [('f3','f4'), ('c3','c4')]:
            p1 = np.mean(bp_power['theta'][CH[ch1], :BASE_END]) + 1e-12
            p2 = np.mean(bp_power['theta'][CH[ch2], ERP_B_END:]) + 1e-12
            feat.append(float(np.log(p2) - np.log(p1)))
    arr = np.array(feat, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def extract_subject_feature_sets(eeg_proc, erp_template=None):
    X_backbone, X_A, X_B, X_C = [], [], [], []
    for i in range(eeg_proc.shape[0]):
        tr = eeg_proc[i]
        X_backbone.append(extract_backbone_features(tr, erp_template))
        X_A.append(extract_zone_features(tr, 'A', erp_template))
        X_B.append(extract_zone_features(tr, 'B', erp_template))
        X_C.append(extract_zone_features(tr, 'C', erp_template))
    return {
        'backbone': np.vstack(X_backbone).astype(np.float32),
        'zoneA': np.vstack(X_A).astype(np.float32),
        'zoneB': np.vstack(X_B).astype(np.float32),
        'zoneC': np.vstack(X_C).astype(np.float32),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────
class TabularBinaryModel:
    def __init__(self, model_type='lda', k_best=None, C=1.0):
        self.model_type = model_type
        self.k_best = k_best
        self.C = C
        self.sel = None
        self.sc = RobustScaler()
        self.model = None
    def _make(self):
        if self.model_type == 'lda':
            return LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        if self.model_type == 'lr':
            return LogisticRegression(C=self.C, max_iter=2000, class_weight='balanced', solver='lbfgs', random_state=42)
        if self.model_type == 'lgbm':
            return lgb.LGBMClassifier(**LGBM_PARAMS)
        raise ValueError(self.model_type)
    def fit(self, X, y, sample_weight=None):
        if self.k_best is not None and self.k_best < X.shape[1]:
            self.sel = SelectKBest(f_classif, k=self.k_best)
            Xs = self.sel.fit_transform(X, y)
        else:
            Xs = X
        Xs = self.sc.fit_transform(Xs)
        self.model = self._make()
        try:
            if sample_weight is not None and self.model_type in ['lr', 'lgbm']:
                self.model.fit(Xs, y, sample_weight=sample_weight)
            else:
                self.model.fit(Xs, y)
        except Exception:
            self.model.fit(Xs, y)
        return self
    def predict_proba(self, X):
        Xs = self.sel.transform(X) if self.sel is not None else X
        Xs = self.sc.transform(Xs)
        try:
            return self.model.predict_proba(Xs)[:,1]
        except Exception:
            scores = self.model.decision_function(Xs)
            return 1 / (1 + np.exp(-scores))


class BackboneClassifier:
    def __init__(self, use_mdm=HAVE_PYRIEMANN):
        self.lda = TabularBinaryModel('lda', k_best=K_BEST_LDA)
        self.lgbm = TabularBinaryModel('lgbm', k_best=None)
        self.lr = TabularBinaryModel('lr', k_best=None, C=0.5)
        self.use_mdm = use_mdm and HAVE_PYRIEMANN
        self.mdm = None
        self.weights = {'lda': 0.30, 'lgbm': 0.40, 'lr': 0.15, 'mdm': 0.15 if self.use_mdm else 0.0}
    def fit(self, X, y, covs=None, sample_weight=None):
        yb = (y == 2).astype(int)
        self.lda.fit(X, yb)
        self.lgbm.fit(X, yb, sample_weight=sample_weight)
        self.lr.fit(X, yb, sample_weight=sample_weight)
        if self.use_mdm and covs is not None:
            self.mdm = MDM(metric='riemann')
            self.mdm.fit(covs, yb)
        return self
    def predict(self, X, covs=None):
        p_lda = self.lda.predict_proba(X)
        p_lgb = self.lgbm.predict_proba(X)
        p_lr = self.lr.predict_proba(X)
        p_mdm = None
        if self.use_mdm and self.mdm is not None and covs is not None:
            p = self.mdm.predict_proba(covs)
            classes = np.unique(self.mdm.classes_ if hasattr(self.mdm, 'classes_') else np.array([0,1]))
            idx1 = np.where(classes == 1)[0]
            p_mdm = p[:, idx1[0]] if len(idx1) > 0 else p[:, -1]
        w = self.weights
        if p_mdm is not None:
            tot = w['lda'] + w['lgbm'] + w['lr'] + w['mdm']
            p_final = (w['lda']*p_lda + w['lgbm']*p_lgb + w['lr']*p_lr + w['mdm']*p_mdm) / tot
        else:
            tot = w['lda'] + w['lgbm'] + w['lr']
            p_final = (w['lda']*p_lda + w['lgbm']*p_lgb + w['lr']*p_lr) / tot
        return {
            'p_trial': p_final.astype(np.float32),
            'p_lda': p_lda.astype(np.float32),
            'p_lgbm': p_lgb.astype(np.float32),
            'p_lr': p_lr.astype(np.float32),
            'p_mdm': None if p_mdm is None else p_mdm.astype(np.float32),
        }


class TemporalModulator:
    def __init__(self):
        self.modA = TabularBinaryModel('lr', k_best=K_BEST_ZONE, C=0.5)
        self.modB = TabularBinaryModel('lr', k_best=K_BEST_ZONE, C=0.5)
        self.modC = TabularBinaryModel('lr', k_best=min(40, K_BEST_ZONE), C=0.5)
        self.alphas = BEST_ALPHAS
    def fit(self, X_A, X_B, X_C, y, sample_weight=None):
        yb = (y == 2).astype(int)
        self.modA.fit(X_A, yb, sample_weight=sample_weight)
        self.modB.fit(X_B, yb, sample_weight=sample_weight)
        self.modC.fit(X_C, yb, sample_weight=sample_weight)
        return self
    def predict_zone_probs(self, X_A, X_B, X_C):
        return {
            'pA': self.modA.predict_proba(X_A).astype(np.float32),
            'pB': self.modB.predict_proba(X_B).astype(np.float32),
            'pC': self.modC.predict_proba(X_C).astype(np.float32),
        }
    def build_timecourse(self, p_trial, zone_probs, alphas=None):
        if alphas is None:
            alphas = self.alphas
        aA, aB, aC = alphas
        pA, pB, pC = zone_probs['pA'], zone_probs['pB'], zone_probs['pC']
        probs = np.tile(p_trial[:, None], (1, N_TP)).astype(np.float32)
        # Zone A
        probs[:, ERP_A_START:ERP_A_END] = ((1-aA) * p_trial[:, None] + aA * pA[:, None]).astype(np.float32)
        # Zone B
        probs[:, ERP_B_START:ERP_B_END] = ((1-aB) * p_trial[:, None] + aB * pB[:, None]).astype(np.float32)
        # Zone C (0-100ms and 700-1000ms)
        probs[:, :BASE_END] = ((1-aC) * p_trial[:, None] + aC * pC[:, None]).astype(np.float32)
        probs[:, ERP_B_END:] = ((1-aC) * p_trial[:, None] + aC * pC[:, None]).astype(np.float32)
        return np.clip(probs, 0.01, 0.99)

# ─────────────────────────────────────────────────────────────────────────────
# Utilities: sample weights, calibration, metric, plots
# ─────────────────────────────────────────────────────────────────────────────
def compute_subject_reliability_weights(train_subjects, cache):
    """One light pass: subject weights based on simple theta-power separability in ERP window."""
    weights = []
    for s, c in zip(train_subjects, cache):
        eeg_proc = c['eeg_proc']
        labels = s['labels']
        # simple proxy: theta power difference AUC inside subject
        X = []
        for tr in range(eeg_proc.shape[0]):
            bp = bandpass(eeg_proc[tr], 4.0, 8.0, FS)
            p = np.mean((np.abs(hilbert(bp.astype(np.float64), axis=-1))**2)[:, ERP_A_START:ERP_A_END], axis=1)
            X.append(float(np.mean(np.log1p(p))))
        X = np.array(X)
        yb = (labels == 2).astype(int)
        try:
            auc = roc_auc_score(yb, X)
        except Exception:
            auc = 0.5
        rel = 0.5 + max(0.0, auc - 0.5) * 2.0  # map 0.5->0.5, 0.6->0.7, etc.
        rel = float(np.clip(rel, 0.50, 1.00))
        weights.append(rel)
    return np.array(weights, dtype=np.float32)


def expand_subject_weights(subject_weights, train_subjects, exclude_idx):
    out = []
    for j, s in enumerate(train_subjects):
        if j == exclude_idx:
            continue
        out.append(np.full(s['labels'].shape[0], subject_weights[j], dtype=np.float32))
    return np.concatenate(out)


def fit_trial_platt(oof_trial_probs, oof_labels):
    p = np.concatenate(oof_trial_probs)[:, None]
    y = (np.concatenate(oof_labels) == 2).astype(int)
    cal = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
    cal.fit(p, y)
    return cal


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


def tune_alphas_global(oof_trial_list, oof_zone_list, oof_labels_list):
    best = BEST_ALPHAS
    best_auc = -1.0
    print("\nTuning temporal alpha grid on pooled OOF...")
    for aA in ALPHA_A_GRID:
        for aB in ALPHA_B_GRID:
            for aC in ALPHA_C_GRID:
                aucs = []
                for p_trial, z, lbl in zip(oof_trial_list, oof_zone_list, oof_labels_list):
                    probs = np.tile(p_trial[:, None], (1, N_TP)).astype(np.float32)
                    probs[:, ERP_A_START:ERP_A_END] = (1-aA)*p_trial[:,None] + aA*z['pA'][:,None]
                    probs[:, ERP_B_START:ERP_B_END] = (1-aB)*p_trial[:,None] + aB*z['pB'][:,None]
                    probs[:, :BASE_END] = (1-aC)*p_trial[:,None] + aC*z['pC'][:,None]
                    probs[:, ERP_B_END:] = (1-aC)*p_trial[:,None] + aC*z['pC'][:,None]
                    yb = (lbl == 2).astype(int)
                    aucs.append(window_auc_score(np.clip(probs,0.01,0.99), yb)['window_auc'])
                mean_auc = float(np.mean(aucs))
                if mean_auc > best_auc:
                    best_auc = mean_auc
                    best = (aA, aB, aC)
    print(f" Best alphas: A={best[0]:.2f} B={best[1]:.2f} C={best[2]:.2f} | pooled wAUC={best_auc:.4f}")
    return best, best_auc


def plot_loso_results(fold_results, save_path):
    try:
        n = len(fold_results)
        win_aucs = [r['window_auc'] for r in fold_results]
        trial_aucs = [r['trial_auc'] for r in fold_results]
        mean_aucs = [r['mean_auc'] for r in fold_results]
        fig = plt.figure(figsize=(20, 8))
        gs = gridspec.GridSpec(1, 3, figure=fig)
        ax1 = fig.add_subplot(gs[0,0])
        colors = ['#2ecc71' if a > 0.5 else '#e74c3c' for a in win_aucs]
        ax1.bar(range(1, n+1), win_aucs, color=colors, edgecolor='black', lw=0.5)
        ax1.axhline(0.5, color='k', ls='--', lw=1.5)
        ax1.axhline(np.mean(win_aucs), color='blue', lw=2)
        ax1.set_title('V10.2: Window-AUC per Fold', fontweight='bold')
        ax1.set_ylim(0.4, 1.0)
        ax2 = fig.add_subplot(gs[0,1])
        colors2 = ['#2ecc71' if a > 0.5 else '#e74c3c' for a in trial_aucs]
        ax2.bar(range(1, n+1), trial_aucs, color=colors2, edgecolor='black', lw=0.5)
        ax2.axhline(0.5, color='k', ls='--', lw=1.5)
        ax2.axhline(np.mean(trial_aucs), color='orange', lw=2)
        ax2.set_title('V10.2: Trial-AUC per Fold', fontweight='bold')
        ax2.set_ylim(0.4, 1.0)
        ax3 = fig.add_subplot(gs[0,2])
        vals = [np.mean(win_aucs), np.mean(trial_aucs), np.mean(mean_aucs)]
        errs = [np.std(win_aucs), np.std(trial_aucs), np.std(mean_aucs)]
        ax3.bar(['Window\nAUC', 'Trial\nAUC', 'Mean\nAUC'], vals, yerr=errs, color=['steelblue','orange','coral'], edgecolor='black', capsize=5)
        ax3.axhline(0.5, color='k', ls='--', lw=1.5)
        plt.suptitle(f'LOSO v10.2 — mean wAUC={np.mean(win_aucs):.4f} | trial AUC={np.mean(trial_aucs):.4f}', fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f" Plot saved → {save_path}")
    except Exception as e:
        print(f" Plot error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# LOSO
# ─────────────────────────────────────────────────────────────────────────────
def build_fold_matrices(train_subjects, cache, exclude_idx, csp_global):
    tmpl = compute_fold_erp_template(cache, exclude_idx)
    Xb_parts, XA_parts, XB_parts, XC_parts, y_parts, cov_parts = [], [], [], [], [], []
    for j, s in enumerate(train_subjects):
        if j == exclude_idx:
            continue
        feat = extract_subject_feature_sets(cache[j]['eeg_proc'], tmpl)
        csp = csp_global.log_var_features(s['eeg'])
        Xb_parts.append(np.hstack([feat['backbone'], csp]))
        XA_parts.append(feat['zoneA'])
        XB_parts.append(feat['zoneB'])
        XC_parts.append(feat['zoneC'])
        y_parts.append(s['labels'])
        cov_parts.append(cache[j]['covs'])
    X_tr = {
        'backbone': np.vstack(Xb_parts).astype(np.float32),
        'zoneA': np.vstack(XA_parts).astype(np.float32),
        'zoneB': np.vstack(XB_parts).astype(np.float32),
        'zoneC': np.vstack(XC_parts).astype(np.float32),
    }
    y_tr = np.concatenate(y_parts)
    cov_tr = np.vstack(cov_parts)

    feat_val = extract_subject_feature_sets(cache[exclude_idx]['eeg_proc'], tmpl)
    csp_v = csp_global.log_var_features(train_subjects[exclude_idx]['eeg'])
    X_val = {
        'backbone': np.hstack([feat_val['backbone'], csp_v]).astype(np.float32),
        'zoneA': feat_val['zoneA'].astype(np.float32),
        'zoneB': feat_val['zoneB'].astype(np.float32),
        'zoneC': feat_val['zoneC'].astype(np.float32),
    }
    y_val = train_subjects[exclude_idx]['labels']
    cov_val = cache[exclude_idx]['covs']
    return X_tr, y_tr, cov_tr, X_val, y_val, cov_val


def run_loso_v102(train_subjects, cache, csp_global):
    global BEST_ALPHAS
    print("\n" + "="*70)
    print(" LOSO Cross-Validation — v10.2 (core backbone + temporal modulator)")
    print("="*70)
    subj_weights = compute_subject_reliability_weights(train_subjects, cache)
    print(f" Subject reliability weights: {np.round(subj_weights, 3)}")

    fold_results = []
    oof_trial_probs, oof_zone_probs, oof_labels = [], [], []

    for i in range(len(train_subjects)):
        t0 = time.time()
        X_tr, y_tr, cov_tr, X_val, y_val, cov_val = build_fold_matrices(train_subjects, cache, i, csp_global)
        sw = expand_subject_weights(subj_weights, train_subjects, i)

        backbone = BackboneClassifier(use_mdm=HAVE_PYRIEMANN)
        backbone.fit(X_tr['backbone'], y_tr, covs=cov_tr, sample_weight=sw)
        base_out = backbone.predict(X_val['backbone'], covs=cov_val)
        p_trial = base_out['p_trial']

        mod = TemporalModulator().fit(X_tr['zoneA'], X_tr['zoneB'], X_tr['zoneC'], y_tr, sample_weight=sw)
        zone_probs = mod.predict_zone_probs(X_val['zoneA'], X_val['zoneB'], X_val['zoneC'])

        # For first pass, use current BEST_ALPHAS (updated globally after loop)
        probs_2d = mod.build_timecourse(p_trial, zone_probs, BEST_ALPHAS)
        yb = (y_val == 2).astype(int)
        try:
            trial_auc = roc_auc_score(yb, p_trial)
        except Exception:
            trial_auc = 0.5
        metric = window_auc_score(probs_2d, yb)
        fold_results.append({**metric, 'trial_auc': float(trial_auc)})
        oof_trial_probs.append(p_trial.astype(np.float32))
        oof_zone_probs.append({k: v.astype(np.float32) for k,v in zone_probs.items()})
        oof_labels.append(y_val.copy())
        print(f" Fold {i+1:2d} | {train_subjects[i]['id'][:22]:22s} | trial_AUC={trial_auc:.4f} | wAUC={metric['window_auc']:.4f} | mAUC={metric['mean_auc']:.4f} ({time.time()-t0:.0f}s)")
        gc.collect()

    # Tune temporal alphas on pooled OOF and re-evaluate
    BEST_ALPHAS, best_oof = tune_alphas_global(oof_trial_probs, oof_zone_probs, oof_labels)
    print("\nRe-evaluating folds with tuned temporal alphas...")
    fold_results = []
    for p_trial, z, lbl in zip(oof_trial_probs, oof_zone_probs, oof_labels):
        probs = np.tile(p_trial[:,None], (1, N_TP)).astype(np.float32)
        aA, aB, aC = BEST_ALPHAS
        probs[:, ERP_A_START:ERP_A_END] = (1-aA)*p_trial[:,None] + aA*z['pA'][:,None]
        probs[:, ERP_B_START:ERP_B_END] = (1-aB)*p_trial[:,None] + aB*z['pB'][:,None]
        probs[:, :BASE_END] = (1-aC)*p_trial[:,None] + aC*z['pC'][:,None]
        probs[:, ERP_B_END:] = (1-aC)*p_trial[:,None] + aC*z['pC'][:,None]
        yb = (lbl == 2).astype(int)
        try:
            trial_auc = roc_auc_score(yb, p_trial)
        except Exception:
            trial_auc = 0.5
        m = window_auc_score(np.clip(probs,0.01,0.99), yb)
        fold_results.append({**m, 'trial_auc': float(trial_auc)})

    win_aucs = [r['window_auc'] for r in fold_results]
    trial_aucs = [r['trial_auc'] for r in fold_results]
    mean_aucs = [r['mean_auc'] for r in fold_results]
    print(f"\n ╔═══════════════════════════════════════════════════════════╗")
    print(f" ║ V10.2 — CORE + TEMPORAL MODULATOR                        ║")
    print(f" ║ Window AUC : {np.mean(win_aucs):.4f} ± {np.std(win_aucs):.4f}                   ║")
    print(f" ║ Trial AUC  : {np.mean(trial_aucs):.4f} ± {np.std(trial_aucs):.4f}                   ║")
    print(f" ║ Mean AUC   : {np.mean(mean_aucs):.4f} ± {np.std(mean_aucs):.4f}                   ║")
    print(f" ║ Best fold  : {max(win_aucs):.4f}                                ║")
    print(f" ║ Alphas     : A={BEST_ALPHAS[0]:.2f} B={BEST_ALPHAS[1]:.2f} C={BEST_ALPHAS[2]:.2f}          ║")
    print(f" ╚═══════════════════════════════════════════════════════════╝")
    return fold_results, oof_trial_probs, oof_labels

# ─────────────────────────────────────────────────────────────────────────────
# Final training + submission
# ─────────────────────────────────────────────────────────────────────────────
def train_final_model(train_subjects, cache, csp_global):
    subj_weights = compute_subject_reliability_weights(train_subjects, cache)
    tmpl = compute_fold_erp_template(cache, exclude_idx=-1)
    Xb_parts, XA_parts, XB_parts, XC_parts, y_parts, cov_parts = [], [], [], [], [], []
    for s, c, w in zip(train_subjects, cache, subj_weights):
        feat = extract_subject_feature_sets(c['eeg_proc'], tmpl)
        csp = csp_global.log_var_features(s['eeg'])
        Xb_parts.append(np.hstack([feat['backbone'], csp]))
        XA_parts.append(feat['zoneA'])
        XB_parts.append(feat['zoneB'])
        XC_parts.append(feat['zoneC'])
        y_parts.append(s['labels'])
        cov_parts.append(c['covs'])
    X = {
        'backbone': np.vstack(Xb_parts).astype(np.float32),
        'zoneA': np.vstack(XA_parts).astype(np.float32),
        'zoneB': np.vstack(XB_parts).astype(np.float32),
        'zoneC': np.vstack(XC_parts).astype(np.float32),
    }
    y = np.concatenate(y_parts)
    covs = np.vstack(cov_parts)
    sw = np.concatenate([np.full(s['labels'].shape[0], subj_weights[i], dtype=np.float32) for i,s in enumerate(train_subjects)])

    backbone = BackboneClassifier(use_mdm=HAVE_PYRIEMANN).fit(X['backbone'], y, covs=covs, sample_weight=sw)
    mod = TemporalModulator().fit(X['zoneA'], X['zoneB'], X['zoneC'], y, sample_weight=sw)
    mod.alphas = BEST_ALPHAS
    return backbone, mod, tmpl


def generate_submission_v102(train_subjects, test_subjects, cache, csp_global, trial_cal=None, output_path=OUTPUT):
    print("\n" + "="*70)
    print(" Final Training → Submission Generation (v10.2)")
    print("="*70)
    backbone, mod, tmpl = train_final_model(train_subjects, cache, csp_global)
    rows = []
    for s in test_subjects:
        subj_id = s['subj_id']
        eeg = s['eeg'].copy()
        eeg, _, _ = zscore_subject(eeg)
        eeg = euclidean_alignment(eeg)
        eeg_proc = np.array([preprocess_trial(eeg[i]) for i in range(eeg.shape[0])], dtype=np.float32)
        feat = extract_subject_feature_sets(eeg_proc, tmpl)
        csp = csp_global.log_var_features(s['eeg'])
        Xb = np.hstack([feat['backbone'], csp]).astype(np.float32)
        covs = compute_covariance_matrices(eeg_proc)
        base_out = backbone.predict(Xb, covs=covs)
        p_trial = base_out['p_trial']
        if trial_cal is not None:
            p_trial = trial_cal.predict_proba(p_trial[:,None])[:,1]
        zone_probs = mod.predict_zone_probs(feat['zoneA'], feat['zoneB'], feat['zoneC'])
        probs_2d = mod.build_timecourse(p_trial, zone_probs)
        print(f" Subject {subj_id} ({s['id']}): trials={probs_2d.shape[0]} range=[{probs_2d.min():.3f}, {probs_2d.max():.3f}] mean={probs_2d.mean():.4f}")
        for tr in range(probs_2d.shape[0]):
            for tp in range(N_TP):
                rows.append({'id': f'{subj_id}_{tr}_{tp}', 'prediction': float(probs_2d[tr, tp])})
    df = pd.DataFrame(rows)
    assert df['prediction'].between(0,1).all()
    assert len(str(df.iloc[0]['id']).split('_')) == 3
    df.to_csv(output_path, index=False)
    print(f"\n✓ Submission saved → {output_path}")
    print(df.head(6).to_string(index=False))
    return df

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║ EEG Emotional Memory — Ultra Pipeline v10.2                            ║
╠══════════════════════════════════════════════════════════════════════════╣
║ Trial-level backbone (Claude core shift)                               ║
║ + 3-zone temporal modulator (A/B/C only; no per-tp TRE)               ║
║ Same paths as your current environment                                 ║
╚══════════════════════════════════════════════════════════════════════════╝
""")
    total_start = time.time()
    print('STEP 1: Loading training data...')
    train_subjects = load_all_training(EMO_DIR, NEU_DIR)
    print('\nSTEP 2: Loading test data...')
    test_subjects = load_all_test(TEST_DIR)
    print('\nSTEP 3: Building cache...')
    cache = build_cache(train_subjects)
    print('\nSTEP 4: Fitting global CSP...')
    eeg_all = np.vstack([s['eeg'] for s in train_subjects])
    lbl_all = np.concatenate([s['labels'] for s in train_subjects])
    csp_global = CSP(n_components=6).fit(eeg_all, lbl_all)
    del eeg_all, lbl_all
    gc.collect()

    if RUN_LOSO:
        print('\nSTEP 5: LOSO CV...')
        fold_results, oof_trial_probs, oof_labels = run_loso_v102(train_subjects, cache, csp_global)
        plot_loso_results(fold_results, os.path.join(BASE, 'loso_results_v10_2.png'))
        print('\nSTEP 6: Trial-level Platt calibration...')
        trial_cal = fit_trial_platt(oof_trial_probs, oof_labels)
        print(f"\n ► Mean Window-AUC = {np.mean([r['window_auc'] for r in fold_results]):.4f} ± {np.std([r['window_auc'] for r in fold_results]):.4f}")
    else:
        fold_results = []
        trial_cal = None

    print('\nSTEP 7: Final training + submission...')
    df = generate_submission_v102(train_subjects, test_subjects, cache, csp_global, trial_cal, OUTPUT)

    print('\nSTEP 8: Validation...')
    assert 'id' in df.columns and 'prediction' in df.columns
    assert df['prediction'].between(0,1).all()
    parts = str(df.iloc[0]['id']).split('_')
    assert len(parts) == 3
    print(f" ✓ Total rows: {len(df):,}")
    print(f" ✓ Pred range: [{df.prediction.min():.4f}, {df.prediction.max():.4f}]")
    print(f"\n✓ DONE ({(time.time()-total_start)/60:.1f} min total)")
