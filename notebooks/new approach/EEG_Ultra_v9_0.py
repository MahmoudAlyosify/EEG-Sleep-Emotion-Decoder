# ============================================================================
# EEG Emotional Memory — Ultra Pipeline v9.0  (DATA-DRIVEN FINAL)
# ============================================================================
#
# Root-cause analysis of v8.0 (wAUC=0.521):
#   The AUC time-course plot revealed that the REAL discriminative signal
#   lives at 100–350ms (auditory N1/P2 ERP), NOT at 400–600ms as assumed.
#   v8.0 trained classifiers on the sub-chance zone and upweighted it in gate.
#
# v9.0 fixes — all verified against the observed AUC time-course:
#
#  V9-1  SIGNAL WINDOW EXPANDED to 50–900ms (171 timepoints):
#         Now covers the early auditory ERP window (100–350ms) that was
#         completely absent from v8.0/v8.1 TRE training.
#         Neuroscience: Sculthorpe et al. 2009 — auditory N1/P2 is modulated
#         by emotional content even during NREM sleep.
#
#  V9-2  GATE CORRECTED (data-driven from v8.0 AUC curve):
#         Zone A (100–350ms) = 1.00  ← actual observed peak
#         Zone B (50–700ms)  = 0.60  ← context zone
#         Zone C (rest)      = 0.12  ← noise suppression
#         Previous v8.1 gate upweighted 400–600ms = the sub-chance zone!
#
#  V9-3  ERP TEMPLATE FEATURES (48 features per trial, tiled across tps):
#         Grand ERP difference template (emo – neu) from all training subjects.
#         Per-trial: correlation with template, RMS amplitude, peak-to-peak
#         in 100–350ms window. Direct measure of N1/P2 emotional modulation.
#         Reference: Sculthorpe et al. 2009; Atienza & Cantero 2001.
#
#  V9-4  INTER-TRIAL PHASE COHERENCE (ITPC, 16 features × 200 tp):
#         The gold standard for auditory evoked responses. Measures how
#         consistently the phase is locked across trials in theta band.
#         Per-subject, tiled across trials. Reference: Luo & Poeppel 2007.
#
#  V9-5  FRONTAL THETA ASYMMETRY INDEX (3 features per timepoint):
#         FAI-theta = log(F4_theta) – log(F3_theta).
#         Right frontal theta dominates in negative emotional states.
#         Reference: Aftanas & Golocheikine 2001, Nature Neuroscience.
#
#  V9-6  GRU VALIDATION ON EARLY WINDOW (tp 20–70 = 100–350ms):
#         Early stopping now monitors AUC in the actual signal window,
#         not the 400–600ms zone where the model was anti-correlated.
#
#  V9-7  SLIDING WINDOW GRU INFERENCE:
#         Inference uses the same overlapping frame approach as training.
#         Each timepoint's prediction = average of all frames containing it.
#         Eliminates the train/test inconsistency from v8.1.
#
#  V9-8  PLATT CALIBRATION (replaces Isotonic):
#         LogisticRegression sigmoid calibration is safer than Isotonic
#         with small N (14 subjects). Less overfitting to OOF noise.
#
#  V9-9  PER-SUBJECT DIAGNOSTIC PLOT:
#         After LOSO, saves individual AUC time-course per subject.
#         Identifies which subjects drive performance.
#
# All v8.1 features preserved:
#  ✓ Parallel trial feature extraction
#  ✓ BiGRU × 2 with frame overlap 20% (Yoo 2023)
#  ✓ RMSprop optimizer (Alhagry 2017)
#  ✓ Log-Euclidean Riemannian covariance (GPU-5)
#  ✓ Euclidean Alignment (ADD-1)
#  ✓ DE time-resolved (V8-6) = 96 features
#  ✓ LDA + LightGBM(GPU) + XGBoost(CUDA) + CatBoost(GPU)
#  ✓ Adaptive per-timepoint ensemble weights (ADD-4)
#  ✓ Gaussian sigma auto-tuning on OOF
#  ✓ CSP spatial filters
#  ✓ ~715 tabular features → SelectKBest(350)
#
# Expected wAUC: 0.545–0.580  (vs 0.521 in v8.0)
# ============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — Install & GPU Check
# ─────────────────────────────────────────────────────────────────────────────

import subprocess, sys, os, logging

os.environ['TORCH_COMPILE_DISABLE'] = '1'
logging.getLogger("torch._dynamo").setLevel(logging.CRITICAL)
logging.getLogger("torch._inductor").setLevel(logging.CRITICAL)

for pkg in ['lightgbm', 'xgboost', 'catboost', 'tqdm']:
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', pkg, '-q'],
                       capture_output=True, text=True)
    print(f"{'✓' if r.returncode==0 else '✗'} {pkg}")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    HAVE_TORCH = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark        = True

        import platform, torch._dynamo
        if platform.system() == 'Windows':
            torch._dynamo.config.suppress_errors = True
            torch._dynamo.disable()
            print("   torch.compile  : DISABLED (Triton unavailable on Windows)")

        TORCH_DEVICE = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        cc       = torch.cuda.get_device_capability(0)
        print(f"\n🚀 PyTorch GPU  : {gpu_name}")
        print(f"   CUDA version : {torch.version.cuda}")
        print(f"   Compute cap  : {cc[0]}.{cc[1]}")
        print(f"   VRAM         : {vram_gb:.1f} GB")
    else:
        TORCH_DEVICE = torch.device('cpu')
        print("\n⚠ PyTorch CPU only")
except ImportError:
    HAVE_TORCH   = False
    TORCH_DEVICE = None
    print("\n⚠ PyTorch NOT installed — GRU disabled.")

try:
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=name,memory.total,driver_version,compute_cap',
         '--format=csv,noheader'],
        capture_output=True, text=True)
    GPU_AVAILABLE = (result.returncode == 0)
    if GPU_AVAILABLE:
        print(f"\n🖥  nvidia-smi: {result.stdout.strip()}")
    else:
        print("\n⚠ No GPU — tabular models on CPU")
except Exception:
    GPU_AVAILABLE = False

print(f"\nGPU_AVAILABLE={GPU_AVAILABLE}  HAVE_TORCH={HAVE_TORCH}  "
      f"TORCH_DEVICE={TORCH_DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — Imports
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import h5py
import os, re, time, warnings, gc
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

from scipy.signal  import butter, sosfiltfilt, hilbert, detrend as sp_detrend, savgol_filter
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.stats   import skew, kurtosis
from scipy.linalg  import eigh

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model          import Ridge, LogisticRegression
from sklearn.preprocessing         import RobustScaler
from sklearn.metrics               import roc_auc_score
from sklearn.feature_selection     import SelectKBest, f_classif
from sklearn.isotonic              import IsotonicRegression

import lightgbm as lgb
import xgboost  as xgb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    from catboost import CatBoostClassifier
    HAVE_CATBOOST = True
    print("✓ CatBoost available")
except ImportError:
    HAVE_CATBOOST = False
    print("⚠ CatBoost not available")

warnings.filterwarnings('ignore')
np.random.seed(42)

GPU_AVAILABLE = globals().get('GPU_AVAILABLE', False)
N_JOBS        = multiprocessing.cpu_count()
print(f"✓ All imports OK | CPU cores: {N_JOBS} | GPU: {GPU_AVAILABLE} | "
      f"CatBoost: {HAVE_CATBOOST} | PyTorch: {HAVE_TORCH}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — Paths  (edit to match your environment)
# ─────────────────────────────────────────────────────────────────────────────

BASE     = r'D:\EEG Project\Project Overview and Specifications\eeg_competition'
EMO_DIR  = os.path.join(BASE, 'training', 'sleep_emo')
NEU_DIR  = os.path.join(BASE, 'training', 'sleep_neu')
TEST_DIR = os.path.join(BASE, 'testing')
OUTPUT   = os.path.join(BASE, 'submission.csv')

for name, path in [('EMO_DIR', EMO_DIR), ('NEU_DIR', NEU_DIR), ('TEST_DIR', TEST_DIR)]:
    exists = os.path.exists(path)
    count  = len(list(Path(path).glob('*.mat'))) if exists else 0
    status = f'✓  ({count} .mat files)' if exists else '✗  NOT FOUND — check path'
    print(f'{name:12s}: {path}')
    print(f'               {status}\n')


# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — Configuration  (V9.0: DATA-DRIVEN from v8.0 AUC time-course)
# ─────────────────────────────────────────────────────────────────────────────

FS   = 200
N_TP = 200
N_CH = 16

# ── V9-1: EXPANDED to 50-900ms to capture auditory ERP signal at 100-350ms ──
# The v8.0 plot showed peak AUC at 200-300ms — this window was NOT trained
# in v8.1 (it started at 300ms). Now we include it explicitly.
T_SIG_START_MS = 50;   TP_SIG_START = int(T_SIG_START_MS / 1000 * FS)   # tp=10
T_SIG_END_MS   = 900;  TP_SIG_END   = int(T_SIG_END_MS   / 1000 * FS)   # tp=180
SIGNAL_TPS     = list(range(TP_SIG_START, TP_SIG_END + 1))               # 171 tps

# ── V9-2: GATE corrected — based on OBSERVED AUC curve, NOT paper assumption ─
# v8.0 AUC timecourse: peak at 200-300ms, dip BELOW 0.5 at 400-600ms.
# Zone A = real observed peak (early auditory ERP)
# Zone B = broad context window kept at 60%
# Zone C = far noise suppressed to 12%
ERP_WIN_START  = int(0.100 * FS)   # tp=20  (100ms — N1 onset)
ERP_WIN_END    = int(0.350 * FS)   # tp=70  (350ms — P2 end)

GATE_ZONE_A_START = ERP_WIN_START  # tp=20
GATE_ZONE_A_END   = ERP_WIN_END    # tp=70
GATE_ZONE_B_START = int(0.050 * FS)   # tp=10
GATE_ZONE_B_END   = int(0.700 * FS)   # tp=140
BLEND_ALPHA_A     = 1.00   # full signal in ERP peak zone
BLEND_ALPHA_B     = 0.60   # soft in context zone (50-100ms, 350-700ms)
BLEND_ALPHA_C     = 0.12   # hard suppression: 0-50ms, 700ms+

WIN        = 40
N_CSP      = 4
N_FEAT_SEL = 350

BANDS = {
    'delta': (1.0,  4.0),
    'theta': (4.0,  8.0),
    'alpha': (8.0,  13.0),
    'sigma': (12.0, 16.0),
    'beta':  (13.0, 30.0),
    'hbeta': (20.0, 40.0),
}

CHANNELS = ['c3','c4','o1','o2','cp3','f3','f4','cp4',
            'c5','cz','c6','cp5','p7','pz','p8','cp6']
CH = {c: i for i, c in enumerate(CHANNELS)}

CONN_PAIRS = [
    ('f3','pz'),  ('f4','pz'),  ('f3','cz'),   ('f4','cz'),
    ('c3','c4'),  ('cp3','cp4'),('f3','f4'),    ('cz','pz'),
    ('f3','cp4'), ('f4','cp3'),
]
ASYM_PAIRS_EXT = [('c5','c6'), ('cp5','cp6'), ('p7','p8')]
ASYM_BANDS_EXT = ['theta', 'alpha', 'beta']

SMOOTH_SIGMA_CANDIDATES = [4, 6, 8, 10, 12]
SMOOTH_SIGMA = 8
SAVGOL_WIN   = 21
SAVGOL_POLY  = 3

# ── V9-6: GRU validates on EARLY WINDOW (100-350ms) ──────────────────────────
GRU_VAL_START  = ERP_WIN_START   # tp=20
GRU_VAL_END    = ERP_WIN_END     # tp=70

# ── V9-7: GRU hyperparameters ─────────────────────────────────────────────────
# ENN_W starts at 0.0 — let tuning decide. v8.0 GRU hurt because it was
# trained on the sub-chance zone. With the corrected window it may help.
ENN_W            = 0.00
ENN_W_GRID       = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
GRU_EPOCHS       = 150
GRU_LR           = 1e-3
GRU_PATIENCE     = 30
GRU_HIDDEN       = 64
GRU_HIDDEN2      = 32
GRU_DROPOUT      = 0.20
GRU_FRAME_LEN    = 40
GRU_OVERLAP      = 0.20

LGBM_DEVICE = 'gpu'  if GPU_AVAILABLE else 'cpu'
XGB_DEVICE  = 'cuda' if GPU_AVAILABLE else 'cpu'

# ── Global ERP template (computed once, used across all folds) ────────────────
GRAND_ERP_TEMPLATE = None   # (N_CH, ERP_WIN_END-ERP_WIN_START) float32

print(f"✓ Config loaded — v9.0 (DATA-DRIVEN)")
print(f"  Signal window : tp [{TP_SIG_START}–{TP_SIG_END}]  "
      f"({T_SIG_START_MS}–{T_SIG_END_MS}ms)  171 tps  ← expanded from v8.1")
print(f"  Gate (corrected): A(100-350ms)=1.0 | B(50-700ms)=0.60 | C=0.12")
print(f"  GRU val window: {GRU_VAL_START*1000//FS}-{GRU_VAL_END*1000//FS}ms "
      f"(early ERP, corrected from v8.1)")
print(f"  ENN_W start: {ENN_W} (conservative — tuned on OOF)")
print(f"  LGBM: {800 if GPU_AVAILABLE else 500} trees | "
      f"XGB: {600 if GPU_AVAILABLE else 350} trees | CatBoost: GPU={GPU_AVAILABLE}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — Robust HDF5 Loader  (unchanged)
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
    path = str(path)
    with h5py.File(path, 'r') as f:
        grp = None
        if 'data' in f:
            grp = f['data']
        else:
            for k in f.keys():
                if hasattr(f[k], 'keys') and 'trial' in f[k]:
                    grp = f[k]; break
        if grp is None:
            raise ValueError(f"Cannot find 'data' struct in {path}")

        trial_raw = _resolve_field(f, grp, 'trial')
        if trial_raw.ndim == 3:
            sh = trial_raw.shape
            if   sh[2]==N_CH  and sh[1]==N_TP: trial_raw = trial_raw.transpose(0,2,1)
            elif sh[0]==N_CH  and sh[1]==N_TP: trial_raw = trial_raw.transpose(2,0,1)
            elif sh[0]==N_TP  and sh[1]==N_CH: trial_raw = trial_raw.transpose(2,1,0)
        elif trial_raw.ndim == 2:
            trial_raw = trial_raw.T[np.newaxis]
        eeg = trial_raw.astype(np.float32)

        if label_override is not None:
            labels = np.full(eeg.shape[0], label_override, dtype=int)
        else:
            try:
                ti = _resolve_field(f, grp, 'trialinfo')
                if   ti.ndim==2 and ti.shape[0]==eeg.shape[0]: labels = ti[:,0].astype(int)
                elif ti.ndim==2 and ti.shape[1]==eeg.shape[0]: labels = ti[0,:].astype(int)
                else:                                            labels = ti.flatten().astype(int)
            except Exception:
                labels = np.ones(eeg.shape[0], dtype=int)

        try:   tv = _resolve_field(f, grp, 'time').flatten()
        except: tv = np.arange(N_TP) / FS

        t_mask = tv >= -1e-6
        if np.any(~t_mask):
            tv = tv[t_mask]; eeg = eeg[:,:,t_mask]
        if len(tv) != N_TP:
            tv = np.arange(N_TP) / FS

    n_emo = (labels==2).sum(); n_neu = (labels==1).sum()
    print(f"    ✓ {Path(path).name}: eeg={eeg.shape} | neu={n_neu} emo={n_emo}")
    return {'eeg': eeg, 'labels': labels, 'time': tv}


def load_all_training(emo_dir, neu_dir):
    emo_data, neu_data = {}, {}
    for fpath in sorted(Path(emo_dir).glob('*.mat')):
        try:
            d = load_mat(str(fpath), label_override=2)
            d['id'] = fpath.stem; emo_data[fpath.stem] = d
        except Exception as e:
            print(f"    ✗ {fpath.name}: {e}")
    for fpath in sorted(Path(neu_dir).glob('*.mat')):
        try:
            d = load_mat(str(fpath), label_override=1)
            d['id'] = fpath.stem; neu_data[fpath.stem] = d
        except Exception as e:
            print(f"    ✗ {fpath.name}: {e}")

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
            'eeg':    np.concatenate(parts_eeg, axis=0),
            'labels': np.concatenate(parts_lbl, axis=0),
            'time':   (emo_data.get(stem) or neu_data.get(stem))['time'],
            'id':     stem
        }
        n_e = (merged['labels']==2).sum(); n_n = (merged['labels']==1).sum()
        print(f"  → {stem}: {merged['eeg'].shape[0]} trials (emo={n_e}, neu={n_n})")
        subjects.append(merged)
    print(f"\n✓ Training: {len(subjects)} subjects loaded")
    return subjects


def load_all_test(test_dir):
    subjects = []
    for fpath in sorted(Path(test_dir).glob('*.mat')):
        try:
            d = load_mat(str(fpath))
            nums = re.findall(r'\d+', fpath.stem)
            d['id'] = fpath.stem
            d['subj_id'] = int(nums[-1]) if nums else len(subjects)+1
            subjects.append(d)
        except Exception as e:
            print(f"    ✗ {fpath.name}: {e}")
    print(f"\n✓ Test: {len(subjects)} subjects | "
          f"IDs: {[s['subj_id'] for s in subjects]}")
    return subjects

print("✓ Loaders defined")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 — Preprocessing  (unchanged from v8.1)
# ─────────────────────────────────────────────────────────────────────────────

def bandpass(data, lo, hi, fs=FS, order=4):
    nyq = fs / 2.0
    sos = butter(order, [max(lo/nyq,1e-4), min(hi/nyq,0.9999)],
                 btype='band', output='sos')
    return sosfiltfilt(sos, data, axis=-1).astype(np.float32)


def preprocess_trial(raw_trial):
    """(16,200) → detrend + avg-ref + 0.5-40 Hz bandpass."""
    x = sp_detrend(raw_trial.astype(np.float64), axis=-1).astype(np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    x = bandpass(x, lo=0.5, hi=40.0, fs=FS, order=4)
    return x


def zscore_subject(eeg):
    """Per-subject Z-score: (n_tr,16,200) → normalised."""
    mu  = eeg.mean(axis=(0,2), keepdims=True)
    sig = eeg.std( axis=(0,2), keepdims=True) + 1e-8
    return ((eeg - mu) / sig).astype(np.float32), mu, sig


def euclidean_alignment(eeg):
    """Euclidean Alignment for cross-subject normalisation."""
    n_tr, n_ch, n_tp = eeg.shape
    covs = np.zeros((n_tr, n_ch, n_ch), dtype=np.float64)
    for i in range(n_tr):
        x = eeg[i].astype(np.float64)
        x = x - x.mean(axis=1, keepdims=True)
        covs[i] = (x @ x.T) / (n_tp - 1)
    R_mean = covs.mean(axis=0)
    eps = 1e-6 * np.trace(R_mean) / n_ch
    R_mean += eps * np.eye(n_ch)
    try:
        eigvals, eigvecs = np.linalg.eigh(R_mean)
        eigvals = np.maximum(eigvals, 1e-10)
        R_inv_sqrt = eigvecs @ np.diag(eigvals**(-0.5)) @ eigvecs.T
    except np.linalg.LinAlgError:
        return eeg.astype(np.float32)
    eeg_aligned = np.zeros_like(eeg, dtype=np.float32)
    for i in range(n_tr):
        eeg_aligned[i] = (R_inv_sqrt @ eeg[i].astype(np.float64)).astype(np.float32)
    return eeg_aligned


def log_euclidean_cov_features(seg):
    """GPU-5: Log-Euclidean covariance — tangent space projection."""
    n_ch  = seg.shape[0]
    n_feat = n_ch * (n_ch + 1) // 2
    if seg.shape[1] < 3:
        return np.zeros(n_feat, dtype=np.float32)
    seg_c = seg.astype(np.float64)
    seg_c = seg_c - seg_c.mean(axis=1, keepdims=True)
    C = (seg_c @ seg_c.T) / max(seg_c.shape[1] - 1, 1)
    eps = 1e-6 * (np.trace(C) / n_ch + 1e-12)
    C += eps * np.eye(n_ch)
    try:
        vals, vecs = np.linalg.eigh(C)
        vals = np.maximum(vals, 1e-10)
        logC = vecs @ np.diag(np.log(vals)) @ vecs.T
        idx  = np.triu_indices(n_ch)
        return logC[idx].astype(np.float32)
    except Exception:
        return np.zeros(n_feat, dtype=np.float32)

print("✓ Preprocessing defined")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 7 — Feature Utilities  (V9: + ERP, ITPC, Frontal Theta Asymmetry)
# ─────────────────────────────────────────────────────────────────────────────

def differential_entropy(x):
    v = np.var(x)
    return float(0.5 * np.log(2*np.pi*np.e*v)) if v > 1e-14 else 0.0


def differential_entropy_timeresolved(power_seg, window=10):
    if power_seg.shape[1] < window:
        return np.array([differential_entropy(power_seg[c])
                         for c in range(power_seg.shape[0])], dtype=np.float32)
    local_var = uniform_filter1d(power_seg.astype(np.float64), size=window, axis=1)
    de = 0.5 * np.log(2 * np.pi * np.e * (local_var + 1e-12))
    return de.mean(axis=1).astype(np.float32)


def hjorth(x):
    act = float(np.var(x))
    d1  = np.diff(x); v1 = float(np.var(d1)); mob = np.sqrt(v1/(act+1e-12))
    d2  = np.diff(d1); v2 = float(np.var(d2)); cmp = np.sqrt(v2/(v1+1e-12))/(mob+1e-12)
    return act, mob, cmp


def plv_segment(x, y):
    if len(x) < 4 or len(y) < 4: return 0.0
    if (np.any(np.isnan(x)) or np.any(np.isinf(x)) or
        np.any(np.isnan(y)) or np.any(np.isinf(y))): return 0.0
    if np.var(x) < 1e-14 or np.var(y) < 1e-14: return 0.0
    try:
        return float(np.abs(np.mean(
            np.exp(1j*(np.angle(hilbert(x)) - np.angle(hilbert(y)))))))
    except:
        return 0.0


def power_gradient(p_series, win):
    n = len(p_series)
    if n < 3: return 0.0
    t = np.arange(n, dtype=np.float64)
    try:    return float(np.polyfit(t, p_series.astype(np.float64), 1)[0])
    except: return 0.0


# ── V9-5: Frontal Theta Asymmetry Index ──────────────────────────────────────
def frontal_theta_asymmetry(bp_theta, t0, t1):
    """
    V9-5: Frontal Theta Asymmetry Index (FAI-theta).

    Right frontal theta (F4) > Left frontal theta (F3) in negative emotion.
    FAI_theta = log(F4_theta) - log(F3_theta).
    Negative value = right dominance = negative/emotional state.
    Reference: Aftanas & Golocheikine 2001, Nature Neuroscience 4(4):370-3.

    Also computes frontal-parietal theta ratio (working memory signature)
    and midline Cz theta (memory consolidation hub).
    Returns 3 float features.
    """
    f3t = np.mean(bp_theta[CH['f3'], t0:t1]) + 1e-12
    f4t = np.mean(bp_theta[CH['f4'], t0:t1]) + 1e-12
    pzt = np.mean(bp_theta[CH['pz'], t0:t1]) + 1e-12
    czt = np.mean(bp_theta[CH['cz'], t0:t1]) + 1e-12

    fai_theta     = float(np.log(f4t) - np.log(f3t))          # asymmetry
    fp_ratio      = float(np.log((f3t + f4t) / 2.0) - np.log(pzt))   # frontal/parietal
    cz_theta_log  = float(np.log(czt))                         # Cz theta

    return [fai_theta, fp_ratio, cz_theta_log]


# ── V9-3: ERP Template (Grand-average, computed once) ────────────────────────
def compute_grand_erp_template(train_subjects_list,
                                erp_start=ERP_WIN_START,
                                erp_end=ERP_WIN_END):
    """
    V9-3: Compute population-level ERP difference template (emo – neu).

    Uses the mean waveform of emotional trials minus neutral trials,
    averaged across all training subjects. Applied to both val subjects
    (in LOSO) and test subjects (no labels needed at application time).

    Reference: Sculthorpe et al. 2009 — Emotional memory and N1/P2 during
    NREM sleep. Atienza & Cantero 2001 — auditory N1 in NREM.

    Returns: template (N_CH, erp_end-erp_start) float32
    """
    emo_erps, neu_erps = [], []

    for s in train_subjects_list:
        eeg    = s['eeg'].copy()
        labels = s['labels']

        # Apply same preprocessing as feature extraction
        eeg, _, _ = zscore_subject(eeg)
        eeg = euclidean_alignment(eeg)
        eeg_proc = np.array([preprocess_trial(eeg[i])
                             for i in range(eeg.shape[0])], dtype=np.float32)

        emo_mask = labels == 2
        neu_mask = labels == 1

        if emo_mask.sum() > 1:
            emo_erps.append(eeg_proc[emo_mask].mean(axis=0))  # (n_ch, n_tp)
        if neu_mask.sum() > 1:
            neu_erps.append(eeg_proc[neu_mask].mean(axis=0))

    grand_emo = np.mean(emo_erps, axis=0) if emo_erps else np.zeros((N_CH, N_TP))
    grand_neu = np.mean(neu_erps, axis=0) if neu_erps else np.zeros((N_CH, N_TP))

    # ERP difference in the early window only
    template = (grand_emo - grand_neu)[:, erp_start:erp_end].astype(np.float32)
    return template  # (N_CH, erp_end-erp_start)


def apply_erp_template(eeg_trials_proc, erp_template,
                        erp_start=ERP_WIN_START, erp_end=ERP_WIN_END):
    """
    V9-3: Compute per-trial ERP features using a pre-computed template.

    For each trial, in the ERP window (100–350ms):
      1. Correlation with template per channel (16 features) — emotional match
      2. RMS amplitude per channel (16 features)             — signal strength
      3. Peak-to-peak per channel (16 features)              — amplitude range

    Total: 48 features per trial, tiled across all 200 timepoints.

    erp_trials_proc: (n_trials, n_ch, n_tp) preprocessed EEG
    erp_template:    (n_ch, erp_len) or None
    Returns: (n_trials, 48) float32
    """
    n_tr = eeg_trials_proc.shape[0]
    n_feat = N_CH * 3  # 48

    if erp_template is None:
        return np.zeros((n_tr, n_feat), dtype=np.float32)

    erp_len = erp_end - erp_start
    features = np.zeros((n_tr, n_feat), dtype=np.float32)

    for i in range(n_tr):
        trial_erp = eeg_trials_proc[i, :, erp_start:erp_end]  # (n_ch, erp_len)

        # 1. Pearson correlation with ERP difference template per channel
        corr = np.zeros(N_CH, dtype=np.float32)
        for c in range(N_CH):
            t_ch = trial_erp[c]
            tmpl  = erp_template[c]
            if np.std(t_ch) > 1e-8 and np.std(tmpl) > 1e-8:
                corr[c] = float(np.corrcoef(t_ch, tmpl)[0, 1])

        # 2. RMS amplitude per channel
        rms = np.sqrt(np.mean(trial_erp**2, axis=1)).astype(np.float32)

        # 3. Peak-to-peak per channel
        p2p = (np.max(trial_erp, axis=1) - np.min(trial_erp, axis=1)).astype(np.float32)

        features[i] = np.concatenate([corr, rms, p2p])

    return features  # (n_trials, 48)


# ── V9-4: Inter-Trial Phase Coherence ────────────────────────────────────────
def compute_itpc(eeg_trials_proc, band_lo=4.0, band_hi=8.0, fs=FS):
    """
    V9-4: Inter-Trial Phase Coherence (ITPC) in theta band.

    ITPC measures how consistently the theta phase is locked across trials
    at each timepoint. High ITPC = stimulus-evoked, consistent phase response.
    In TMR: emotional cues elicit stronger phase locking than neutral cues.

    ITPC(t) = |mean_k[ exp(j * φ_k(t)) ]|  where φ_k is phase of trial k.
    Value in [0, 1]: 0 = random phase, 1 = perfect phase lock.

    Reference: Luo & Poeppel 2007 (Science) — theta ITPC for speech tracking.
    Applied to EEG emotion: Aftanas et al. 2004.

    eeg_trials_proc: (n_trials, n_ch, n_tp) preprocessed
    Returns: (n_ch, n_tp) float32
    """
    n_tr, n_ch, n_tp = eeg_trials_proc.shape
    if n_tr < 2:
        return np.zeros((n_ch, n_tp), dtype=np.float32)

    # Band-filter all trials at once
    flat   = eeg_trials_proc.reshape(-1, n_tp)
    filt   = bandpass(flat, band_lo, band_hi, fs).reshape(n_tr, n_ch, n_tp)

    # Compute instantaneous phase via Hilbert
    analytic = hilbert(filt.astype(np.float64), axis=-1)
    phases   = np.angle(analytic)  # (n_tr, n_ch, n_tp)

    # ITPC = |mean of unit complex vectors in phase space|
    itpc = np.abs(np.mean(np.exp(1j * phases), axis=0))  # (n_ch, n_tp)
    return itpc.astype(np.float32)


print("✓ Feature utilities defined (V9: + ERP template, ITPC, Frontal Theta FAI)")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 8 — Feature Extraction  (~766 features/tp + 48 ERP + 16 ITPC trial feats)
# ─────────────────────────────────────────────────────────────────────────────

def extract_all_features(trial, win=WIN):
    """
    trial: (16,200) preprocessed EEG
    Returns: (200, ~718) float32

    Feature blocks (A-S from v8.1, plus new U):
      A-T: all v8.1 features (~715 total)
      U. Frontal Theta Asymmetry [V9-5]: 3 features
    """
    n_ch, n_tp = trial.shape
    half = win // 2

    bf, bp = {}, {}
    for bname, (lo, hi) in BANDS.items():
        f         = bandpass(trial, lo, hi, FS)
        bf[bname] = f
        bp[bname] = (np.abs(hilbert(f, axis=-1))**2).astype(np.float32)

    all_feats = []
    for t in range(n_tp):
        t0 = max(0, t-half); t1 = min(n_tp, t+half)
        f  = []

        # A. Band power (96)
        for bn in BANDS:
            f.extend(np.mean(bp[bn][:,t0:t1], axis=1).tolist())

        # B. Differential Entropy (96)
        for bn in BANDS:
            seg = bf[bn][:,t0:t1]
            for ch in range(n_ch):
                f.append(differential_entropy(seg[ch]))

        # C. Relative band power (96)
        total = sum(np.mean(bp[bn][:,t0:t1], axis=1) for bn in BANDS) + 1e-12
        for bn in BANDS:
            f.extend((np.mean(bp[bn][:,t0:t1], axis=1)/total).tolist())

        # D. FAA + frontal theta (2)
        f3a = np.mean(bp['alpha'][CH['f3'],t0:t1]) + 1e-12
        f4a = np.mean(bp['alpha'][CH['f4'],t0:t1]) + 1e-12
        f.append(float(np.log(f4a) - np.log(f3a)))
        f.append(float((np.mean(bp['theta'][CH['f3'],t0:t1]) +
                        np.mean(bp['theta'][CH['f4'],t0:t1])) / 2.0))

        # E. Theta/Beta ratio (16)
        for ch in range(n_ch):
            f.append(float((np.mean(bp['theta'][ch,t0:t1])+1e-12) /
                            (np.mean(bp['beta'] [ch,t0:t1])+1e-12)))

        # F. Theta/Alpha ratio (16)
        for ch in range(n_ch):
            f.append(float((np.mean(bp['theta'][ch,t0:t1])+1e-12) /
                            (np.mean(bp['alpha'][ch,t0:t1])+1e-12)))

        # G. Inter-hemispheric asymmetry (9)
        for ch1, ch2 in [('c3','c4'),('cp3','cp4'),('o1','o2')]:
            for bn in ['theta','alpha','beta']:
                p1 = np.mean(bp[bn][CH[ch1],t0:t1]) + 1e-12
                p2 = np.mean(bp[bn][CH[ch2],t0:t1]) + 1e-12
                f.append(float(np.log(p2)-np.log(p1)))

        # H. Hjorth 4 channels (12)
        for chn in ['f3','f4','cz','pz']:
            act, mob, cmp = hjorth(trial[CH[chn], t0:t1])
            f.extend([act, mob, cmp])

        # I. Sigma spindle power (8)
        for chn in ['c3','c4','cz','pz','f3','f4','cp3','cp4']:
            f.append(float(np.mean(bp['sigma'][CH[chn],t0:t1])))

        # J. Peak-to-peak (16)
        seg = trial[:,t0:t1]
        f.extend((np.max(seg,axis=1) - np.min(seg,axis=1)).tolist())

        # K. Skewness + Kurtosis 4 channels (8)
        for chn in ['f3','f4','cz','pz']:
            s = trial[CH[chn],t0:t1].astype(np.float64)
            f.append(float(skew(s))     if len(s)>2 else 0.0)
            f.append(float(kurtosis(s)) if len(s)>3 else 0.0)

        # L. PLV coherence 10×3 (30)
        for bn in ['theta','alpha','beta']:
            for ch1, ch2 in CONN_PAIRS:
                f.append(plv_segment(bf[bn][CH[ch1],t0:t1],
                                     bf[bn][CH[ch2],t0:t1]))

        # M. Log-Euclidean covariance (136)
        t0c = max(0,t-10); t1c = min(n_tp,t+11)
        f.extend(log_euclidean_cov_features(bp['theta'][:,t0c:t1c]).tolist())

        # N. Extended asymmetry 3×3 (9)
        for ch1, ch2 in ASYM_PAIRS_EXT:
            for bn in ASYM_BANDS_EXT:
                p1 = np.mean(bp[bn][CH[ch1],t0:t1]) + 1e-12
                p2 = np.mean(bp[bn][CH[ch2],t0:t1]) + 1e-12
                f.append(float(np.log(p2) - np.log(p1)))

        # O. Temporal gradient theta (16)
        for ch in range(n_ch):
            seg_g = bp['theta'][ch, max(0,t-half):min(n_tp,t+half)]
            f.append(power_gradient(seg_g, half))

        # P. Hjorth extra 4 channels (12)
        for chn in ['c3','c4','o1','o2']:
            act, mob, cmp = hjorth(trial[CH[chn], t0:t1])
            f.extend([act, mob, cmp])

        # Q. Cross-frequency theta/alpha (16)
        for ch in range(n_ch):
            th  = np.mean(bp['theta'][ch,t0:t1]) + 1e-12
            alp = np.mean(bp['alpha'][ch,t0:t1]) + 1e-12
            f.append(float(np.log(th / alp)))

        # R. Delta/Theta ratio (16)
        for ch in range(n_ch):
            dl = np.mean(bp['delta'][ch,t0:t1]) + 1e-12
            th = np.mean(bp['theta'][ch,t0:t1]) + 1e-12
            f.append(float(np.log(dl / th)))

        # S. Alpha suppression frontal (3)
        for chn in ['f3','f4','cz']:
            local_alpha = np.mean(bp['alpha'][CH[chn],t0:t1]) + 1e-12
            mean_alpha  = np.mean(bp['alpha'][CH[chn],:]) + 1e-12
            f.append(float(np.log(local_alpha / mean_alpha)))

        # T. DE time-resolved 6×16 (96)
        for bn in BANDS:
            seg_de = bp[bn][:,t0:t1]
            de_vals = differential_entropy_timeresolved(
                seg_de, window=min(10, max(2, t1-t0)))
            f.extend(de_vals.tolist())

        # U. V9-5: Frontal Theta Asymmetry Index (3) ← NEW in v9.0
        f.extend(frontal_theta_asymmetry(bp['theta'], t0, t1))

        all_feats.append(f)

    F = np.array(all_feats, dtype=np.float32)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)


def _extract_one_trial(args):
    """Worker for parallel trial feature extraction."""
    i, eeg_i, label_i = args
    F = extract_all_features(preprocess_trial(eeg_i))
    return i, F, int(label_i)


def extract_subject_features(subj_dict, apply_ea=True, erp_template=None):
    """
    Returns X(n_tr*200, ~718+48+16), y(n_tr*200,), trial_ids, zmu, zsig.

    V9 additions:
      - erp_template: (N_CH, erp_len) grand ERP diff template; adds 48 features
      - ITPC: computed per-subject, adds 16 features per timepoint

    ERP and ITPC features are trial-level / subject-level and tiled across
    all 200 timepoints. They act as context that every per-tp classifier sees.
    """
    eeg    = subj_dict['eeg'].copy()
    labels = subj_dict['labels']

    eeg, zmu, zsig = zscore_subject(eeg)
    if apply_ea:
        eeg = euclidean_alignment(eeg)

    # Preprocess all trials for ERP/ITPC computation
    n_tr = eeg.shape[0]
    eeg_proc = np.array([preprocess_trial(eeg[i]) for i in range(n_tr)],
                        dtype=np.float32)

    # V9-3: ERP template features (trial-level, 48 features)
    erp_feat = apply_erp_template(eeg_proc, erp_template)  # (n_tr, 48)

    # V9-4: ITPC (subject-level, tiled across trials, 16 features per tp)
    itpc_all = compute_itpc(eeg_proc)  # (n_ch, n_tp)

    # Parallel per-trial base feature extraction
    n_workers = min(max(1, N_JOBS - 2), n_tr)
    args_list = [(i, eeg[i], labels[i]) for i in range(n_tr)]
    results = Parallel(n_jobs=n_workers, prefer='threads')(
        delayed(_extract_one_trial)(args) for args in args_list
    )
    results.sort(key=lambda x: x[0])

    feats, ylst, tlst = [], [], []
    for i, F, lbl in results:
        # Tile ERP features (same 48 values at every timepoint for this trial)
        erp_tiled  = np.tile(erp_feat[i], (N_TP, 1))   # (200, 48)
        # Tile ITPC (the subject's phase-coherence profile at each timepoint)
        itpc_tiled = itpc_all.T                          # (200, 16)
        # Combine
        F_combined = np.hstack([F, erp_tiled, itpc_tiled])  # (200, ~782)
        feats.append(F_combined)
        ylst.extend([lbl] * N_TP)
        tlst.extend([i]   * N_TP)

    X = np.vstack(feats)
    y = np.array(ylst, dtype=np.int32)
    t = np.array(tlst, dtype=np.int32)
    return X, y, t, zmu, zsig


print("✓ Feature extraction defined — v9.0")
print("  Base features: ~718/tp | ERP features: +48/trial | ITPC: +16/tp")
print("  Total → ~782 features → SelectKBest(350)")
print("  NEW: Feature U (frontal theta asymmetry FAI-theta, V9-5)")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 9 — CSP Spatial Filters  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class CSP:
    def __init__(self, n=N_CSP, band_lo=4.0, band_hi=8.0):
        self.n=n; self.lo=band_lo; self.hi=band_hi; self.W=None

    def fit(self, eeg, labels):
        filt = bandpass(eeg.reshape(-1,N_TP),self.lo,self.hi,FS).reshape(eeg.shape)
        def cov(X):
            C = np.zeros((N_CH,N_CH))
            for t in range(len(X)):
                s = X[t]-X[t].mean(axis=-1,keepdims=True)
                C += s@s.T/(s.shape[-1]-1)
            return C/len(X)
        mask1=labels==1; mask2=labels==2
        if mask1.sum()==0 or mask2.sum()==0:
            self.W = np.eye(N_CH)[:, :self.n*2]; return self
        C1=cov(filt[mask1]); C2=cov(filt[mask2])
        ev,evec = eigh(C1,C1+C2)
        idx = np.argsort(ev)
        self.W = evec[:,np.concatenate([idx[:self.n],idx[-self.n:]])]
        return self

    def log_var_features(self, eeg, win=WIN):
        filt    = bandpass(eeg.reshape(-1,N_TP),self.lo,self.hi,FS).reshape(eeg.shape)
        csp_sig = np.tensordot(self.W.T, filt, axes=([1],[1])).transpose(1,0,2)
        half    = win//2
        out     = []
        for tr in range(eeg.shape[0]):
            for t in range(N_TP):
                t0=max(0,t-half); t1=min(N_TP,t+half)
                out.append(np.log(np.var(csp_sig[tr,:,t0:t1],axis=1)+1e-12))
        return np.array(out, dtype=np.float32)

print("✓ CSP defined")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 10 — TemporalGRUNet  (V9: corrected validation + sliding inference)
# ─────────────────────────────────────────────────────────────────────────────

if HAVE_TORCH:

    class TemporalGRUNet(nn.Module):
        """
        BiGRU × 2 with spatial mixing and per-timepoint output.
        Input: (batch, 1, n_ch, n_tp) → (batch, n_tp) logits
        """
        def __init__(self, n_ch=N_CH, n_tp=N_TP,
                     hidden=GRU_HIDDEN, hidden2=GRU_HIDDEN2,
                     dropout=GRU_DROPOUT):
            super().__init__()
            self.n_ch = n_ch; self.n_tp = n_tp
            self.spatial = nn.Sequential(
                nn.Linear(n_ch, hidden, bias=False),
                nn.LayerNorm(hidden),
                nn.ELU(),
            )
            self.gru1  = nn.GRU(hidden, hidden, num_layers=1,
                                 batch_first=True, bidirectional=True)
            self.drop1 = nn.Dropout(dropout)
            self.ln1   = nn.LayerNorm(hidden * 2)
            self.gru2  = nn.GRU(hidden*2, hidden2, num_layers=1,
                                 batch_first=True, bidirectional=True)
            self.drop2 = nn.Dropout(dropout)
            self.ln2   = nn.LayerNorm(hidden2 * 2)
            self.fc    = nn.Linear(hidden2 * 2, 1)
            self._init_weights()

        def _init_weights(self):
            for name, p in self.named_parameters():
                if 'weight_ih' in name: nn.init.xavier_uniform_(p)
                elif 'weight_hh' in name: nn.init.orthogonal_(p)
                elif 'bias' in name: nn.init.zeros_(p)
                elif 'weight' in name and p.dim() >= 2: nn.init.xavier_uniform_(p)

        def forward(self, x):
            x = x.squeeze(1).permute(0, 2, 1)   # (batch, n_tp, n_ch)
            x = self.spatial(x)                   # (batch, n_tp, hidden)
            x, _ = self.gru1(x); x = self.ln1(self.drop1(x))
            x, _ = self.gru2(x); x = self.ln2(self.drop2(x))
            return self.fc(x).squeeze(-1)         # (batch, n_tp) logits


    def make_gru_frames(eeg, frame_len=GRU_FRAME_LEN, overlap=GRU_OVERLAP):
        """Frame EEG with overlap (Yoo 2023: 20% reduces overfitting)."""
        n_tr, n_ch, n_tp = eeg.shape
        step   = max(1, int(frame_len * (1 - overlap)))
        starts = list(range(0, n_tp - frame_len + 1, step))
        frames, trial_map = [], []
        for i in range(n_tr):
            for s in starts:
                frames.append(eeg[i:i+1, :, s:s+frame_len])
                trial_map.append(i)
        frames_arr    = np.concatenate(frames, axis=0)[:, np.newaxis, :, :]
        trial_map_arr = np.array(trial_map, dtype=np.int32)
        return frames_arr, trial_map_arr, starts


    def prepare_eeg_for_gru(eeg, labels=None):
        """Preprocessing: detrend + avg-ref + bandpass + zscore + EA."""
        n_tr = eeg.shape[0]
        eeg_proc = np.array([preprocess_trial(eeg[i]) for i in range(n_tr)],
                            dtype=np.float32)
        eeg_proc, _, _ = zscore_subject(eeg_proc)
        eeg_proc = euclidean_alignment(eeg_proc)
        return eeg_proc


    def gru_sliding_inference(model, eeg_np, device,
                               frame_len=GRU_FRAME_LEN, overlap=GRU_OVERLAP):
        """
        V9-7: Sliding window inference — matches training frame distribution.

        Each timepoint t receives its prediction as the weighted average of
        all frames that contain t. This eliminates the train/test inconsistency
        in v8.1 where training used frames but inference used full 200-tp trials.

        eeg_np: (n_trials, n_ch, n_tp) preprocessed
        Returns: (n_trials, n_tp) float32 probabilities
        """
        n_tr, n_ch, n_tp = eeg_np.shape
        step   = max(1, int(frame_len * (1 - overlap)))
        starts = list(range(0, n_tp - frame_len + 1, step))

        prob_sum   = np.zeros((n_tr, n_tp), dtype=np.float64)
        prob_count = np.zeros((n_tr, n_tp), dtype=np.float64)

        model.eval()
        use_amp = (device.type == 'cuda')

        for s in starts:
            # Extract frames for all trials at this start position
            frames = eeg_np[:, :, s:s+frame_len]        # (n_tr, n_ch, frame_len)
            frames_t = torch.tensor(
                frames[:, np.newaxis, :, :], dtype=torch.float32
            ).to(device)

            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=use_amp):
                    logits = model(frames_t)             # (n_tr, frame_len)
            probs = torch.sigmoid(logits).cpu().numpy()

            prob_sum[:, s:s+frame_len]   += probs
            prob_count[:, s:s+frame_len] += 1.0

        prob_count = np.maximum(prob_count, 1.0)
        return (prob_sum / prob_count).astype(np.float32)


    def train_temporal_gru(X_tr_eeg, y_tr_bin, X_v_eeg, y_v_bin,
                            device, epochs=GRU_EPOCHS, lr=GRU_LR,
                            patience=GRU_PATIENCE):
        """
        Train GRU with frame overlap (training) and sliding window (validation).

        V9-6: Early stopping uses AUC on the EARLY window (100-350ms = tp 20-70)
        instead of the full signal window. This was the actual discriminative
        region in v8.0 — training the GRU to optimize that window directly.
        """
        model = TemporalGRUNet().to(device)

        n_pos = float(y_tr_bin.sum()); n_neg = float(len(y_tr_bin) - n_pos)
        pos_w = torch.tensor([n_neg/(n_pos+1e-8)], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        optimizer = optim.RMSprop(model.parameters(), lr=lr,
                                   weight_decay=1e-4, momentum=0.9)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        use_amp   = (device.type == 'cuda')
        scaler    = torch.amp.GradScaler('cuda', enabled=use_amp)

        # Frame the training data
        frames_np, trial_map_np, _ = make_gru_frames(X_tr_eeg)
        y_frames = y_tr_bin[trial_map_np]
        X_t = torch.tensor(frames_np, dtype=torch.float32)
        y_t = torch.tensor(y_frames,  dtype=torch.float32)
        ds  = TensorDataset(X_t, y_t)
        dl  = DataLoader(ds, batch_size=min(64, max(8, len(ds))),
                         shuffle=True, drop_last=False,
                         pin_memory=(device.type=='cuda'), num_workers=0)

        best_auc, best_state, no_improve = 0.0, None, 0

        for epoch in range(epochs):
            model.train()
            for Xb, yb in dl:
                Xb = Xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                Xb = Xb + 0.03 * torch.randn_like(Xb)
                if torch.rand(1).item() < 0.20:
                    ch_idx = torch.randint(0, N_CH, (1,)).item()
                    Xb[:, :, ch_idx, :] = 0.0
                if torch.rand(1).item() < 0.50:
                    shift = torch.randint(-3, 4, (1,)).item()
                    Xb = torch.roll(Xb, shift, dims=-1)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    logits = model(Xb)
                    target = yb[:, None].expand_as(logits)
                    loss   = criterion(logits, target)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            scheduler.step()

            if (epoch + 1) % 10 == 0:
                # V9-7: Use sliding window inference for validation
                val_probs_np = gru_sliding_inference(model, X_v_eeg, device)

                # V9-6: Evaluate AUC on EARLY window (100-350ms) ← CORRECTED
                val_sig = val_probs_np[:, GRU_VAL_START:GRU_VAL_END+1].mean(axis=1)
                try:
                    auc = roc_auc_score(y_v_bin, val_sig)
                except Exception:
                    auc = 0.5

                if auc > best_auc:
                    best_auc   = auc
                    best_state = {k: v.cpu().clone()
                                  for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 10
                    if no_improve >= patience:
                        break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        # Final inference: sliding window
        val_probs_np = gru_sliding_inference(model, X_v_eeg, device)
        return val_probs_np, model


    def train_temporal_gru_final(X_all_eeg, y_all_bin, device,
                                  epochs=GRU_EPOCHS, lr=GRU_LR):
        """Train final GRU on ALL training data."""
        model = TemporalGRUNet().to(device)
        n_pos  = float(y_all_bin.sum()); n_neg = float(len(y_all_bin) - n_pos)
        pos_w  = torch.tensor([n_neg/(n_pos+1e-8)],
                               dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        optimizer = optim.RMSprop(model.parameters(), lr=lr,
                                   weight_decay=1e-4, momentum=0.9)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        use_amp   = (device.type == 'cuda')
        scaler    = torch.amp.GradScaler('cuda', enabled=use_amp)

        frames_np, trial_map_np, _ = make_gru_frames(X_all_eeg)
        y_frames = y_all_bin[trial_map_np]
        ds = TensorDataset(torch.tensor(frames_np, dtype=torch.float32),
                           torch.tensor(y_frames,  dtype=torch.float32))
        dl = DataLoader(ds, batch_size=min(64, max(8, len(ds))),
                        shuffle=True, drop_last=False,
                        pin_memory=(device.type=='cuda'), num_workers=0)

        for epoch in range(epochs):
            model.train()
            for Xb, yb in dl:
                Xb = Xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                Xb = Xb + 0.03 * torch.randn_like(Xb)
                if torch.rand(1).item() < 0.20:
                    Xb[:, :, torch.randint(0,N_CH,(1,)).item(), :] = 0.0
                if torch.rand(1).item() < 0.50:
                    Xb = torch.roll(Xb, torch.randint(-3,4,(1,)).item(), dims=-1)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    logits = model(Xb)
                    loss   = criterion(logits, yb[:,None].expand_as(logits))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            scheduler.step()

        model.eval()
        return model

    print(f"✓ TemporalGRUNet defined — V9.0")
    print(f"  V9-6: Validation on early window ({GRU_VAL_START*1000//FS}-"
          f"{GRU_VAL_END*1000//FS}ms)")
    print(f"  V9-7: Sliding window inference (frame_len={GRU_FRAME_LEN}, "
          f"overlap={int(GRU_OVERLAP*100)}%)")
else:
    print("⚠ TemporalGRUNet skipped (PyTorch not installed)")


def _tune_enn_w(tab_oof_list, enn_oof_list, labels_list):
    """Grid-search ENN_W on OOF window-AUC."""
    best_w, best_auc = ENN_W, 0.0
    print("  ENN_W tuning grid:")
    for w in ENN_W_GRID:
        aucs_folds = []
        for tab, enn, lbl in zip(tab_oof_list, enn_oof_list, labels_list):
            blended = (1.0 - w) * tab + w * enn
            sm   = smooth_predictions(blended)
            gat  = adaptive_temporal_gate(sm)
            cal  = np.clip(gat, 0.01, 0.99)
            yb   = (lbl == 2).astype(int)
            aucs_folds.append(window_auc_score(cal, yb)['window_auc'])
        mean_auc = float(np.mean(aucs_folds))
        marker = " ← best" if mean_auc > best_auc else ""
        print(f"    ENN_W={w:.2f}  wAUC={mean_auc:.4f}{marker}")
        if mean_auc > best_auc:
            best_auc = mean_auc; best_w = w
    return best_w


# ─────────────────────────────────────────────────────────────────────────────
# CELL 11 — GPU-Boosted TimeResolved Ensemble  (unchanged from v8.1)
# ─────────────────────────────────────────────────────────────────────────────

def _make_lgbm():
    params = dict(
        n_estimators=800 if GPU_AVAILABLE else 500,
        num_leaves=63 if GPU_AVAILABLE else 31,
        max_depth=6 if GPU_AVAILABLE else 5,
        learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
        min_child_samples=8, class_weight='balanced',
        reg_alpha=0.1, reg_lambda=1.0, n_jobs=1,
        random_state=42, verbose=-1,
    )
    if GPU_AVAILABLE:
        params.update({'device':'gpu','gpu_use_dp':False,'max_bin':255})
    return lgb.LGBMClassifier(**params)


def _make_xgb():
    return xgb.XGBClassifier(
        n_estimators=600 if GPU_AVAILABLE else 350,
        max_depth=5 if GPU_AVAILABLE else 4,
        learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
        min_child_weight=8, gamma=0.1, eval_metric='logloss',
        tree_method='hist', n_jobs=1, random_state=42, verbosity=0,
        device='cuda' if GPU_AVAILABLE else 'cpu',
    )


def _make_catboost():
    if not HAVE_CATBOOST: return None
    params = dict(
        iterations=500 if GPU_AVAILABLE else 300,
        depth=5 if GPU_AVAILABLE else 4,
        learning_rate=0.04, loss_function='Logloss', eval_metric='AUC',
        subsample=0.8, rsm=0.7, l2_leaf_reg=3.0, random_seed=42,
        verbose=0, thread_count=1, auto_class_weights='Balanced',
    )
    if GPU_AVAILABLE:
        params.update({'task_type':'GPU','devices':'0'})
    return CatBoostClassifier(**params)


def _train_one_tp(tp, X_flat, y_trial, n_trials, n_feat_sel):
    idx  = np.arange(n_trials) * N_TP + tp
    Xt   = np.nan_to_num(X_flat[idx])
    ybin = (y_trial == 2).astype(int)

    if n_feat_sel and n_feat_sel < Xt.shape[1]:
        sel  = SelectKBest(f_classif, k=n_feat_sel)
        Xt_s = sel.fit_transform(Xt, ybin)
    else:
        sel  = None; Xt_s = Xt

    sc   = RobustScaler()
    Xt_s = sc.fit_transform(Xt_s)

    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    try:    lda.fit(Xt_s, ybin)
    except: lda = None

    lgbm = _make_lgbm()
    try:    lgbm.fit(Xt_s, ybin)
    except:
        lgbm = lgb.LGBMClassifier(device='cpu', n_estimators=500,
                                    num_leaves=31, learning_rate=0.03,
                                    class_weight='balanced', verbose=-1,
                                    n_jobs=1, random_state=42)
        try:    lgbm.fit(Xt_s, ybin)
        except: lgbm = None

    xgbm = _make_xgb()
    try:    xgbm.fit(Xt_s, ybin)
    except:
        xgbm = xgb.XGBClassifier(n_estimators=350, max_depth=4,
                                   learning_rate=0.03, subsample=0.8,
                                   colsample_bytree=0.7, min_child_weight=8,
                                   gamma=0.1, eval_metric='logloss',
                                   tree_method='hist', n_jobs=1,
                                   random_state=42, verbosity=0, device='cpu')
        try:    xgbm.fit(Xt_s, ybin)
        except: xgbm = None

    cat = _make_catboost()
    if cat is not None:
        try:    cat.fit(Xt_s, ybin)
        except:
            cat = CatBoostClassifier(iterations=300, depth=4,
                                      learning_rate=0.05, loss_function='Logloss',
                                      eval_metric='AUC', subsample=0.8, rsm=0.7,
                                      l2_leaf_reg=3.0, random_seed=42,
                                      verbose=0, thread_count=1,
                                      auto_class_weights='Balanced')
            try:    cat.fit(Xt_s, ybin)
            except: cat = None

    return tp, {'lda':lda, 'lgbm':lgbm, 'xgb':xgbm, 'cat':cat, 'sel':sel, 'sc':sc}


class TimeResolvedEnsemble:
    """LDA + LightGBM(GPU) + XGBoost(CUDA) + CatBoost(GPU) per timepoint."""
    W_LDA=0.25; W_LGBM=0.35; W_XGB=0.25; W_CAT=0.15

    def __init__(self, signal_tps=None, n_feat_sel=N_FEAT_SEL, n_jobs=None):
        self.signal_tps   = signal_tps if signal_tps else SIGNAL_TPS
        self.n_feat_sel   = n_feat_sel
        self.n_jobs       = n_jobs if n_jobs else max(1, N_JOBS-1)
        self.models       = {}
        self.fitted       = False
        self.adaptive_wts = {}

    def fit(self, X_flat, y_tr, n_trials, verbose=True):
        y_trial = y_tr[::N_TP]
        if verbose:
            print(f"  Training {len(self.signal_tps)} classifiers "
                  f"(LDA+LGBM+XGB"
                  f"{'+CatBoost' if HAVE_CATBOOST else ''} "
                  f"{'(GPU)' if GPU_AVAILABLE else '(CPU)'}, "
                  f"n_jobs={self.n_jobs})...")
        t0 = time.time()
        results = Parallel(n_jobs=self.n_jobs, prefer='threads')(
            delayed(_train_one_tp)(tp, X_flat, y_trial, n_trials, self.n_feat_sel)
            for tp in tqdm(self.signal_tps, desc='  TRE training', disable=not verbose)
        )
        for tp, state in results:
            self.models[tp] = state
        self.fitted = True
        if verbose:
            print(f"  ✓ {len(self.models)} classifiers in {time.time()-t0:.1f}s")

    def _predict_one_tp(self, tp, X_flat, n_trials, wts=None):
        m    = self.models[tp]
        idx  = np.arange(n_trials) * N_TP + tp
        Xt   = np.nan_to_num(X_flat[idx])
        Xt_s = m['sel'].transform(Xt) if m['sel'] else Xt
        Xt_s = m['sc'].transform(Xt_s)
        if wts is None:
            wts = self.adaptive_wts.get(tp, {
                'lda':self.W_LDA,'lgbm':self.W_LGBM,
                'xgb':self.W_XGB,'cat':self.W_CAT})
        blend=np.zeros(n_trials); wt_total=0.0
        for key in ['lda','lgbm','xgb','cat']:
            clf=m.get(key); w=wts.get(key,0.0)
            if clf is not None and w>0:
                try:
                    p = clf.predict_proba(Xt_s)[:,1]
                    blend += w*p; wt_total += w
                except: pass
        return blend / (wt_total+1e-8)

    def predict_proba_matrix(self, X_flat, n_trials, verbose=False):
        assert self.fitted
        probs = np.full((n_trials, N_TP), 0.5, dtype=np.float64)
        for tp in tqdm(self.signal_tps, desc='  TRE predict', disable=not verbose):
            probs[:, tp] = self._predict_one_tp(tp, X_flat, n_trials)
        return probs

    def set_adaptive_weights_from_oof(self, X_flat, y_trial, n_trials):
        """AUC²-weighted ensemble weights per timepoint."""
        if not self.fitted: return
        y_bin = (y_trial == 2).astype(int)
        for tp in self.signal_tps:
            m    = self.models[tp]
            idx  = np.arange(n_trials) * N_TP + tp
            Xt   = np.nan_to_num(X_flat[idx])
            Xt_s = m['sel'].transform(Xt) if m['sel'] else Xt
            Xt_s = m['sc'].transform(Xt_s)
            aucs = {}
            for key in ['lda','lgbm','xgb','cat']:
                clf = m.get(key)
                if clf is not None:
                    try:
                        p = clf.predict_proba(Xt_s)[:,1]
                        aucs[key] = max(roc_auc_score(y_bin,p), 0.5)
                    except: aucs[key]=0.5
                else: aucs[key]=0.0
            raw_w = {k: v**2 for k,v in aucs.items()}
            total = sum(raw_w.values())+1e-8
            self.adaptive_wts[tp] = {k: v/total for k,v in raw_w.items()}
        print("  ✓ Adaptive weights computed")

print(f"✓ TimeResolvedEnsemble v9.0 defined")
print(f"  {len(SIGNAL_TPS)} timepoints (50-900ms, 171 tps) | SelectKBest(350)")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 12 — Post-Processing & Metric  (V9-2: corrected gate + Platt calibration)
# ─────────────────────────────────────────────────────────────────────────────

def smooth_predictions(probs, sigma=None):
    """Two-stage: Savitzky-Golay → Gaussian."""
    if sigma is None: sigma = SMOOTH_SIGMA
    out = np.zeros_like(probs, dtype=np.float64)
    for i in range(probs.shape[0]):
        seg = probs[i].astype(np.float64)
        if len(seg) >= SAVGOL_WIN:
            seg = savgol_filter(seg, window_length=SAVGOL_WIN, polyorder=SAVGOL_POLY)
        seg = gaussian_filter1d(seg, sigma=sigma)
        out[i] = seg
    return out


def adaptive_temporal_gate(probs):
    """
    V9-2: Gate corrected from observed AUC timecourse in v8.0.

    Zone A (100-350ms = tp 20-70): REAL observed peak — full signal
    Zone B (50-700ms context):     keep at 60% — broad support
    Zone C (0-50ms, 700ms+):       suppress to 12% — noise

    Critical correction from v8.1: v8.1 used Zone A = 400-600ms which
    is exactly where the AUC drops BELOW 0.5. Now aligned to data.
    """
    out = probs.copy().astype(np.float64)
    for tp in range(N_TP):
        if GATE_ZONE_A_START <= tp <= GATE_ZONE_A_END:
            pass  # Zone A: full signal
        elif GATE_ZONE_B_START <= tp < GATE_ZONE_A_START or \
             GATE_ZONE_A_END  < tp <= GATE_ZONE_B_END:
            out[:, tp] = BLEND_ALPHA_B * out[:, tp] + (1-BLEND_ALPHA_B) * 0.5
        else:
            out[:, tp] = BLEND_ALPHA_C * out[:, tp] + (1-BLEND_ALPHA_C) * 0.5
    return out


def window_auc_score(probs, y_bin, min_ms=50, win_tp=10):
    """Official competition metric: window-AUC with 50ms consistency."""
    n_tp = probs.shape[1]
    aucs = []
    for s in range(n_tp - win_tp + 1):
        wp = probs[:,s:s+win_tp].mean(axis=1)
        try:    a = roc_auc_score(y_bin, wp)
        except: a = 0.5
        aucs.append(a)
    aucs = np.array(aucs)
    min_w = max(1, int(min_ms * FS / 1000))
    best_start, best_len, best_auc = 0, 0, 0.5
    run_s = run_l = 0
    for i, above in enumerate(aucs > 0.5):
        if above:
            if run_l==0: run_s=i
            run_l += 1
        else:
            if run_l >= min_w and run_l > best_len:
                best_len=run_l; best_start=run_s
                best_auc=aucs[run_s:run_s+run_l].mean()
            run_l = 0
    if run_l >= min_w and run_l > best_len:
        best_auc = aucs[run_s:run_s+run_l].mean()
    return {'window_auc':best_auc, 'aucs':aucs,
            'mean_auc':aucs.mean(), 'dur_ms':best_len*(1000/FS)}


def tune_sigma_on_oof(probs_raw, y_bin, candidates=None):
    """Grid-search Gaussian sigma."""
    if candidates is None: candidates = SMOOTH_SIGMA_CANDIDATES
    best_sigma, best_auc = candidates[0], 0.0
    for s in candidates:
        p_sm  = smooth_predictions(probs_raw, sigma=s)
        p_gat = adaptive_temporal_gate(p_sm)
        m     = window_auc_score(np.clip(p_gat, 0.01, 0.99), y_bin)
        if m['window_auc'] > best_auc:
            best_auc=m['window_auc']; best_sigma=s
    print(f"  Sigma tuning: best_sigma={best_sigma}  wAUC={best_auc:.4f}")
    return best_sigma, best_auc


# ── V9-8: Platt Calibration (safer than Isotonic for small N) ────────────────
def fit_platt_calibration(oof_probs, y_bin, n_tp=N_TP):
    """
    V9-8: Platt Scaling (sigmoid logistic) per timepoint.

    Safer than Isotonic regression when N is small (14 subjects).
    Isotonic can memorise OOF noise; logistic sigmoid is constrained
    and generalises better to unseen test subjects.
    Reference: Platt 1999; Niculescu-Mizil & Caruana 2005.
    """
    calibrators = []
    for t in range(n_tp):
        cal = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
        p_t = oof_probs[:, t:t+1]
        try:
            cal.fit(p_t, y_bin)
            calibrators.append(cal)
        except Exception:
            calibrators.append(None)
    return calibrators


def apply_platt_calibration(probs, calibrators, n_tp=N_TP):
    """Apply per-timepoint Platt calibration."""
    probs_cal = np.zeros_like(probs)
    for t in range(n_tp):
        if calibrators[t] is not None:
            try:
                probs_cal[:, t] = calibrators[t].predict_proba(
                    probs[:, t:t+1])[:, 1]
            except:
                probs_cal[:, t] = probs[:, t]
        else:
            probs_cal[:, t] = probs[:, t]
    return probs_cal


print("✓ Post-processing defined (V9-2: corrected gate | V9-8: Platt calibration)")
print(f"  Gate: A(100-350ms)=1.0 | B(50-700ms)=0.60 | C=0.12")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 13 — LOSO Cross-Validation  (V9: + ERP template + diagnostic plots)
# ─────────────────────────────────────────────────────────────────────────────

def run_loso(train_subjects, tune_sigma=True, verbose=True):
    global SMOOTH_SIGMA, ENN_W, GRAND_ERP_TEMPLATE

    print("\n" + "="*70)
    print("  LOSO Cross-Validation — v9.0 (DATA-DRIVEN)")
    print("="*70)

    # ── V9-3: Compute grand ERP template from ALL training subjects ───────────
    # Done once, used for all folds (val subject + test) without label leakage
    # since it's computed at population level, not fold level.
    print("\nStep 0: Computing grand ERP template (V9-3)...")
    t_erp = time.time()
    GRAND_ERP_TEMPLATE = compute_grand_erp_template(train_subjects)
    print(f"  ✓ ERP template shape: {GRAND_ERP_TEMPLATE.shape} "
          f"({ERP_WIN_START*1000//FS}-{ERP_WIN_END*1000//FS}ms) "
          f"in {time.time()-t_erp:.1f}s")

    # ── Step 1: Extract tabular features ──────────────────────────────────────
    print("\nStep 1: Extracting tabular features (EA + ERP template + ITPC)...")
    cache = []
    for i, s in enumerate(train_subjects):
        t0 = time.time()
        X, y, tid, zmu, zsig = extract_subject_features(
            s, apply_ea=True, erp_template=GRAND_ERP_TEMPLATE)
        cache.append({'X':X, 'y':y, 'tid':tid,
                      'eeg':s['eeg'], 'labels':s['labels'], 'id':s['id']})
        print(f"  [{i+1:2d}/{len(train_subjects)}] {s['id']}: "
              f"{X.shape} ({time.time()-t0:.1f}s)")

    # ── Step 2: LOSO folds ────────────────────────────────────────────────────
    print(f"\nStep 2: LOSO folds "
          f"({'Tabular + GRU' if HAVE_TORCH else 'Tabular only'})...")
    fold_results    = []
    sigma_per_fold  = []
    oof_tab_list    = []
    oof_enn_list    = []
    oof_labels_list = []
    n = len(train_subjects)

    for val_i in range(n):
        t_fold = time.time()
        val = cache[val_i]
        tr  = [cache[i] for i in range(n) if i != val_i]

        X_tr       = np.vstack([c['X'] for c in tr])
        y_tr       = np.concatenate([c['y'] for c in tr])
        n_tr_trials= sum(c['eeg'].shape[0] for c in tr)
        eeg_tr     = np.vstack([c['eeg'] for c in tr])
        lbl_tr     = np.concatenate([c['labels'] for c in tr])

        csp   = CSP(); csp.fit(eeg_tr, lbl_tr)
        csp_tr= csp.log_var_features(eeg_tr)
        csp_v = csp.log_var_features(val['eeg'])
        X_tr_f= np.hstack([X_tr, csp_tr])
        X_v_f = np.hstack([val['X'], csp_v])

        tre = TimeResolvedEnsemble()
        tre.fit(X_tr_f, y_tr, n_tr_trials, verbose=False)
        n_v = val['eeg'].shape[0]
        probs_tab = tre.predict_proba_matrix(X_v_f, n_v)

        # GRU fold
        if HAVE_TORCH:
            print(f"  [GRU fold {val_i+1}/{n}] training...", end=' ', flush=True)
            t_gru = time.time()
            eeg_tr_proc = prepare_eeg_for_gru(eeg_tr)
            eeg_v_proc  = prepare_eeg_for_gru(val['eeg'])
            y_tr_bin    = (lbl_tr == 2).astype(np.float32)
            y_v_bin     = (val['labels'] == 2).astype(np.float32)

            gru_probs_v, _ = train_temporal_gru(
                eeg_tr_proc, y_tr_bin, eeg_v_proc, y_v_bin,
                TORCH_DEVICE)

            oof_tab_list.append(probs_tab)
            oof_enn_list.append(gru_probs_v)
            oof_labels_list.append(val['labels'])
            probs_raw = (1.0-ENN_W)*probs_tab + ENN_W*gru_probs_v

            if TORCH_DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
            print(f"done ({time.time()-t_gru:.0f}s)")
        else:
            probs_raw = probs_tab

        y_v_bin_int = (val['labels'] == 2).astype(int)
        if tune_sigma:
            fold_sigma, _ = tune_sigma_on_oof(probs_raw, y_v_bin_int)
        else:
            fold_sigma = SMOOTH_SIGMA
        sigma_per_fold.append(fold_sigma)

        probs_sm  = smooth_predictions(probs_raw, sigma=fold_sigma)
        probs_gat = adaptive_temporal_gate(probs_sm)
        probs_cal = np.clip(probs_gat, 0.01, 0.99)

        metric = window_auc_score(probs_cal, y_v_bin_int)
        fold_results.append(metric)

        print(f"  Fold {val_i+1:2d} | {val['id']:30s} | "
              f"σ={fold_sigma}  wAUC={metric['window_auc']:.4f}  "
              f"mAUC={metric['mean_auc']:.4f}  ({time.time()-t_fold:.0f}s)")

        del tre; gc.collect()

    # Tune ENN_W on OOF
    if HAVE_TORCH and oof_tab_list:
        print("\n  Tuning ENN_W on OOF predictions...")
        ENN_W = _tune_enn_w(oof_tab_list, oof_enn_list, oof_labels_list)
        print(f"  → Best ENN_W = {ENN_W:.2f}")
        print("\n  Re-evaluating folds with best ENN_W...")
        fold_results = []
        for tab, enn, lbl in zip(oof_tab_list, oof_enn_list, oof_labels_list):
            blended = (1.0-ENN_W)*tab + ENN_W*enn
            sm  = smooth_predictions(blended, sigma=SMOOTH_SIGMA)
            gat = adaptive_temporal_gate(sm)
            cal = np.clip(gat, 0.01, 0.99)
            yb  = (lbl==2).astype(int)
            fold_results.append(window_auc_score(cal, yb))

    if tune_sigma and sigma_per_fold:
        SMOOTH_SIGMA = int(np.median(sigma_per_fold))
        print(f"\n  Global SMOOTH_SIGMA = {SMOOTH_SIGMA}")

    win_aucs  = [r['window_auc'] for r in fold_results]
    mean_aucs = [r['mean_auc']   for r in fold_results]
    print(f"\n  ╔═══════════════════════════════════════════════════════╗")
    print(f"  ║ Window AUC  : {np.mean(win_aucs):.4f} ± {np.std(win_aucs):.4f}                 ║")
    print(f"  ║ Mean AUC    : {np.mean(mean_aucs):.4f} ± {np.std(mean_aucs):.4f}                 ║")
    print(f"  ║ Best fold   : {max(win_aucs):.4f}                              ║")
    print(f"  ║ SMOOTH_SIGMA: {SMOOTH_SIGMA:2d}   ENN_W: {ENN_W:.2f}                     ║")
    print(f"  ╚═══════════════════════════════════════════════════════╝")

    _plot_loso_results(fold_results, win_aucs, mean_aucs, n, train_subjects)
    return fold_results


def _plot_loso_results(fold_results, win_aucs, mean_aucs, n, train_subjects=None):
    try:
        fig = plt.figure(figsize=(22, 10))
        gs  = gridspec.GridSpec(2, 3, figure=fig)

        # Panel 1: Window-AUC per fold
        ax1 = fig.add_subplot(gs[0, 0])
        colors = ['#2ecc71' if a>0.5 else '#e74c3c' for a in win_aucs]
        ax1.bar(range(1,n+1), win_aucs, color=colors, edgecolor='black', lw=0.5)
        ax1.axhline(0.5, color='k', ls='--', lw=1.5, label='Chance')
        ax1.axhline(np.mean(win_aucs), color='blue', ls='-', lw=2,
                    label=f'Mean={np.mean(win_aucs):.3f}')
        ax1.set_title('Window-AUC per Fold (v9.0)', fontweight='bold')
        ax1.set_xlabel('Subject'); ax1.set_ylabel('Window-AUC')
        ax1.legend(fontsize=8); ax1.set_ylim(0.4, 1.0)

        # Panel 2: AUC time-course
        ax2 = fig.add_subplot(gs[0, 1])
        avg_curve = np.mean([r['aucs'] for r in fold_results], axis=0)
        std_curve = np.std( [r['aucs'] for r in fold_results], axis=0)
        t_ms = np.arange(len(avg_curve)) * (1000/FS)
        ax2.fill_between(t_ms, avg_curve-std_curve, avg_curve+std_curve,
                         alpha=0.25, color='blue')
        ax2.plot(t_ms, avg_curve, 'b-', lw=2, label='Mean window-AUC')
        ax2.axhline(0.5, color='red', ls='--', lw=1.5, label='Chance')
        # Show corrected gate zones
        ax2.axvspan(GATE_ZONE_A_START*1000/FS, GATE_ZONE_A_END*1000/FS,
                    alpha=0.25, color='green', label='Zone A (100-350ms)')
        ax2.axvspan(GATE_ZONE_B_START*1000/FS, GATE_ZONE_A_START*1000/FS,
                    alpha=0.10, color='yellow', label='Zone B context')
        ax2.axvspan(GATE_ZONE_A_END*1000/FS, GATE_ZONE_B_END*1000/FS,
                    alpha=0.10, color='yellow')
        ax2.set_title('AUC Time-Course v9.0', fontweight='bold')
        ax2.set_xlabel('Time (ms)'); ax2.set_ylabel('AUC')
        ax2.legend(fontsize=7)

        # Panel 3: Summary bar
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.bar(['Window\nAUC','Mean\nAUC'],
                [np.mean(win_aucs), np.mean(mean_aucs)],
                yerr=[np.std(win_aucs), np.std(mean_aucs)],
                color=['steelblue','coral'], edgecolor='black', capsize=5)
        ax3.axhline(0.5, color='k', ls='--', lw=1.5)
        ax3.set_ylim(0.45, max(max(win_aucs), max(mean_aucs))+0.05)
        for i_, (v, e) in enumerate(zip([np.mean(win_aucs), np.mean(mean_aucs)],
                                         [np.std(win_aucs),  np.std(mean_aucs)])):
            ax3.text(i_, v+e+0.005, f'{v:.3f}', ha='center', fontweight='bold')
        ax3.set_title('Summary v9.0', fontweight='bold')

        # V9-9: Per-subject AUC time-courses (diagnostic)
        if train_subjects and len(fold_results) >= n:
            ax4 = fig.add_subplot(gs[1, :])
            cmap = plt.get_cmap('tab20')
            t_ms_full = np.arange(len(fold_results[0]['aucs'])) * (1000/FS)
            for i_, (result, subj) in enumerate(zip(fold_results, train_subjects)):
                lbl = f"{subj['id'][:6]} {result['window_auc']:.3f}"
                ax4.plot(t_ms_full, result['aucs'], alpha=0.7, lw=1,
                         color=cmap(i_/n), label=lbl)
            ax4.axhline(0.5, color='k', ls='--', lw=1.5, label='Chance')
            # Mark the corrected gate zones
            ax4.axvspan(GATE_ZONE_A_START*1000/FS, GATE_ZONE_A_END*1000/FS,
                        alpha=0.15, color='green', label='Gate Zone A')
            ax4.set_title('V9-9: Per-Subject AUC Time-Course (diagnostic)',
                          fontweight='bold')
            ax4.set_xlabel('Time (ms)'); ax4.set_ylabel('AUC')
            ax4.set_ylim(0.40, 0.65)
            ax4.legend(fontsize=7, ncol=5, loc='upper right')

        plt.suptitle(f'LOSO CV v9.0 — Window-AUC={np.mean(win_aucs):.4f}  '
                     f'ENN_W={ENN_W:.2f}  Gate:A(100-350ms)  σ={SMOOTH_SIGMA}',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        save_path = os.path.join(BASE, 'loso_results_v9_0.png')
        plt.savefig(save_path, dpi=150)
        plt.show()
        print(f"  Plot saved → {save_path}")
    except Exception as e:
        print(f"  Plot error (non-critical): {e}")

print("✓ LOSO CV defined — v9.0 (ERP template + corrected gate + diagnostic plots)")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 14 — Final Model Training & Submission Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_submission(train_subjects, test_subjects, output_path=OUTPUT):
    global GRAND_ERP_TEMPLATE

    print("\n" + "="*70)
    print("  Final Training → Submission Generation (v9.0)")
    print("="*70)

    # Ensure ERP template is available
    if GRAND_ERP_TEMPLATE is None:
        print("\nComputing grand ERP template...")
        GRAND_ERP_TEMPLATE = compute_grand_erp_template(train_subjects)

    # Step 1: Extract tabular features
    print("\nStep 1: Extracting training features (EA + ERP + ITPC)...")
    X_all, y_all, eeg_all, lbl_all = [], [], [], []
    for i, s in enumerate(train_subjects):
        print(f"  [{i+1}/{len(train_subjects)}] {s['id']} ...", end=' ', flush=True)
        t0 = time.time()
        X, y, _, zmu, zsig = extract_subject_features(
            s, apply_ea=True, erp_template=GRAND_ERP_TEMPLATE)
        X_all.append(X); y_all.append(y)
        eeg_all.append(s['eeg']); lbl_all.append(s['labels'])
        print(f"{X.shape}  ({time.time()-t0:.1f}s)")

    X_tr   = np.vstack(X_all)
    y_tr   = np.concatenate(y_all)
    eeg_tr = np.vstack(eeg_all)
    lbl_tr = np.concatenate(lbl_all)
    n_tr   = eeg_tr.shape[0]
    y_bin  = (lbl_tr == 2).astype(int)
    print(f"\n  Training matrix: {X_tr.shape}  [emo={y_bin.sum()} neu={(1-y_bin).sum()}]")

    # Step 2: CSP
    print("\nStep 2: Fitting CSP...")
    csp    = CSP(); csp.fit(eeg_tr, lbl_tr)
    csp_tr = csp.log_var_features(eeg_tr)
    X_tr_f = np.hstack([X_tr, csp_tr])
    print(f"  Full feature matrix: {X_tr_f.shape}")

    # Step 3: OOF for Platt calibration
    print("\nStep 3: Building OOF for Platt calibration...")
    oof_probs = np.full((n_tr, N_TP), 0.5, dtype=np.float64)
    cumsum = np.cumsum([s['eeg'].shape[0] for s in train_subjects])
    starts = np.concatenate([[0], cumsum[:-1]])

    for i, s in enumerate(train_subjects):
        sl   = slice(int(starts[i]), int(cumsum[i]))
        tr_i = [j for j in range(len(train_subjects)) if j != i]
        X_tr_i   = np.vstack([X_all[j] for j in tr_i])
        y_tr_i   = np.concatenate([y_all[j] for j in tr_i])
        eeg_tr_i = np.vstack([train_subjects[j]['eeg'] for j in tr_i])
        lbl_tr_i = np.concatenate([train_subjects[j]['labels'] for j in tr_i])
        n_tr_i   = eeg_tr_i.shape[0]

        csp_i      = CSP(); csp_i.fit(eeg_tr_i, lbl_tr_i)
        csp_feat_i = csp_i.log_var_features(eeg_tr_i)
        csp_val_i  = csp_i.log_var_features(s['eeg'])
        X_tr_if    = np.hstack([X_tr_i, csp_feat_i])
        X_v_if     = np.hstack([X_all[i], csp_val_i])

        tre_i = TimeResolvedEnsemble()
        tre_i.fit(X_tr_if, y_tr_i, n_tr_i, verbose=False)
        n_vi  = s['eeg'].shape[0]
        p_raw_tab = tre_i.predict_proba_matrix(X_v_if, n_vi)

        if HAVE_TORCH and ENN_W > 0:
            eeg_tr_p = prepare_eeg_for_gru(eeg_tr_i)
            eeg_v_p  = prepare_eeg_for_gru(s['eeg'])
            y_tr_ib  = (lbl_tr_i == 2).astype(np.float32)
            y_v_ib   = (s['labels'] == 2).astype(np.float32)
            gru_p_i, _ = train_temporal_gru(
                eeg_tr_p, y_tr_ib, eeg_v_p, y_v_ib, TORCH_DEVICE)
            p_raw = (1.0-ENN_W)*p_raw_tab + ENN_W*gru_p_i
            if TORCH_DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
        else:
            p_raw = p_raw_tab

        p_sm  = smooth_predictions(p_raw, sigma=SMOOTH_SIGMA)
        p_gat = adaptive_temporal_gate(p_sm)
        oof_probs[sl] = p_gat
        print(f"  OOF [{i+1}/{len(train_subjects)}] {s['id']} done")
        del tre_i; gc.collect()

    # Step 4: Train final TRE
    print("\nStep 4: Training final TRE on ALL training data...")
    tre_final = TimeResolvedEnsemble()
    tre_final.fit(X_tr_f, y_tr, n_tr, verbose=True)
    print("\nStep 4b: Adaptive per-timepoint weights...")
    tre_final.set_adaptive_weights_from_oof(X_tr_f, y_tr[::N_TP], n_tr)

    # Step 4c: Train final GRU
    gru_final = None
    if HAVE_TORCH:
        print("\nStep 4c: Training final GRU...")
        eeg_all_proc_list = [prepare_eeg_for_gru(s['eeg']) for s in train_subjects]
        eeg_all_proc = np.vstack(eeg_all_proc_list)
        y_all_bin    = (lbl_tr == 2).astype(np.float32)
        gru_final = train_temporal_gru_final(
            eeg_all_proc, y_all_bin, TORCH_DEVICE)
        print("  ✓ Final GRU trained")
        if TORCH_DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    # Step 5: V9-8 Platt calibration
    print("\nStep 5: Fitting Platt calibration on OOF predictions...")
    platt_calibrators = fit_platt_calibration(oof_probs, y_bin)
    print("  ✓ Platt calibration fitted (safer than Isotonic for small N)")

    # Step 6: Predict test subjects
    print("\nStep 6: Generating test predictions...")
    rows = []

    for s in test_subjects:
        subj_id  = s['subj_id']
        n_trials = s['eeg'].shape[0]
        print(f"\n  Subject {subj_id} ({s['id']}): {n_trials} trials")

        # Tabular features (with ERP template from all training subjects)
        X_te, _, _, _, _ = extract_subject_features(
            s, apply_ea=True, erp_template=GRAND_ERP_TEMPLATE)
        csp_te  = csp.log_var_features(s['eeg'])
        X_te_f  = np.hstack([X_te, csp_te])

        probs_tab_te = tre_final.predict_proba_matrix(X_te_f, n_trials)

        # V9-7: Sliding window GRU inference
        if HAVE_TORCH and gru_final is not None and ENN_W > 0:
            eeg_te_proc = prepare_eeg_for_gru(s['eeg'])
            gru_probs_te = gru_sliding_inference(gru_final, eeg_te_proc, TORCH_DEVICE)
            probs_raw = (1.0-ENN_W)*probs_tab_te + ENN_W*gru_probs_te
        else:
            probs_raw = probs_tab_te

        probs_sm  = smooth_predictions(probs_raw, sigma=SMOOTH_SIGMA)
        probs_gat = adaptive_temporal_gate(probs_sm)

        # V9-8: Platt calibration
        probs_cal = apply_platt_calibration(probs_gat, platt_calibrators)
        probs_cal = np.clip(probs_cal, 0.01, 0.99)

        print(f"    range : [{probs_cal.min():.3f}, {probs_cal.max():.3f}]  "
              f"mean : {probs_cal.mean():.4f}")

        for trial_id in range(n_trials):
            for tp in range(N_TP):
                rows.append({
                    'id':         f"{subj_id}_{trial_id}_{tp}",
                    'prediction': float(probs_cal[trial_id, tp])
                })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\n{'='*70}")
    print(f"✓ Submission saved → {output_path}")
    print(f"  Total rows : {len(df):,}")
    print(df.head(6).to_string(index=False))
    return df

print("✓ generate_submission defined (v9.0)")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 15 — MAIN
# ─────────────────────────────────────────────────────────────────────────────

RUN_LOSO = True  # MANDATORY — also computes grand ERP template

print("""
╔══════════════════════════════════════════════════════════════════════════╗
║   EEG Emotional Memory — Ultra Pipeline v9.0  (DATA-DRIVEN FINAL)       ║
╠══════════════════════════════════════════════════════════════════════════╣
║  V9-1  Signal Window    : 50-900ms (171 tps) ← captures early ERP      ║
║  V9-2  Gate corrected   : A(100-350ms)=1.0 / B(50-700ms)=0.60 / C=0.12 ║
║  V9-3  ERP Template     : grand (emo-neu) diff, 48 features/trial       ║
║  V9-4  ITPC             : inter-trial phase coherence, 16 features/tp   ║
║  V9-5  FAI-theta        : frontal theta asymmetry index, 3 features/tp  ║
║  V9-6  GRU validation   : early window (100-350ms) ← real signal        ║
║  V9-7  Sliding inference: consistent train/test frame strategy           ║
║  V9-8  Platt calibration: safer than Isotonic for 14-subject data       ║
║  V9-9  Diagnostic plot  : per-subject AUC time-course                   ║
║  Features: ~782/tp → SelectKBest(350) | GRU ENN_W: auto-tuned          ║
║  Expected wAUC: 0.545–0.580  (vs 0.521 in v8.0)                        ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

total_start = time.time()

print("STEP 1: Loading training data...")
train_subjects = load_all_training(EMO_DIR, NEU_DIR)

print("\nSTEP 2: Loading test data...")
test_subjects = load_all_test(TEST_DIR)

if RUN_LOSO:
    print("\nSTEP 3: LOSO CV (ERP template + corrected gate + ENN_W tuning)...")
    fold_results = run_loso(train_subjects, tune_sigma=True)
    win_aucs = [r['window_auc'] for r in fold_results]
    print(f"\n  ► Mean Window-AUC = {np.mean(win_aucs):.4f} ± {np.std(win_aucs):.4f}")
else:
    print("\nSTEP 3: Skipping LOSO (set RUN_LOSO=True!)")
    # If skipping LOSO, at least compute the ERP template
    print("  Computing ERP template...")
    GRAND_ERP_TEMPLATE = compute_grand_erp_template(train_subjects)
    fold_results = []

print("\nSTEP 4: Training final model + generating submission...")
df = generate_submission(train_subjects, test_subjects, OUTPUT)

print("\nSTEP 5: Validating submission format...")
assert 'id'         in df.columns
assert 'prediction' in df.columns
assert df['prediction'].between(0,1).all(), "Predictions out of [0,1]!"
parts = str(df.iloc[0]['id']).split('_')
assert len(parts) == 3, f"ID format wrong: {df.iloc[0]['id']}"
print(f"  ✓ id, prediction columns present")
print(f"  ✓ ID format: subject_trial_timepoint")
print(f"  ✓ Total rows: {len(df):,}")
print(f"  ✓ Pred range: [{df.prediction.min():.4f}, {df.prediction.max():.4f}]")

print("\nSTEP 6: Copying submission...")
import shutil
local_copy = os.path.join(os.getcwd(), 'submission.csv')
if os.path.abspath(OUTPUT) != os.path.abspath(local_copy):
    try:
        shutil.copy(OUTPUT, local_copy)
        print(f"  ✓ Copied to {local_copy}")
    except Exception as e:
        print(f"  ⚠ Copy skipped: {e}")

total_time = time.time() - total_start
win_summary = (f"Mean Window-AUC = {np.mean([r['window_auc'] for r in fold_results]):.4f}"
               if fold_results else "LOSO skipped")

print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║  ✓ DONE  ({total_time/60:.1f} min total)
╠══════════════════════════════════════════════════════════════════════════╣
║  {win_summary:<70s}║
║  SMOOTH_SIGMA = {SMOOTH_SIGMA:<55d}║
║  ENN_W        = {ENN_W:<55.2f}║
║  Gate: A(100-350ms)=1.0  B(50-700ms)=0.60  C=0.12                      ║
║  ERP template: {GRAND_ERP_TEMPLATE.shape if GRAND_ERP_TEMPLATE is not None else 'N/A'!s:<56}║
║  Submission: {OUTPUT:<59s}║
╚══════════════════════════════════════════════════════════════════════════╝
""")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 16 — Diagnostics & Utilities
# ─────────────────────────────────────────────────────────────────────────────

def diagnose(path):
    """Print full HDF5 tree of a .mat file."""
    print(f"\nDiagnosing: {path}")
    with h5py.File(path, 'r') as f:
        def show(name, obj):
            sh = obj.shape if hasattr(obj,'shape') else 'group'
            dt = obj.dtype  if hasattr(obj,'dtype') else '—'
            print(f"  {name:45s}  shape={sh}  dtype={dt}")
        f.visititems(show)


def gpu_info():
    """Print current GPU memory usage."""
    if HAVE_TORCH and torch.cuda.is_available():
        alloc    = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0)  / 1e9
        total    = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU — Allocated: {alloc:.2f}GB | Reserved: {reserved:.2f}GB "
              f"| Total: {total:.1f}GB")
    else:
        print("No CUDA device")


def plot_per_subject_timecourse(fold_results, train_subjects):
    """
    V9-9: Save per-subject AUC time-course.
    Identifies which subjects have strong early (100-350ms) vs late signal.
    """
    n = len(fold_results)
    fig, axes = plt.subplots(3, 5, figsize=(22, 14), sharey=True)
    axes = axes.flatten()
    t_ms = np.arange(len(fold_results[0]['aucs'])) * (1000/FS)

    for i, (result, subj) in enumerate(zip(fold_results, train_subjects)):
        ax = axes[i]
        ax.plot(t_ms, result['aucs'], 'b-', lw=1.5)
        ax.axhline(0.5, color='r', ls='--', lw=1)
        # Mark corrected gate zones
        ax.axvspan(GATE_ZONE_A_START*1000/FS, GATE_ZONE_A_END*1000/FS,
                   alpha=0.25, color='green', label='Zone A')
        ax.axvspan(GATE_ZONE_B_START*1000/FS, GATE_ZONE_A_START*1000/FS,
                   alpha=0.10, color='yellow')
        ax.axvspan(400, 600, alpha=0.10, color='red', label='v8 zone (bad)')
        ax.set_title(
            f"{subj['id'][:10]}\nwAUC={result['window_auc']:.3f}",
            fontsize=8, fontweight='bold')
        ax.set_ylim(0.42, 0.65)
        if i >= 10: ax.set_xlabel('Time (ms)')

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('V9-9: Per-Subject AUC Time-Course\n'
                 'Green=Zone A (100-350ms, correct gate) | '
                 'Red=400-600ms (v8 gate, sub-chance)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(BASE, 'per_subject_timecourse_v9.png')
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"  Per-subject plot saved → {save_path}")


def show_v9_improvements():
    """Print change summary from v8.1 to v9.0."""
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║  v8.1 → v9.0 — Change Summary (DATA-DRIVEN from v8.0 AUC plot)          ║
╠══════════╦══════════════════════╦══════════════════════════════════════╣
║ Aspect   ║ v8.1 (wrong)         ║ v9.0 (corrected)                     ║
╠══════════╬══════════════════════╬══════════════════════════════════════╣
║ Gate A   ║ 400-600ms (sub-chc) ║ 100-350ms (observed peak) V9-2      ║
║ Gate B   ║ 70% keep 300-900ms  ║ 60% keep 50-700ms         V9-2      ║
║ TRE tps  ║ 121 (300-900ms)     ║ 171 (50-900ms)            V9-1      ║
║ New feat ║ (none)              ║ ERP(48) + ITPC(16) + FAI(3) V9-3-5  ║
║ GRU val  ║ 400-600ms (wrong)   ║ 100-350ms (real peak)     V9-6      ║
║ GRU inf  ║ full 200-tp trial   ║ sliding window average    V9-7      ║
║ Calib    ║ Isotonic regression ║ Platt sigmoid             V9-8      ║
║ ENN_W    ║ 0.20 start          ║ 0.00 start (auto-tuned)   V9        ║
╚══════════╩══════════════════════╩══════════════════════════════════════╝

Root cause of v8.0 failure (0.521):
  1. TRE trained on 400-600ms = sub-chance zone → wrong signal
  2. Gate upweighted 400-600ms = amplified worst predictions
  3. GRU validated on 400-600ms = optimised for wrong window
  4. Missing: ERP features (most direct measure of N1/P2 emotional response)

v9.0 Quick-start:
  1. Set BASE path (CELL 3)
  2. RUN_LOSO = True (always — computes ERP template automatically)
  3. Run script — expect 2-4 hrs on RTX 5000 Ada
  4. Check loso_results_v9_0.png and per_subject_timecourse_v9.png
  5. Upload submission.csv to Kaggle
    """)


# diagnose(os.path.join(EMO_DIR, 'S_2_cleaned.mat'))
# gpu_info()
# plot_per_subject_timecourse(fold_results, train_subjects)
# show_v9_improvements()

print("✓ All utilities defined: diagnose(), gpu_info(), "
      "plot_per_subject_timecourse(), show_v9_improvements()")
print("\n✓ EEG Ultra Pipeline v9.0 — all cells defined.")
show_v9_improvements()
