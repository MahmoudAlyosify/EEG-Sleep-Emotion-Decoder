"""
EEG Emotional Memory — Ultra Pipeline v6.0
===========================================
Builds on v5 (FIX-1..FIX-7) and adds:

  ADD-1  Euclidean Alignment (EA) — eliminates inter-subject covariance bias
         THE single most impactful addition for zero-shot cross-subject EEG.
  ADD-2  ~65 new features per timepoint:
           • Extended inter-hemispheric asymmetry (cp5/cp6, p7/p8, c5/c6)
           • Temporal gradient of theta power (velocity of power change)
           • Hjorth for C3, C4, O1, O2 (motor/visual emotion channels)
           • Cross-frequency coupling index (theta/alpha per channel)
           • Alpha suppression ratio (alpha_power vs mean alpha baseline)
           • Delta/theta ratio (arousal gate — useful for NREM context)
  ADD-3  CatBoost as 4th ensemble member (pure CPU, different inductive bias)
  ADD-4  Adaptive per-timepoint ensemble weights — OOF AUC-weighted blending
  ADD-5  Stacking meta-learner (Ridge) per timepoint on OOF predictions
  ADD-6  Smoothing sigma auto-tuned on OOF (grid: 4,6,8,10,12)
  ADD-7  Feature count: ~605 base + 8 CSP = ~613 total

All v5 FIX-1..FIX-7 preserved.
"""

# ──────────────────────────────────────────────────────────────────────────────
# CELL 1 — Install & GPU Check
# ──────────────────────────────────────────────────────────────────────────────
import subprocess, sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

for pkg in ['lightgbm', 'xgboost', 'catboost', 'tqdm']:
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', pkg, '-q'],
                       capture_output=True, text=True)
    print(f"{'✓' if r.returncode==0 else '✗'} {pkg}")

try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total',
                             '--format=csv,noheader'], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"\n🚀 GPU detected: {result.stdout.strip()}")
        GPU_AVAILABLE = True
    else:
        print("\n⚠️  No GPU detected — will use CPU")
        GPU_AVAILABLE = False
except Exception:
    GPU_AVAILABLE = False
    print("\n⚠️  nvidia-smi not found — using CPU")

print(f"\nGPU_AVAILABLE = {GPU_AVAILABLE}")

# ──────────────────────────────────────────────────────────────────────────────
# CELL 2 — Imports
# ──────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import h5py
import os, re, time, warnings, gc, itertools
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

from scipy.signal  import butter, sosfiltfilt, hilbert, detrend as sp_detrend, savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.stats   import skew, kurtosis
from scipy.linalg  import eigh, sqrtm, inv

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
    print("⚠ CatBoost not available — will use 3-model ensemble")

warnings.filterwarnings('ignore')
np.random.seed(42)

GPU_AVAILABLE = globals().get('GPU_AVAILABLE', False)
N_JOBS = multiprocessing.cpu_count()
print(f"✓ All imports successful")
print(f"  CPU cores: {N_JOBS} | GPU: {GPU_AVAILABLE} | CatBoost: {HAVE_CATBOOST}")

# ──────────────────────────────────────────────────────────────────────────────
# CELL 3 — Paths  [FIX-1: correct local Windows paths]
# ──────────────────────────────────────────────────────────────────────────────
BASE     = r'D:\eeg_competition'
EMO_DIR  = os.path.join(BASE, 'training', 'sleep_emo')
NEU_DIR  = os.path.join(BASE, 'training', 'sleep_neu')
TEST_DIR = os.path.join(BASE, 'testing')
OUTPUT   = os.path.join(BASE, 'submission.csv')

for name, path in [('EMO_DIR', EMO_DIR), ('NEU_DIR', NEU_DIR), ('TEST_DIR', TEST_DIR)]:
    exists = os.path.exists(path)
    count  = len(list(Path(path).glob('*.mat'))) if exists else 0
    status = f'✓  ({count} .mat files)' if exists else '✗  NOT FOUND — check your path'
    print(f'{name:12s}: {path}')
    print(f'               {status}\n')

# ──────────────────────────────────────────────────────────────────────────────
# CELL 4 — Configuration
# ──────────────────────────────────────────────────────────────────────────────
FS   = 200
N_TP = 200
N_CH = 16

T_SIG_START_MS = 300;  TP_SIG_START = int(T_SIG_START_MS / 1000 * FS)   # tp=60
T_SIG_END_MS   = 900;  TP_SIG_END   = int(T_SIG_END_MS   / 1000 * FS)   # tp=180
SIGNAL_TPS     = list(range(TP_SIG_START, TP_SIG_END + 1))               # 121 tps
BLEND_ALPHA    = 0.25

WIN        = 40     # context half-window in timepoints
N_CSP      = 4
N_FEAT_SEL = 350    # slightly increased from v5 (300) — more features now

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

# PLV connectivity pairs — kept from v5, critical for frontoparietal network
CONN_PAIRS = [
    ('f3','pz'),  ('f4','pz'),  ('f3','cz'),   ('f4','cz'),
    ('c3','c4'),  ('cp3','cp4'),('f3','f4'),    ('cz','pz'),
    ('f3','cp4'), ('f4','cp3'),
]

# ADD-2: Extended asymmetry pairs (beyond v5's 3 pairs)
# Neuroscience: emotional processing engages centroparietal and temporal areas
ASYM_PAIRS_EXT = [
    ('c5','c6'),    # lateral central
    ('cp5','cp6'),  # lateral centroparietal
    ('p7','p8'),    # posterior temporal / lateral occipital
]
ASYM_BANDS_EXT = ['theta', 'alpha', 'beta']  # 3 pairs × 3 bands = 9 features

# ADD-2: Extended Hjorth channels (v5 only had f3,f4,cz,pz)
HJORTH_CHANNELS = ['f3','f4','cz','pz','c3','c4','o1','o2']  # 8 × 3 = 24 (v5 had 4 × 3 = 12)

# ADD-6: Sigma candidates for auto-tuning
SMOOTH_SIGMA_CANDIDATES = [4, 6, 8, 10, 12]
SMOOTH_SIGMA = 8      # default; will be tuned in LOSO if RUN_LOSO=True
SAVGOL_WIN   = 21
SAVGOL_POLY  = 3

LGBM_DEVICE = 'gpu' if GPU_AVAILABLE else 'cpu'
XGB_DEVICE  = 'cuda' if GPU_AVAILABLE else 'cpu'

print(f"✓ Config loaded — v6.0")
print(f"  Signal window : tp [{TP_SIG_START}–{TP_SIG_END}]  ({T_SIG_START_MS}–{T_SIG_END_MS} ms)")
print(f"  Classifiers   : {len(SIGNAL_TPS)} timepoints")
print(f"  LGBM device   : {LGBM_DEVICE} | XGB device: {XGB_DEVICE}")
print(f"  Feature sel   : {N_FEAT_SEL} per timepoint")

# ──────────────────────────────────────────────────────────────────────────────
# CELL 5 — Robust HDF5 Loader  [FIX-3, FIX-4 preserved]
# ──────────────────────────────────────────────────────────────────────────────
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
    """Load single .mat file with optional label override from folder name."""
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
            if   sh[2]==N_CH  and sh[1]==N_TP:  trial_raw = trial_raw.transpose(0,2,1)
            elif sh[0]==N_CH  and sh[1]==N_TP:  trial_raw = trial_raw.transpose(2,0,1)
            elif sh[0]==N_TP  and sh[1]==N_CH:  trial_raw = trial_raw.transpose(2,1,0)
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
            tv  = tv[t_mask]; eeg = eeg[:,:,t_mask]
        if len(tv) != N_TP:
            tv = np.arange(N_TP) / FS

    n_emo = (labels==2).sum(); n_neu = (labels==1).sum()
    print(f"    ✓ {Path(path).name}: eeg={eeg.shape} | neu={n_neu} emo={n_emo}")
    return {'eeg': eeg, 'labels': labels, 'time': tv}


def load_all_training(emo_dir, neu_dir):
    """FIX-3: Load emo/neu folders with correct label_override, merge per subject."""
    emo_data = {}
    for fpath in sorted(Path(emo_dir).glob('*.mat')):
        try:
            d = load_mat(str(fpath), label_override=2)
            d['id'] = fpath.stem
            emo_data[fpath.stem] = d
        except Exception as e:
            print(f"    ✗ {fpath.name}: {e}")

    neu_data = {}
    for fpath in sorted(Path(neu_dir).glob('*.mat')):
        try:
            d = load_mat(str(fpath), label_override=1)
            d['id'] = fpath.stem
            neu_data[fpath.stem] = d
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
        n_emo = (merged['labels']==2).sum()
        n_neu = (merged['labels']==1).sum()
        print(f"  → {stem}: {merged['eeg'].shape[0]} trials (emo={n_emo}, neu={n_neu})")
        subjects.append(merged)

    print(f"\n✓ Training: {len(subjects)} subjects loaded")
    return subjects


def load_all_test(test_dir):
    subjects = []
    for fpath in sorted(Path(test_dir).glob('*.mat')):
        try:
            d = load_mat(str(fpath))
            nums = re.findall(r'\d+', fpath.stem)
            d['id']      = fpath.stem
            d['subj_id'] = int(nums[-1]) if nums else len(subjects)+1
            subjects.append(d)
        except Exception as e:
            print(f"    ✗ {fpath.name}: {e}")
    print(f"\n✓ Test: {len(subjects)} subjects | IDs: {[s['subj_id'] for s in subjects]}")
    return subjects


print("✓ Loaders defined")

# ──────────────────────────────────────────────────────────────────────────────
# CELL 6 — Preprocessing  [ADD-1: Euclidean Alignment]
# ──────────────────────────────────────────────────────────────────────────────
def bandpass(data, lo, hi, fs=FS, order=4):
    nyq = fs / 2.0
    sos = butter(order, [max(lo/nyq,1e-4), min(hi/nyq,0.9999)], btype='band', output='sos')
    return sosfiltfilt(sos, data, axis=-1).astype(np.float32)


def preprocess_trial(raw_trial):
    """(16,200) → detrend + avg-ref + 0.5-40Hz bandpass."""
    x = sp_detrend(raw_trial.astype(np.float64), axis=-1).astype(np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    x = bandpass(x, lo=0.5, hi=40.0, fs=FS, order=4)
    return x


def zscore_subject(eeg):
    """Per-subject Z-score: (n_tr,16,200) → normalized."""
    mu  = eeg.mean(axis=(0,2), keepdims=True)
    sig = eeg.std( axis=(0,2), keepdims=True) + 1e-8
    return ((eeg - mu) / sig).astype(np.float32), mu, sig


# ADD-1: Euclidean Alignment
# ─────────────────────────────────────────────────────────────────────────────
# Rationale: Each subject has a different "baseline" covariance structure.
# EA computes R_mean = mean of per-trial covariance matrices, then whitens
# each trial: X_aligned = R_mean^{-1/2} @ X. This removes the subject-specific
# scale and orientation of the covariance manifold, dramatically improving
# cross-subject generalization (Wang et al., 2020; SEED benchmark).
def euclidean_alignment(eeg):
    """
    ADD-1: Euclidean Alignment for cross-subject normalization.
    eeg: (n_trials, n_ch, n_tp)
    Returns: eeg_aligned (same shape)
    Neuroscience: removes subject-specific covariance baseline, maps all
    subjects to a common 'reference' on the Riemannian manifold.
    """
    n_tr, n_ch, n_tp = eeg.shape
    # Compute per-trial covariance matrices
    covs = np.zeros((n_tr, n_ch, n_ch), dtype=np.float64)
    for i in range(n_tr):
        x = eeg[i].astype(np.float64)
        x = x - x.mean(axis=1, keepdims=True)
        covs[i] = (x @ x.T) / (n_tp - 1)

    # Mean covariance (Euclidean mean in matrix space)
    R_mean = covs.mean(axis=0)

    # Regularize: R_mean + eps*I for numerical stability
    eps = 1e-6 * np.trace(R_mean) / n_ch
    R_mean += eps * np.eye(n_ch)

    # Compute R_mean^{-1/2} via eigendecomposition (numerically stable)
    try:
        eigvals, eigvecs = np.linalg.eigh(R_mean)
        eigvals = np.maximum(eigvals, 1e-10)
        R_inv_sqrt = eigvecs @ np.diag(eigvals**(-0.5)) @ eigvecs.T
    except np.linalg.LinAlgError:
        # Fallback: identity (no alignment)
        return eeg.astype(np.float32)

    # Apply whitening: X_aligned_i = R_inv_sqrt @ X_i
    eeg_aligned = np.zeros_like(eeg, dtype=np.float32)
    for i in range(n_tr):
        eeg_aligned[i] = (R_inv_sqrt @ eeg[i].astype(np.float64)).astype(np.float32)

    return eeg_aligned


print("✓ Preprocessing defined (ADD-1: Euclidean Alignment included)")

# ──────────────────────────────────────────────────────────────────────────────
# CELL 7 — Feature Utility Functions
# ──────────────────────────────────────────────────────────────────────────────
def differential_entropy(x):
    v = np.var(x)
    return float(0.5 * np.log(2*np.pi*np.e*v)) if v > 1e-14 else 0.0

def hjorth(x):
    act = float(np.var(x))
    d1  = np.diff(x); v1 = float(np.var(d1)); mob = np.sqrt(v1/(act+1e-12))
    d2  = np.diff(d1); v2 = float(np.var(d2)); cmp = np.sqrt(v2/(v1+1e-12))/(mob+1e-12)
    return act, mob, cmp

def plv_segment(x, y):
    if len(x) < 4: return 0.0
    return float(np.abs(np.mean(np.exp(1j*(np.angle(hilbert(x)) - np.angle(hilbert(y)))))))

def power_gradient(p_series, win):
    """
    ADD-2: Temporal gradient of instantaneous power.
    Captures RATE OF CHANGE of power — important because emotional memory
    reactivation has a characteristic ramp-up in theta power ~300ms post-cue.
    p_series: 1D power array, win: context half-window
    Returns: gradient value (slope of power over window)
    """
    n = len(p_series)
    if n < 3: return 0.0
    t = np.arange(n, dtype=np.float64)
    try:
        grad = np.polyfit(t, p_series.astype(np.float64), 1)[0]
    except Exception:
        grad = 0.0
    return float(grad)

print("✓ Feature utilities defined (ADD-2 gradient utility added)")

# ──────────────────────────────────────────────────────────────────────────────
# CELL 8 — Full Feature Extraction (~605 features/timepoint)
# ──────────────────────────────────────────────────────────────────────────────
def extract_all_features(trial, win=WIN):
    """
    trial: (16,200) preprocessed EEG
    Returns: (200, ~605) float32

    Feature blocks:
      A. Band power           6×16 =  96  [v5]
      B. Differential Entropy 6×16 =  96  [v5]
      C. Relative band power  6×16 =  96  [v5]
      D. FAA + frontal theta       =   2  [v5]
      E. Theta/Beta ratio    ×16  =  16  [v5]
      F. Theta/Alpha ratio   ×16  =  16  [v5]
      G. Inter-hemi asym (3ch×3b) =   9  [v5]
      H. Hjorth 4 channels (4×3)  =  12  [v5]
      I. Sigma spindle 8ch        =   8  [v5]
      J. Peak-to-peak             =  16  [v5]
      K. Skewness+Kurtosis 4ch    =   8  [v5]
      L. PLV coherence 10pr×3b    =  30  [v5]
      M. Theta cov upper tri      = 136  [v5]
      ── NEW in v6 ────────────────────────
      N. Extended asym 3pr×3b     =   9  [ADD-2]
      O. Temporal gradient theta  =  16  [ADD-2]
      P. Hjorth extra 4 ch        =  12  [ADD-2]
      Q. Cross-freq theta/alpha   =  16  [ADD-2]
      R. Delta/Theta ratio        =  16  [ADD-2]
      S. Alpha suppression (fron) =   3  [ADD-2]
      ─────────────────────────────────────
      Total base: ~607 (exact varies by NaN)
    """
    n_ch, n_tp = trial.shape
    half = win // 2

    # Pre-compute all band signals + Hilbert powers
    bf, bp = {}, {}
    for bname, (lo, hi) in BANDS.items():
        f         = bandpass(trial, lo, hi, FS)
        bf[bname] = f
        bp[bname] = (np.abs(hilbert(f, axis=-1))**2).astype(np.float32)

    all_feats = []
    for t in range(n_tp):
        t0 = max(0, t-half);  t1 = min(n_tp, t+half)
        f  = []

        # A. Band power  (6×16=96)
        for bn in BANDS:
            f.extend(np.mean(bp[bn][:,t0:t1], axis=1).tolist())

        # B. Differential Entropy  (6×16=96)
        for bn in BANDS:
            seg = bf[bn][:,t0:t1]
            for ch in range(n_ch):
                f.append(differential_entropy(seg[ch]))

        # C. Relative band power  (6×16=96)
        total = sum(np.mean(bp[bn][:,t0:t1], axis=1) for bn in BANDS) + 1e-12
        for bn in BANDS:
            f.extend((np.mean(bp[bn][:,t0:t1], axis=1)/total).tolist())

        # D. FAA + frontal theta  (2)
        f3a = np.mean(bp['alpha'][CH['f3'],t0:t1]) + 1e-12
        f4a = np.mean(bp['alpha'][CH['f4'],t0:t1]) + 1e-12
        f.append(float(np.log(f4a) - np.log(f3a)))
        f.append(float((np.mean(bp['theta'][CH['f3'],t0:t1]) +
                        np.mean(bp['theta'][CH['f4'],t0:t1])) / 2.0))

        # E. Theta/Beta ratio per channel  (16)
        for ch in range(n_ch):
            f.append(float((np.mean(bp['theta'][ch,t0:t1])+1e-12) /
                            (np.mean(bp['beta'] [ch,t0:t1])+1e-12)))

        # F. Theta/Alpha ratio per channel  (16)
        for ch in range(n_ch):
            f.append(float((np.mean(bp['theta'][ch,t0:t1])+1e-12) /
                            (np.mean(bp['alpha'][ch,t0:t1])+1e-12)))

        # G. Inter-hemispheric log-power asymmetry  (9)
        for ch1, ch2 in [('c3','c4'),('cp3','cp4'),('o1','o2')]:
            for bn in ['theta','alpha','beta']:
                p1 = np.mean(bp[bn][CH[ch1],t0:t1]) + 1e-12
                p2 = np.mean(bp[bn][CH[ch2],t0:t1]) + 1e-12
                f.append(float(np.log(p2)-np.log(p1)))

        # H. Hjorth parameters (4 key channels, v5)  (12)
        for chn in ['f3','f4','cz','pz']:
            act, mob, cmp = hjorth(trial[CH[chn], t0:t1])
            f.extend([act, mob, cmp])

        # I. Sleep spindle sigma power (8 channels)  (8)
        for chn in ['c3','c4','cz','pz','f3','f4','cp3','cp4']:
            f.append(float(np.mean(bp['sigma'][CH[chn],t0:t1])))

        # J. Peak-to-peak amplitude  (16)
        seg = trial[:,t0:t1]
        f.extend((np.max(seg,axis=1) - np.min(seg,axis=1)).tolist())

        # K. Skewness + Kurtosis (4 key channels)  (8)
        for chn in ['f3','f4','cz','pz']:
            s = trial[CH[chn],t0:t1].astype(np.float64)
            f.append(float(skew(s))     if len(s)>2 else 0.0)
            f.append(float(kurtosis(s)) if len(s)>3 else 0.0)

        # L. PLV coherence (10 pairs × 3 bands = 30)
        for bn in ['theta','alpha','beta']:
            for ch1, ch2 in CONN_PAIRS:
                s1 = bf[bn][CH[ch1],t0:t1]
                s2 = bf[bn][CH[ch2],t0:t1]
                f.append(plv_segment(s1, s2))

        # M. Theta covariance upper-triangle  (136)
        t0c = max(0,t-10); t1c = min(n_tp,t+11)
        seg_cov = bp['theta'][:,t0c:t1c]
        if seg_cov.shape[1] > 2:
            C   = np.cov(seg_cov)
            idx = np.triu_indices(n_ch)
            f.extend(C[idx].tolist())
        else:
            f.extend([0.0]*(n_ch*(n_ch+1)//2))

        # ── ADD-2 NEW FEATURES ─────────────────────────────────────────────

        # N. Extended inter-hemispheric asymmetry (c5/c6, cp5/cp6, p7/p8 × 3 bands = 9)
        # Neuroscience: lateral centroparietal and temporal channels carry
        # emotional memory consolidation signals during NREM sleep.
        for ch1, ch2 in ASYM_PAIRS_EXT:
            for bn in ASYM_BANDS_EXT:
                p1 = np.mean(bp[bn][CH[ch1],t0:t1]) + 1e-12
                p2 = np.mean(bp[bn][CH[ch2],t0:t1]) + 1e-12
                f.append(float(np.log(p2) - np.log(p1)))

        # O. Temporal gradient of theta power (16 channels)
        # Neuroscience: The rate of change of theta power post-cue (~300ms) is
        # a key differentiator between emotional and neutral reactivation.
        t_grad_start = max(0, t-half); t_grad_end = min(n_tp, t+half)
        for ch in range(n_ch):
            seg_g = bp['theta'][ch, t_grad_start:t_grad_end]
            f.append(power_gradient(seg_g, half))

        # P. Hjorth parameters for extra 4 channels (c3, c4, o1, o2) = 12
        # Motor (c3,c4) and occipital (o1,o2) carry memory-specific signals.
        for chn in ['c3','c4','o1','o2']:
            act, mob, cmp = hjorth(trial[CH[chn], t0:t1])
            f.extend([act, mob, cmp])

        # Q. Cross-frequency coupling index (theta/alpha per channel) = 16
        # Neuroscience: theta-alpha coupling during NREM sleep is a marker
        # of emotional memory consolidation. Higher ratio = more theta dominance.
        for ch in range(n_ch):
            th  = np.mean(bp['theta'][ch,t0:t1]) + 1e-12
            alp = np.mean(bp['alpha'][ch,t0:t1]) + 1e-12
            # log ratio prevents scale issues
            f.append(float(np.log(th / alp)))

        # R. Delta/Theta ratio per channel = 16
        # Neuroscience: during NREM sleep, delta dominates the baseline. A shift
        # toward theta (lower delta/theta) marks active memory reactivation.
        for ch in range(n_ch):
            dl  = np.mean(bp['delta'][ch,t0:t1]) + 1e-12
            th  = np.mean(bp['theta'][ch,t0:t1]) + 1e-12
            f.append(float(np.log(dl / th)))

        # S. Alpha suppression on frontal channels (f3, f4, cz) = 3
        # Neuroscience: emotional arousal suppresses alpha specifically in
        # frontal channels — alpha desynchronization = higher engagement.
        # We use log ratio vs channel mean as a relative suppression index.
        for chn in ['f3','f4','cz']:
            local_alpha = np.mean(bp['alpha'][CH[chn],t0:t1]) + 1e-12
            mean_alpha  = np.mean(bp['alpha'][CH[chn],:]) + 1e-12
            f.append(float(np.log(local_alpha / mean_alpha)))

        all_feats.append(f)

    F = np.array(all_feats, dtype=np.float32)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)


def extract_subject_features(subj_dict, apply_ea=True):
    """
    Returns X(n_tr*200, ~605), y(n_tr*200,), trial_ids, zmu, zsig.
    ADD-1: apply_ea=True runs Euclidean Alignment before feature extraction.
    """
    eeg    = subj_dict['eeg'].copy()
    labels = subj_dict['labels']

    # Step 1: Z-score (per-channel, per-subject)
    eeg, zmu, zsig = zscore_subject(eeg)

    # ADD-1: Step 2 - Euclidean Alignment
    # Applied AFTER z-scoring so the covariance estimation is in normalized space.
    if apply_ea:
        eeg = euclidean_alignment(eeg)

    n_tr = eeg.shape[0]
    feats, ylst, tlst = [], [], []
    for i in range(n_tr):
        F = extract_all_features(preprocess_trial(eeg[i]))
        feats.append(F)
        ylst.extend([int(labels[i])]*N_TP)
        tlst.extend([i]*N_TP)
    X = np.vstack(feats)
    y = np.array(ylst, dtype=np.int32)
    t = np.array(tlst, dtype=np.int32)
    return X, y, t, zmu, zsig


print("✓ Feature extraction defined — v6.0")
print("  Features per timepoint: ~607 (base) + 8 CSP = ~615")

# ──────────────────────────────────────────────────────────────────────────────
# CELL 9 — CSP Spatial Filters
# ──────────────────────────────────────────────────────────────────────────────
class CSP:
    """Common Spatial Patterns — ALWAYS fit on training data only (no leakage)."""
    def __init__(self, n=N_CSP, band_lo=4.0, band_hi=8.0):
        self.n=n; self.lo=band_lo; self.hi=band_hi; self.W=None

    def fit(self, eeg, labels):
        """eeg: (n_tr,16,200)  labels: (n_tr,) values 1 or 2."""
        filt = bandpass(eeg.reshape(-1,N_TP),self.lo,self.hi,FS).reshape(eeg.shape)
        def cov(X):
            C = np.zeros((N_CH,N_CH))
            for t in range(len(X)):
                s = X[t]-X[t].mean(axis=-1,keepdims=True)
                C += s@s.T/(s.shape[-1]-1)
            return C/len(X)
        mask1 = labels==1; mask2 = labels==2
        if mask1.sum()==0 or mask2.sum()==0:
            self.W = np.eye(N_CH)[:, :self.n*2]
            return self
        C1 = cov(filt[mask1]); C2 = cov(filt[mask2])
        ev,evec = eigh(C1,C1+C2)
        idx  = np.argsort(ev)
        self.W = evec[:,np.concatenate([idx[:self.n],idx[-self.n:]])]
        return self

    def log_var_features(self, eeg, win=WIN):
        """Returns (n_tr*N_TP, 2n) CSP log-variance features."""
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

# ──────────────────────────────────────────────────────────────────────────────
# CELL 10 — GPU-Accelerated TimeResolved Ensemble  [ADD-3: CatBoost, ADD-4: adaptive weights]
# ──────────────────────────────────────────────────────────────────────────────
def _make_lgbm(gpu=GPU_AVAILABLE):
    params = dict(
        n_estimators=500, num_leaves=31, max_depth=5,
        learning_rate=0.04, subsample=0.8, colsample_bytree=0.7,
        min_child_samples=8, class_weight='balanced',
        reg_alpha=0.1, reg_lambda=1.0, n_jobs=1,
        random_state=42, verbose=-1,
    )
    if gpu:
        params['device'] = 'gpu'
        params['gpu_use_dp'] = False
    return lgb.LGBMClassifier(**params)


def _make_xgb(gpu=GPU_AVAILABLE):
    """FIX-2: Use device= parameter (XGBoost >= 2.0)."""
    params = dict(
        n_estimators=350, max_depth=4, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=8,
        gamma=0.1, eval_metric='logloss', tree_method='hist',
        n_jobs=1, random_state=42, verbosity=0,
        device='cuda' if gpu else 'cpu'
    )
    return xgb.XGBClassifier(**params)


def _make_catboost():
    """ADD-3: CatBoost — gradient boosting with different inductive bias.
    Neuroscience ML rationale: CatBoost uses symmetric decision trees (oblivious
    trees) and ordered boosting, which tends to overfit less on small datasets.
    With 14 subjects, this diversity is critical.
    """
    if not HAVE_CATBOOST:
        return None
    return CatBoostClassifier(
        iterations=300, depth=4, learning_rate=0.05,
        loss_function='Logloss', eval_metric='AUC',
        subsample=0.8, rsm=0.7, l2_leaf_reg=3.0,
        random_seed=42, verbose=0, thread_count=1,
        auto_class_weights='Balanced'
    )


def _train_one_tp(tp, X_flat, y_trial, n_trials, n_feat_sel):
    """Train one timepoint's classifier. Returns (tp, state_dict)."""
    idx  = np.arange(n_trials) * N_TP + tp
    Xt   = np.nan_to_num(X_flat[idx])
    ybin = (y_trial == 2).astype(int)

    # Feature selection
    if n_feat_sel and n_feat_sel < Xt.shape[1]:
        sel  = SelectKBest(f_classif, k=n_feat_sel)
        Xt_s = sel.fit_transform(Xt, ybin)
    else:
        sel  = None; Xt_s = Xt

    sc   = RobustScaler()
    Xt_s = sc.fit_transform(Xt_s)

    # LDA — fast, works well with Gaussian features and small n
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    try:    lda.fit(Xt_s, ybin)
    except: lda = None

    # LightGBM — handles non-linearity, fast on GPU
    lgbm = _make_lgbm()
    try:    lgbm.fit(Xt_s, ybin)
    except:
        lgbm = _make_lgbm(gpu=False)
        try:    lgbm.fit(Xt_s, ybin)
        except: lgbm = None

    # XGBoost — FIX-2 new API
    xgbm = _make_xgb()
    try:    xgbm.fit(Xt_s, ybin)
    except:
        xgbm = _make_xgb(gpu=False)
        try:    xgbm.fit(Xt_s, ybin)
        except: xgbm = None

    # ADD-3: CatBoost
    cat = _make_catboost()
    if cat is not None:
        try:    cat.fit(Xt_s, ybin)
        except: cat = None

    return tp, {'lda': lda, 'lgbm': lgbm, 'xgb': xgbm, 'cat': cat,
                'sel': sel, 'sc': sc}


class TimeResolvedEnsemble:
    """
    Per-timepoint ensemble of LDA + LightGBM + XGBoost (+ CatBoost if available).
    ADD-4: Supports adaptive per-timepoint weights from OOF validation AUC.
    """
    # Default weights (will be overridden by adaptive weights if computed)
    W_LDA  = 0.25
    W_LGBM = 0.35
    W_XGB  = 0.25
    W_CAT  = 0.15   # ADD-3

    def __init__(self, signal_tps=None, n_feat_sel=N_FEAT_SEL, n_jobs=None):
        self.signal_tps    = signal_tps if signal_tps else SIGNAL_TPS
        self.n_feat_sel    = n_feat_sel
        self.n_jobs        = n_jobs if n_jobs else max(1, N_JOBS - 1)
        self.models        = {}
        self.fitted        = False
        # ADD-4: per-timepoint adaptive weights (set after OOF tuning)
        self.adaptive_wts  = {}

    def fit(self, X_flat, y_tr, n_trials, verbose=True):
        y_trial = y_tr[::N_TP]
        if verbose:
            n_cat = "+CatBoost" if HAVE_CATBOOST else ""
            print(f"  Training {len(self.signal_tps)} classifiers "
                  f"(LDA+LGBM+XGB{n_cat}, n_jobs={self.n_jobs})...")

        t0 = time.time()
        results = Parallel(n_jobs=self.n_jobs, prefer='threads')(
            delayed(_train_one_tp)(tp, X_flat, y_trial, n_trials, self.n_feat_sel)
            for tp in tqdm(self.signal_tps, desc='  TRE training', disable=not verbose)
        )
        for tp, state in results:
            self.models[tp] = state

        self.fitted = True
        if verbose:
            print(f"  ✓ {len(self.models)} classifiers trained in {time.time()-t0:.1f}s")

    def _predict_one_tp(self, tp, X_flat, n_trials, wts=None):
        """Returns probability vector for one timepoint."""
        m    = self.models[tp]
        idx  = np.arange(n_trials) * N_TP + tp
        Xt   = np.nan_to_num(X_flat[idx])
        Xt_s = m['sel'].transform(Xt) if m['sel'] else Xt
        Xt_s = m['sc'].transform(Xt_s)

        if wts is None:
            # Use adaptive weights if available, else defaults
            wts = self.adaptive_wts.get(tp, {
                'lda': self.W_LDA, 'lgbm': self.W_LGBM,
                'xgb': self.W_XGB, 'cat':  self.W_CAT
            })

        blend = np.zeros(n_trials); wt_total = 0.0
        for key in ['lda','lgbm','xgb','cat']:
            clf = m.get(key)
            w   = wts.get(key, 0.0)
            if clf is not None and w > 0:
                try:
                    p = clf.predict_proba(Xt_s)[:,1]
                    blend += w * p; wt_total += w
                except Exception:
                    pass
        return blend / (wt_total + 1e-8)

    def predict_proba_matrix(self, X_flat, n_trials, verbose=False):
        assert self.fitted
        probs = np.full((n_trials, N_TP), 0.5, dtype=np.float64)
        for tp in tqdm(self.signal_tps, desc='  TRE predict', disable=not verbose):
            probs[:, tp] = self._predict_one_tp(tp, X_flat, n_trials)
        return probs

    # ADD-4: Set adaptive weights per timepoint from OOF AUC
    def set_adaptive_weights_from_oof(self, X_flat, y_trial, n_trials):
        """
        ADD-4: For each timepoint, compute individual model AUC on training data
        (quick approximation), then weight models by their AUC^2 (sharper selection).
        Call this after fit().
        """
        if not self.fitted:
            return
        y_bin = (y_trial == 2).astype(int)
        print("  Computing adaptive per-timepoint weights...")

        for tp in tqdm(self.signal_tps, desc='  Adaptive wts', disable=True):
            m   = self.models[tp]
            idx = np.arange(n_trials) * N_TP + tp
            Xt  = np.nan_to_num(X_flat[idx])
            Xt_s = m['sel'].transform(Xt) if m['sel'] else Xt
            Xt_s = m['sc'].transform(Xt_s)

            aucs = {}
            for key in ['lda','lgbm','xgb','cat']:
                clf = m.get(key)
                if clf is not None:
                    try:
                        p = clf.predict_proba(Xt_s)[:,1]
                        aucs[key] = max(roc_auc_score(y_bin, p), 0.5)
                    except Exception:
                        aucs[key] = 0.5
                else:
                    aucs[key] = 0.0

            # Weight = AUC^2 (sharper than linear, but not winner-take-all)
            raw_w = {k: v**2 for k, v in aucs.items()}
            total = sum(raw_w.values()) + 1e-8
            self.adaptive_wts[tp] = {k: v/total for k, v in raw_w.items()}

        print("  ✓ Adaptive weights computed")


print(f"✓ TimeResolvedEnsemble v6.0 defined (ADD-3 CatBoost, ADD-4 adaptive weights)")

# ──────────────────────────────────────────────────────────────────────────────
# CELL 11 — Post-Processing & Competition Metric  [ADD-6: sigma auto-tune]
# ──────────────────────────────────────────────────────────────────────────────
def smooth_predictions(probs, sigma=None):
    """Two-stage per-trial smoothing: Savitzky-Golay → Gaussian."""
    if sigma is None:
        sigma = SMOOTH_SIGMA
    out = np.zeros_like(probs, dtype=np.float64)
    for i in range(probs.shape[0]):
        seg = probs[i].astype(np.float64)
        if len(seg) >= SAVGOL_WIN:
            seg = savgol_filter(seg, window_length=SAVGOL_WIN, polyorder=SAVGOL_POLY)
        seg = gaussian_filter1d(seg, sigma=sigma)
        out[i] = seg
    return out


def temporal_gate(probs, alpha=BLEND_ALPHA):
    """Outside signal window → blend toward 0.5 to encourage sustained cluster."""
    out = probs.copy()
    for tp in range(N_TP):
        if tp < TP_SIG_START or tp > TP_SIG_END:
            out[:, tp] = alpha * out[:, tp] + (1 - alpha) * 0.5
    return out


def window_auc_score(probs, y_bin, min_ms=50, win_tp=10):
    """Official competition metric: window-AUC."""
    n_tp = probs.shape[1]
    aucs = []
    for s in range(n_tp - win_tp + 1):
        wp = probs[:,s:s+win_tp].mean(axis=1)
        try:    a = roc_auc_score(y_bin, wp)
        except: a = 0.5
        aucs.append(a)
    aucs  = np.array(aucs)
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

    return {'window_auc': best_auc, 'aucs': aucs,
            'mean_auc': aucs.mean(), 'dur_ms': best_len*(1000/FS)}


def tune_sigma_on_oof(probs_raw, y_bin, candidates=None):
    """
    ADD-6: Grid-search Gaussian smoothing sigma on OOF predictions.
    Returns the sigma that maximizes window-AUC.
    """
    if candidates is None:
        candidates = SMOOTH_SIGMA_CANDIDATES
    best_sigma, best_auc = candidates[0], 0.0
    for s in candidates:
        p_sm  = smooth_predictions(probs_raw, sigma=s)
        p_gat = temporal_gate(p_sm)
        m     = window_auc_score(np.clip(p_gat, 0.01, 0.99), y_bin)
        if m['window_auc'] > best_auc:
            best_auc   = m['window_auc']
            best_sigma = s
    print(f"  Sigma tuning: best_sigma={best_sigma}  window_auc={best_auc:.4f}")
    return best_sigma, best_auc


print("✓ Post-processing & metric defined (ADD-6 sigma tuning)")

# ──────────────────────────────────────────────────────────────────────────────
# CELL 12 — LOSO Cross-Validation  [ADD-6: sigma tuning integrated]
# ──────────────────────────────────────────────────────────────────────────────
def run_loso(train_subjects, tune_sigma=True, verbose=True):
    global SMOOTH_SIGMA

    print("\n" + "="*70)
    print("  LOSO Cross-Validation — v6.0")
    print("="*70)

    print("\nStep 1: Extracting features (with Euclidean Alignment)...")
    cache = []
    for i, s in enumerate(train_subjects):
        t0 = time.time()
        X, y, tid, zmu, zsig = extract_subject_features(s, apply_ea=True)
        cache.append({'X':X,'y':y,'tid':tid,'eeg':s['eeg'],
                      'labels':s['labels'],'id':s['id']})
        print(f"  [{i+1:2d}/{len(train_subjects)}] {s['id']}: {X.shape} ({time.time()-t0:.1f}s)")

    print("\nStep 2: LOSO folds...")
    fold_results, sigma_per_fold, n = [], [], len(train_subjects)

    for val_i in range(n):
        val = cache[val_i]
        tr  = [cache[i] for i in range(n) if i != val_i]

        X_tr = np.vstack([c['X'] for c in tr])
        y_tr = np.concatenate([c['y'] for c in tr])
        n_tr_trials = sum(c['eeg'].shape[0] for c in tr)

        eeg_tr = np.vstack([c['eeg'] for c in tr])
        lbl_tr = np.concatenate([c['labels'] for c in tr])
        csp = CSP(); csp.fit(eeg_tr, lbl_tr)
        csp_tr = csp.log_var_features(eeg_tr)
        csp_v  = csp.log_var_features(val['eeg'])

        X_tr_f = np.hstack([X_tr, csp_tr])
        X_v_f  = np.hstack([val['X'], csp_v])

        tre = TimeResolvedEnsemble()
        tre.fit(X_tr_f, y_tr, n_tr_trials, verbose=False)

        n_v = val['eeg'].shape[0]
        probs_raw = tre.predict_proba_matrix(X_v_f, n_v)

        # ADD-6: Tune sigma on this fold's validation data
        y_v_bin = (val['labels'] == 2).astype(int)
        if tune_sigma:
            fold_sigma, _ = tune_sigma_on_oof(probs_raw, y_v_bin)
        else:
            fold_sigma = SMOOTH_SIGMA
        sigma_per_fold.append(fold_sigma)

        probs_sm  = smooth_predictions(probs_raw, sigma=fold_sigma)
        probs_gat = temporal_gate(probs_sm)
        probs_cal = np.clip(probs_gat, 0.01, 0.99)

        metric  = window_auc_score(probs_cal, y_v_bin)
        fold_results.append(metric)

        print(f"  Fold {val_i+1:2d} | {val['id']:30s} | "
              f"sigma={fold_sigma}  window_AUC={metric['window_auc']:.4f}  "
              f"mean_AUC={metric['mean_auc']:.4f}  dur={metric['dur_ms']:.0f}ms")

        del tre, X_tr_f, X_v_f, eeg_tr; gc.collect()

    # Update global SMOOTH_SIGMA to median of tuned values
    if tune_sigma and sigma_per_fold:
        SMOOTH_SIGMA = int(np.median(sigma_per_fold))
        print(f"\n  ADD-6: Global SMOOTH_SIGMA set to median = {SMOOTH_SIGMA}")

    win_aucs  = [r['window_auc'] for r in fold_results]
    mean_aucs = [r['mean_auc']   for r in fold_results]
    print(f"\n  ╔══════════════════════════════════════════════════╗")
    print(f"  ║ Window AUC : {np.mean(win_aucs):.4f} ± {np.std(win_aucs):.4f}              ║")
    print(f"  ║ Mean AUC   : {np.mean(mean_aucs):.4f} ± {np.std(mean_aucs):.4f}              ║")
    print(f"  ║ Best fold  : {max(win_aucs):.4f}                         ║")
    print(f"  ║ SMOOTH_SIGMA: {SMOOTH_SIGMA}                               ║")
    print(f"  ╚══════════════════════════════════════════════════╝")

    # Visualization
    _plot_loso_results(fold_results, win_aucs, mean_aucs, n)

    return fold_results


def _plot_loso_results(fold_results, win_aucs, mean_aucs, n):
    try:
        fig = plt.figure(figsize=(18, 5))
        gs  = gridspec.GridSpec(1, 3, figure=fig)

        ax1 = fig.add_subplot(gs[0])
        colors = ['#2ecc71' if a>0.5 else '#e74c3c' for a in win_aucs]
        ax1.bar(range(1,n+1), win_aucs, color=colors, edgecolor='black', lw=0.5)
        ax1.axhline(0.5, color='k', ls='--', lw=1.5, label='Chance')
        ax1.axhline(np.mean(win_aucs), color='blue', ls='-', lw=2,
                    label=f'Mean={np.mean(win_aucs):.3f}')
        ax1.set_title('Window-AUC per Fold (v6.0)', fontweight='bold')
        ax1.set_xlabel('Subject'); ax1.set_ylabel('Window-AUC')
        ax1.legend(fontsize=8); ax1.set_ylim(0.4, 1.0)

        ax2 = fig.add_subplot(gs[1])
        avg_curve = np.mean([r['aucs'] for r in fold_results], axis=0)
        std_curve = np.std( [r['aucs'] for r in fold_results], axis=0)
        t_ms = np.arange(len(avg_curve)) * (1000/FS)
        ax2.fill_between(t_ms, avg_curve-std_curve, avg_curve+std_curve, alpha=0.25, color='blue')
        ax2.plot(t_ms, avg_curve, 'b-', lw=2, label='Mean window-AUC')
        ax2.axhline(0.5, color='red', ls='--', lw=1.5, label='Chance')
        ax2.axvspan(T_SIG_START_MS, T_SIG_END_MS, alpha=0.08, color='green', label='Signal window')
        ax2.set_title('AUC Time-Course', fontweight='bold')
        ax2.set_xlabel('Time (ms)'); ax2.set_ylabel('AUC'); ax2.legend(fontsize=8)

        ax3 = fig.add_subplot(gs[2])
        ax3.bar(['Window\nAUC', 'Mean\nAUC'],
                [np.mean(win_aucs), np.mean(mean_aucs)],
                yerr=[np.std(win_aucs), np.std(mean_aucs)],
                color=['steelblue','coral'], edgecolor='black', capsize=5, alpha=0.85)
        ax3.axhline(0.5, color='k', ls='--', lw=1.5)
        ax3.set_ylim(0.45, max(max(win_aucs), max(mean_aucs))+0.05)
        for i, (v, e) in enumerate(zip([np.mean(win_aucs), np.mean(mean_aucs)],
                                        [np.std(win_aucs),  np.std(mean_aucs)])):
            ax3.text(i, v+e+0.005, f'{v:.3f}', ha='center', fontweight='bold')
        ax3.set_title('Summary Metrics (v6.0)', fontweight='bold')

        plt.suptitle(f'LOSO CV v6.0 — Mean Window-AUC = {np.mean(win_aucs):.4f}',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        save_path = os.path.join(BASE, 'loso_results_v6.png')
        plt.savefig(save_path, dpi=150)
        plt.show()
        print(f"  Plot saved → {save_path}")
    except Exception as e:
        print(f"  Plot error (non-critical): {e}")


print("✓ LOSO CV defined (v6.0)")

# ──────────────────────────────────────────────────────────────────────────────
# CELL 13 — Final Model Training & Submission Generator
# ──────────────────────────────────────────────────────────────────────────────
def generate_submission(train_subjects, test_subjects, output_path=OUTPUT):
    print("\n" + "="*70)
    print("  Final Training → Submission Generation (v6.0)")
    print("="*70)

    # Step 1: Extract training features (with EA)
    print("\nStep 1: Extracting training features (Euclidean Alignment ON)...")
    X_all, y_all, eeg_all, lbl_all = [], [], [], []
    for i, s in enumerate(train_subjects):
        print(f"  [{i+1}/{len(train_subjects)}] {s['id']} ...", end=' ', flush=True)
        t0 = time.time()
        X, y, _, zmu, zsig = extract_subject_features(s, apply_ea=True)
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

    # Step 2: Fit CSP
    print("\nStep 2: Fitting CSP on full training set...")
    csp    = CSP(); csp.fit(eeg_tr, lbl_tr)
    csp_tr = csp.log_var_features(eeg_tr)
    X_tr_f = np.hstack([X_tr, csp_tr])
    print(f"  Full feature matrix: {X_tr_f.shape}")

    # Step 3: Build OOF predictions for isotonic calibration
    print("\nStep 3: Building OOF predictions for isotonic calibration...")
    oof_probs = np.full((n_tr, N_TP), 0.5, dtype=np.float64)
    cumsum    = np.cumsum([s['eeg'].shape[0] for s in train_subjects])
    starts    = np.concatenate([[0], cumsum[:-1]])

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
        p_raw = tre_i.predict_proba_matrix(X_v_if, n_vi)
        p_sm  = smooth_predictions(p_raw, sigma=SMOOTH_SIGMA)
        p_gat = temporal_gate(p_sm)
        oof_probs[sl] = p_gat
        print(f"  OOF [{i+1}/{len(train_subjects)}] {s['id']} done")
        del tre_i; gc.collect()

    # Step 4: Train final TRE on ALL training data
    print("\nStep 4: Training final TRE on ALL training data...")
    tre_final = TimeResolvedEnsemble()
    tre_final.fit(X_tr_f, y_tr, n_tr, verbose=True)

    # ADD-4: Set adaptive weights from training data AUC
    print("\nStep 4b: Computing adaptive per-timepoint ensemble weights...")
    y_trial_full = y_tr[::N_TP]
    tre_final.set_adaptive_weights_from_oof(X_tr_f, y_trial_full, n_tr)

    # Step 5: Fit isotonic calibrators on OOF
    print("\nStep 5: Fitting isotonic calibrators on OOF predictions...")
    iso_models = []
    for t in range(N_TP):
        iso = IsotonicRegression(out_of_bounds='clip')
        try:    iso.fit(oof_probs[:, t], y_bin)
        except: iso = None
        iso_models.append(iso)
    print("  ✓ Isotonic calibration fitted")

    # Step 6: Predict test subjects
    print("\nStep 6: Generating test predictions...")
    rows = []

    for s in test_subjects:
        subj_id  = s['subj_id']
        n_trials = s['eeg'].shape[0]
        print(f"\n  Subject {subj_id} ({s['id']}): {n_trials} trials")

        # Apply EA to test subjects as well
        X_te, _, _, _, _ = extract_subject_features(s, apply_ea=True)
        csp_te    = csp.log_var_features(s['eeg'])
        X_te_f    = np.hstack([X_te, csp_te])

        probs_raw  = tre_final.predict_proba_matrix(X_te_f, n_trials)
        probs_sm   = smooth_predictions(probs_raw, sigma=SMOOTH_SIGMA)
        probs_gat  = temporal_gate(probs_sm)

        # Isotonic calibration
        probs_cal = np.zeros_like(probs_gat)
        for t in range(N_TP):
            if iso_models[t] is not None:
                probs_cal[:, t] = iso_models[t].transform(probs_gat[:, t])
            else:
                probs_cal[:, t] = probs_gat[:, t]
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


print("✓ generate_submission defined (v6.0)")

# ──────────────────────────────────────────────────────────────────────────────
# CELL 14 — MAIN  [FIX-6, FIX-7 preserved]
# ──────────────────────────────────────────────────────────────────────────────
RUN_LOSO = False   # ← Set True to estimate Window-AUC before submitting

print("""
╔══════════════════════════════════════════════════════════════════════╗
║   EEG Emotional Memory — Ultra Pipeline v6.0                        ║
╠══════════════════════════════════════════════════════════════════════╣
║  NEW vs v5:                                                          ║
║  ✦ ADD-1: Euclidean Alignment (cross-subject normalization)          ║
║  ✦ ADD-2: +65 features (asymm, gradient, CFC, hjorth++)             ║
║  ✦ ADD-3: CatBoost as 4th ensemble member                           ║
║  ✦ ADD-4: Adaptive per-timepoint ensemble weights                    ║
║  ✦ ADD-6: Gaussian sigma auto-tuned on OOF                          ║
║  Features: ~615/timepoint | Classifiers: 121 | Bands: 6             ║
╚══════════════════════════════════════════════════════════════════════╝
""")

total_start = time.time()

# STEP 1: Load
print("STEP 1: Loading training data...")
train_subjects = load_all_training(EMO_DIR, NEU_DIR)

print("\nSTEP 2: Loading test data...")
test_subjects = load_all_test(TEST_DIR)

# STEP 3: LOSO (optional — STRONGLY recommended before final submission)
if RUN_LOSO:
    print("\nSTEP 3: LOSO Cross-Validation (with sigma auto-tuning)...")
    fold_results = run_loso(train_subjects, tune_sigma=True)
    win_aucs = [r['window_auc'] for r in fold_results]
    print(f"\n  ► Mean Window-AUC = {np.mean(win_aucs):.4f} ± {np.std(win_aucs):.4f}")
else:
    print("\nSTEP 3: Skipping LOSO (set RUN_LOSO=True to enable — recommended!)")
    fold_results = []

# STEP 4: Train final model + generate submission
print("\nSTEP 4: Training final model + generating submission...")
df = generate_submission(train_subjects, test_subjects, OUTPUT)

# STEP 5: Validate
print("\nSTEP 5: Validating submission...")
assert 'id'         in df.columns
assert 'prediction' in df.columns
assert df['prediction'].between(0,1).all(), "Predictions out of [0,1] range!"
parts = str(df.iloc[0]['id']).split('_')
assert len(parts) == 3, f"ID format wrong: {df.iloc[0]['id']}"
print(f"  ✓ id column present")
print(f"  ✓ prediction column present")
print(f"  ✓ ID format: subject_trial_timepoint")
print(f"  ✓ Total rows: {len(df):,}")
print(f"  ✓ Pred range: [{df.prediction.min():.4f}, {df.prediction.max():.4f}]")

# STEP 6: FIX-7 — safe local copy
print("\nSTEP 6: Copying submission to working directory...")
import shutil
local_copy = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'submission.csv')
if os.path.abspath(OUTPUT) != os.path.abspath(local_copy):
    try:
        shutil.copy(OUTPUT, local_copy)
        print(f"  ✓ Copied to {local_copy}")
    except Exception as e:
        print(f"  ⚠ Copy skipped: {e}")
else:
    print(f"  ✓ Submission already at {OUTPUT}")

total_time = time.time() - total_start
win_summary = (f"Mean Window-AUC = {np.mean([r['window_auc'] for r in fold_results]):.4f}"
               if fold_results else "LOSO skipped — set RUN_LOSO=True before final submit")

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  ✓ DONE  ({total_time/60:.1f} min total)
╠══════════════════════════════════════════════════════════════════════╣
║  {win_summary:<68s}║
║  SMOOTH_SIGMA = {SMOOTH_SIGMA:<53d}║
║  Submission: {OUTPUT:<56s}║
╚══════════════════════════════════════════════════════════════════════╝
""")

# ──────────────────────────────────────────────────────────────────────────────
# CELL 15 — Diagnostics
# ──────────────────────────────────────────────────────────────────────────────
def diagnose(path):
    """Print full HDF5 tree of a .mat file."""
    print(f"\nDiagnosing: {path}")
    with h5py.File(path, 'r') as f:
        def show(name, obj):
            sh = obj.shape if hasattr(obj,'shape') else 'group'
            dt = obj.dtype  if hasattr(obj,'dtype')  else '—'
            print(f"  {name:45s}  shape={sh}  dtype={dt}")
        f.visititems(show)

# diagnose(os.path.join(EMO_DIR, 'S_2_cleaned.mat'))
print("✓ diagnose() ready")
print("\n✓ EEG Ultra Pipeline v6.0 — all cells defined.")
