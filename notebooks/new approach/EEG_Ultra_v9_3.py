
# ============================================================================
# EEG Emotional Memory — Ultra Pipeline v9.3 (Mixture-of-Experts, Trial-Level)
# ============================================================================
#
# Design principles:
# - Generalization-first: one probability per TRIAL, tiled to 200 timepoints
# - Fold-safe ERP template: no LOSO leakage
# - Mixture-of-Experts:
#     * Sentinel      : broad trial-level detector (all features)
#     * ERP specialist: morphology / ERP-template expert
#     * SPEC expert   : spectral + asymmetry expert
#     * RIEM expert   : log-Euclidean covariance expert
#     * Meta-gater    : logistic regression on expert probabilities + disagreement
# - No GRU by default (previous OOF indicated ENN_W=0.00)
# - No ITPC tiling by default (subject-fingerprint risk)
#
# Output strategy:
#   Predict one probability per trial, then repeat it across all 200 timepoints.
#   This satisfies the competition's 50ms consistency requirement by construction.
# ============================================================================

import os, re, time, gc, warnings, subprocess, sys, logging, json
from pathlib import Path
import multiprocessing

os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['TORCHINDUCTOR_DISABLE'] = '1'
logging.getLogger("torch._dynamo").setLevel(logging.CRITICAL)
logging.getLogger("torch._inductor").setLevel(logging.CRITICAL)

for pkg in ['lightgbm', 'tqdm']:
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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.signal import butter, sosfiltfilt, hilbert, detrend as sp_detrend
from scipy.ndimage import gaussian_filter1d
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif

warnings.filterwarnings('ignore')
np.random.seed(42)
GPU_AVAILABLE = torch.cuda.is_available() if HAVE_TORCH else False
N_JOBS = multiprocessing.cpu_count()

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
FS = 200
N_TP = 200
N_CH = 16
WIN = 40
ERP_WIN_START = int(0.100 * FS)   # 20
ERP_WIN_END   = int(0.350 * FS)   # 70
BASE_WIN_START = 0
BASE_WIN_END   = int(0.100 * FS)  # 20
LATE_WIN_START = ERP_WIN_END      # 70
LATE_WIN_END   = int(0.700 * FS)  # 140

BANDS = {
    'delta': (1.0, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'sigma': (12.0, 16.0),
    'beta':  (13.0, 30.0),
}

CHANNELS = ['c3','c4','o1','o2','cp3','f3','f4','cp4',
            'c5','cz','c6','cp5','p7','pz','p8','cp6']
CH = {c: i for i, c in enumerate(CHANNELS)}
ASYM_PAIRS = [('c3','c4'), ('cp3','cp4'), ('f3','f4'), ('p7','p8')]
ASYM_BANDS = ['theta', 'alpha', 'beta']

N_FEAT_SEL_SPEC = 180
N_FEAT_SEL_SENT = 220

BASE = r'D:\EEG Project\Project Overview and Specifications\eeg_competition'
EMO_DIR = os.path.join(BASE, 'training', 'sleep_emo')
NEU_DIR = os.path.join(BASE, 'training', 'sleep_neu')
TEST_DIR = os.path.join(BASE, 'testing')
OUTPUT = os.path.join(BASE, 'submission_v9_3.csv')
RUN_LOSO = True

print(f"GPU_AVAILABLE={GPU_AVAILABLE} HAVE_TORCH={HAVE_TORCH} TORCH_DEVICE={TORCH_DEVICE}")
print("✓ Config loaded — v9.3 (Mixture-of-Experts, Trial-Level)")
print(f" ERP window : {ERP_WIN_START*1000//FS}-{ERP_WIN_END*1000//FS} ms")
print(f" Late window: {LATE_WIN_START*1000//FS}-{LATE_WIN_END*1000//FS} ms")
print(" Experts    : Sentinel + ERP + SPEC + RIEM + Meta-Gater")
print(" Output     : flat trial probability tiled to 200 timepoints")

# ─────────────────────────────────────────────────────────────────────────────
# Robust loader
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
                    grp = f[k]
                    break
        if grp is None:
            raise ValueError(f"Cannot find 'data' struct in {path}")

        trial_raw = _resolve_field(f, grp, 'trial')
        if trial_raw.ndim == 3:
            sh = trial_raw.shape
            if sh[2] == N_CH and sh[1] == N_TP:
                trial_raw = trial_raw.transpose(0, 2, 1)
            elif sh[0] == N_CH and sh[1] == N_TP:
                trial_raw = trial_raw.transpose(2, 0, 1)
            elif sh[0] == N_TP and sh[1] == N_CH:
                trial_raw = trial_raw.transpose(2, 1, 0)
        elif trial_raw.ndim == 2:
            trial_raw = trial_raw.T[np.newaxis]
        eeg = trial_raw.astype(np.float32)

        if label_override is not None:
            labels = np.full(eeg.shape[0], label_override, dtype=int)
        else:
            try:
                ti = _resolve_field(f, grp, 'trialinfo')
                if ti.ndim == 2 and ti.shape[0] == eeg.shape[0]:
                    labels = ti[:, 0].astype(int)
                elif ti.ndim == 2 and ti.shape[1] == eeg.shape[0]:
                    labels = ti[0, :].astype(int)
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
            'id': stem
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
    x = x - x.mean(axis=0, keepdims=True)  # average reference per timepoint
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
        covs[i] = (x @ x.T) / max(n_tp - 1, 1)
    R_mean = covs.mean(axis=0)
    eps = 1e-6 * np.trace(R_mean) / n_ch
    R_mean += eps * np.eye(n_ch)
    try:
        eigvals, eigvecs = np.linalg.eigh(R_mean)
        eigvals = np.maximum(eigvals, 1e-10)
        R_inv_sqrt = eigvecs @ np.diag(eigvals**(-0.5)) @ eigvecs.T
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
    C = (seg_c @ seg_c.T) / max(seg_c.shape[1] - 1, 1)
    eps = 1e-6 * (np.trace(C) / n_ch + 1e-12)
    C += eps * np.eye(n_ch)
    try:
        vals, vecs = np.linalg.eigh(C)
        vals = np.maximum(vals, 1e-10)
        logC = vecs @ np.diag(np.log(vals)) @ vecs.T
        idx = np.triu_indices(n_ch)
        return logC[idx].astype(np.float32)
    except Exception:
        return np.zeros(n_feat, dtype=np.float32)


def frontal_theta_asymmetry(theta_power, t0, t1):
    f3t = np.mean(theta_power[CH['f3'], t0:t1]) + 1e-12
    f4t = np.mean(theta_power[CH['f4'], t0:t1]) + 1e-12
    pzt = np.mean(theta_power[CH['pz'], t0:t1]) + 1e-12
    czt = np.mean(theta_power[CH['cz'], t0:t1]) + 1e-12
    fai_theta = float(np.log(f4t) - np.log(f3t))
    fp_ratio = float(np.log((f3t + f4t)/2.0) - np.log(pzt))
    cz_theta = float(np.log(czt))
    return [fai_theta, fp_ratio, cz_theta]

# ─────────────────────────────────────────────────────────────────────────────
# ERP template cache (LOSO-safe)
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_subject_for_cache(subj_dict):
    eeg = subj_dict['eeg'].copy()
    labels = subj_dict['labels']
    eeg, _, _ = zscore_subject(eeg)
    eeg = euclidean_alignment(eeg)
    eeg_proc = np.array([preprocess_trial(eeg[i]) for i in range(eeg.shape[0])], dtype=np.float32)
    emo_mask = labels == 2
    neu_mask = labels == 1
    erp_emo = eeg_proc[emo_mask].mean(axis=0) if emo_mask.sum() > 1 else None
    erp_neu = eeg_proc[neu_mask].mean(axis=0) if neu_mask.sum() > 1 else None
    return {'eeg_proc': eeg_proc, 'erp_emo': erp_emo, 'erp_neu': erp_neu}


def compute_fold_erp_template_from_cache(cache_list, exclude_index, erp_start=ERP_WIN_START, erp_end=ERP_WIN_END):
    emo_list, neu_list = [], []
    for j, c in enumerate(cache_list):
        if j == exclude_index:
            continue
        if c['erp_emo'] is not None:
            emo_list.append(c['erp_emo'])
        if c['erp_neu'] is not None:
            neu_list.append(c['erp_neu'])
    if len(emo_list) == 0 or len(neu_list) == 0:
        return np.zeros((N_CH, erp_end-erp_start), dtype=np.float32)
    grand_emo = np.mean(emo_list, axis=0)
    grand_neu = np.mean(neu_list, axis=0)
    return (grand_emo - grand_neu)[:, erp_start:erp_end].astype(np.float32)


def compute_global_erp_template(train_subjects):
    cache = [preprocess_subject_for_cache(s) for s in train_subjects]
    return compute_fold_erp_template_from_cache(cache, exclude_index=-1)

# ─────────────────────────────────────────────────────────────────────────────
# Trial-level feature extraction (Mixture-of-Experts blocks)
# ─────────────────────────────────────────────────────────────────────────────
def _band_power_block(trial, window):
    t0, t1 = window
    feats = []
    band_cache = {}
    for bname, (lo, hi) in BANDS.items():
        bf = bandpass(trial, lo, hi, FS)
        pw = (np.abs(hilbert(bf, axis=-1))**2)
        band_cache[bname] = pw
        feats.extend(np.log1p(pw[:, t0:t1].mean(axis=1)).tolist())
    return np.array(feats, dtype=np.float32), band_cache


def _baseline_corrected_band_block(band_cache, win_signal=(ERP_WIN_START, ERP_WIN_END), win_base=(BASE_WIN_START, BASE_WIN_END)):
    s0, s1 = win_signal
    b0, b1 = win_base
    feats = []
    for bname in BANDS:
        p_sig = band_cache[bname][:, s0:s1].mean(axis=1)
        p_bas = band_cache[bname][:, b0:b1].mean(axis=1)
        feats.extend((np.log1p(p_sig) - np.log1p(p_bas + 1e-12)).tolist())
    return np.array(feats, dtype=np.float32)


def _late_band_block(band_cache, win=(LATE_WIN_START, LATE_WIN_END)):
    t0, t1 = win
    feats = []
    for bname in BANDS:
        feats.extend(np.log1p(band_cache[bname][:, t0:t1].mean(axis=1)).tolist())
    return np.array(feats, dtype=np.float32)


def _erp_morphology_block(trial, erp_template=None, win=(ERP_WIN_START, ERP_WIN_END)):
    t0, t1 = win
    erp = trial[:, t0:t1]
    feats = []
    feats.extend(erp.max(axis=1).tolist())
    feats.extend(erp.min(axis=1).tolist())
    feats.extend(erp.std(axis=1).tolist())
    feats.extend((erp.max(axis=1) - erp.min(axis=1)).tolist())
    if erp_template is not None:
        for c in range(N_CH):
            x = erp[c]
            tmpl = erp_template[c]
            if np.std(x) > 1e-8 and np.std(tmpl) > 1e-8:
                feats.append(float(np.corrcoef(x, tmpl)[0, 1]))
            else:
                feats.append(0.0)
    else:
        feats.extend([0.0] * N_CH)
    return np.array(feats, dtype=np.float32)


def _asymmetry_block(band_cache, win=(ERP_WIN_START, ERP_WIN_END)):
    t0, t1 = win
    feats = []
    for ch1, ch2 in ASYM_PAIRS:
        for bn in ASYM_BANDS:
            p1 = band_cache[bn][CH[ch1], t0:t1].mean() + 1e-12
            p2 = band_cache[bn][CH[ch2], t0:t1].mean() + 1e-12
            feats.append(float(np.log(p2) - np.log(p1)))
    return np.array(feats, dtype=np.float32)


def _theta_ratio_block(band_cache, win=(ERP_WIN_START, ERP_WIN_END)):
    t0, t1 = win
    theta_pow = band_cache['theta'][:, t0:t1].mean(axis=1)
    alpha_pow = band_cache['alpha'][:, t0:t1].mean(axis=1)
    sigma_pow = band_cache['sigma'][:, LATE_WIN_START:LATE_WIN_END].mean(axis=1)
    feats = []
    feats.extend((np.log1p(theta_pow) - np.log1p(alpha_pow + 1e-12)).tolist())
    feats.extend(np.log1p(sigma_pow).tolist())
    feats.extend(frontal_theta_asymmetry(band_cache['theta'], t0, t1))
    return np.array(feats, dtype=np.float32)


def _riemann_block(band_cache, win=(ERP_WIN_START, ERP_WIN_END)):
    t0, t1 = win
    return log_euclidean_cov_features(band_cache['theta'][:, t0:t1])


def extract_trial_feature_blocks(trial, erp_template=None):
    # trial is already preprocessed
    band_erp, band_cache = _band_power_block(trial, (ERP_WIN_START, ERP_WIN_END))
    band_basecorr = _baseline_corrected_band_block(band_cache)
    band_late = _late_band_block(band_cache)
    erp_morph = _erp_morphology_block(trial, erp_template)
    asym = _asymmetry_block(band_cache)
    ratios = _theta_ratio_block(band_cache)
    riem = _riemann_block(band_cache)

    # Experts
    erp_feat = np.concatenate([erp_morph], axis=0)
    spec_feat = np.concatenate([band_erp, band_basecorr, band_late, asym, ratios], axis=0)
    riem_feat = np.concatenate([riem], axis=0)
    sent_feat = np.concatenate([erp_feat, spec_feat, riem_feat], axis=0)

    return {
        'erp': erp_feat.astype(np.float32),
        'spec': spec_feat.astype(np.float32),
        'riem': riem_feat.astype(np.float32),
        'sent': sent_feat.astype(np.float32),
    }


def extract_subject_trial_features(subj_dict, erp_template=None, precomputed_proc=None):
    labels = subj_dict['labels']
    if precomputed_proc is None:
        eeg = subj_dict['eeg'].copy()
        eeg, _, _ = zscore_subject(eeg)
        eeg = euclidean_alignment(eeg)
        eeg_proc = np.array([preprocess_trial(eeg[i]) for i in range(eeg.shape[0])], dtype=np.float32)
    else:
        eeg_proc = precomputed_proc

    blocks = {'erp': [], 'spec': [], 'riem': [], 'sent': []}
    for i in range(eeg_proc.shape[0]):
        fb = extract_trial_feature_blocks(eeg_proc[i], erp_template)
        for k in blocks:
            blocks[k].append(fb[k])
    for k in blocks:
        blocks[k] = np.vstack(blocks[k]).astype(np.float32)
    return blocks, labels.copy(), eeg_proc

# ─────────────────────────────────────────────────────────────────────────────
# Mixture-of-Experts models
# ─────────────────────────────────────────────────────────────────────────────
class ExpertModel:
    def __init__(self, model_type='lda', k_best=None):
        self.model_type = model_type
        self.k_best = k_best
        self.sel = None
        self.sc = RobustScaler()
        self.model = None

    def _make_model(self):
        if self.model_type == 'lda':
            return LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        elif self.model_type == 'logreg':
            return LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
        elif self.model_type == 'lgbm':
            params = dict(
                n_estimators=350 if GPU_AVAILABLE else 250,
                num_leaves=31,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.7,
                min_child_samples=5,
                class_weight='balanced',
                reg_alpha=0.1,
                reg_lambda=1.0,
                n_jobs=4,
                random_state=42,
                verbose=-1,
            )
            if GPU_AVAILABLE:
                params.update({'device': 'gpu', 'gpu_use_dp': False, 'max_bin': 255})
            else:
                params.update({'device': 'cpu'})
            return lgb.LGBMClassifier(**params)
        else:
            raise ValueError(self.model_type)

    def fit(self, X, y):
        if self.k_best is not None and self.k_best < X.shape[1]:
            self.sel = SelectKBest(f_classif, k=self.k_best)
            Xs = self.sel.fit_transform(X, y)
        else:
            Xs = X
        Xs = self.sc.fit_transform(Xs)
        self.model = self._make_model()
        self.model.fit(Xs, y)
        return self

    def predict_proba(self, X):
        Xs = self.sel.transform(X) if self.sel is not None else X
        Xs = self.sc.transform(Xs)
        try:
            return self.model.predict_proba(Xs)[:, 1]
        except Exception:
            scores = self.model.decision_function(Xs)
            return 1 / (1 + np.exp(-scores))


class MoETrialClassifier:
    def __init__(self):
        self.erp_expert = ExpertModel('lda', k_best=min(96, N_CH*5))
        self.spec_expert = ExpertModel('lgbm', k_best=N_FEAT_SEL_SPEC)
        self.riem_expert = ExpertModel('lda', k_best=None)
        self.sentinel = ExpertModel('logreg', k_best=N_FEAT_SEL_SENT)
        self.meta = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
        self.meta_sc = RobustScaler()

    @staticmethod
    def _build_meta_features(p_sent, p_erp, p_spec, p_riem):
        P = np.vstack([p_sent, p_erp, p_spec, p_riem]).T
        mean_p = P.mean(axis=1)
        std_p  = P.std(axis=1)
        mx_p   = P.max(axis=1)
        mn_p   = P.min(axis=1)
        disagree = np.abs(p_erp - p_spec) + np.abs(p_erp - p_riem) + np.abs(p_spec - p_riem)
        return np.column_stack([P, mean_p, std_p, mx_p, mn_p, disagree]).astype(np.float32)

    def fit(self, X_blocks, y):
        y_bin = (y == 2).astype(int)

        # Base experts on full train
        self.erp_expert.fit(X_blocks['erp'], y_bin)
        self.spec_expert.fit(X_blocks['spec'], y_bin)
        self.riem_expert.fit(X_blocks['riem'], y_bin)
        self.sentinel.fit(X_blocks['sent'], y_bin)

        # Internal OOF for meta-gater
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        oof_sent = np.zeros(len(y_bin), dtype=np.float32)
        oof_erp  = np.zeros(len(y_bin), dtype=np.float32)
        oof_spec = np.zeros(len(y_bin), dtype=np.float32)
        oof_riem = np.zeros(len(y_bin), dtype=np.float32)

        for tr_idx, va_idx in skf.split(X_blocks['sent'], y_bin):
            e1 = ExpertModel('lda', k_best=min(96, X_blocks['erp'].shape[1]))
            e2 = ExpertModel('lgbm', k_best=min(N_FEAT_SEL_SPEC, X_blocks['spec'].shape[1]))
            e3 = ExpertModel('lda', k_best=None)
            s1 = ExpertModel('logreg', k_best=min(N_FEAT_SEL_SENT, X_blocks['sent'].shape[1]))

            e1.fit(X_blocks['erp'][tr_idx], y_bin[tr_idx])
            e2.fit(X_blocks['spec'][tr_idx], y_bin[tr_idx])
            e3.fit(X_blocks['riem'][tr_idx], y_bin[tr_idx])
            s1.fit(X_blocks['sent'][tr_idx], y_bin[tr_idx])

            oof_erp[va_idx]  = e1.predict_proba(X_blocks['erp'][va_idx])
            oof_spec[va_idx] = e2.predict_proba(X_blocks['spec'][va_idx])
            oof_riem[va_idx] = e3.predict_proba(X_blocks['riem'][va_idx])
            oof_sent[va_idx] = s1.predict_proba(X_blocks['sent'][va_idx])

        meta_X = self._build_meta_features(oof_sent, oof_erp, oof_spec, oof_riem)
        meta_Xs = self.meta_sc.fit_transform(meta_X)
        self.meta.fit(meta_Xs, y_bin)
        return self

    def predict_trial_proba(self, X_blocks):
        p_erp  = self.erp_expert.predict_proba(X_blocks['erp'])
        p_spec = self.spec_expert.predict_proba(X_blocks['spec'])
        p_riem = self.riem_expert.predict_proba(X_blocks['riem'])
        p_sent = self.sentinel.predict_proba(X_blocks['sent'])
        meta_X = self._build_meta_features(p_sent, p_erp, p_spec, p_riem)
        meta_Xs = self.meta_sc.transform(meta_X)
        p_final = self.meta.predict_proba(meta_Xs)[:, 1]
        return {
            'p_final': p_final.astype(np.float32),
            'p_sent': p_sent.astype(np.float32),
            'p_erp': p_erp.astype(np.float32),
            'p_spec': p_spec.astype(np.float32),
            'p_riem': p_riem.astype(np.float32),
        }

# ─────────────────────────────────────────────────────────────────────────────
# Metric / plotting
# ─────────────────────────────────────────────────────────────────────────────
def tile_trial_probs(p_trial, n_tp=N_TP):
    return np.tile(p_trial[:, None], (1, n_tp)).astype(np.float32)


def window_auc_score(probs, y_bin, min_ms=50, win_tp=10):
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
        best_len = run_l
    return {'window_auc': float(best_auc), 'aucs': aucs, 'mean_auc': float(aucs.mean()), 'dur_ms': best_len * (1000/FS)}


def plot_loso_results(fold_results, train_subjects, save_path):
    try:
        n = len(fold_results)
        win_aucs = [r['window_auc'] for r in fold_results]
        mean_aucs = [r['mean_auc'] for r in fold_results]

        fig = plt.figure(figsize=(22, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        colors = ['#2ecc71' if a > 0.5 else '#e74c3c' for a in win_aucs]
        ax1.bar(range(1, n+1), win_aucs, color=colors, edgecolor='black', lw=0.5)
        ax1.axhline(0.5, color='k', ls='--', lw=1.5)
        ax1.axhline(np.mean(win_aucs), color='blue', lw=2)
        ax1.set_title('Window-AUC per Fold (v9.3)', fontweight='bold')
        ax1.set_xlabel('Subject')
        ax1.set_ylabel('Window-AUC')
        ax1.set_ylim(0.4, 1.0)

        ax2 = fig.add_subplot(gs[0, 1])
        avg_curve = np.mean([r['aucs'] for r in fold_results], axis=0)
        std_curve = np.std([r['aucs'] for r in fold_results], axis=0)
        t_ms = np.arange(len(avg_curve)) * (1000/FS)
        ax2.fill_between(t_ms, avg_curve - std_curve, avg_curve + std_curve, alpha=0.25, color='blue')
        ax2.plot(t_ms, avg_curve, 'b-', lw=2)
        ax2.axhline(0.5, color='red', ls='--', lw=1.5)
        ax2.axvspan(ERP_WIN_START*1000/FS, ERP_WIN_END*1000/FS, alpha=0.25, color='green', label='ERP 100-350ms')
        ax2.set_title('AUC Time-Course v9.3', fontweight='bold')
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('AUC')
        ax2.legend(fontsize=8)

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.bar(['Window\nAUC', 'Mean\nAUC'], [np.mean(win_aucs), np.mean(mean_aucs)],
                yerr=[np.std(win_aucs), np.std(mean_aucs)], color=['steelblue', 'coral'],
                edgecolor='black', capsize=5)
        ax3.axhline(0.5, color='k', ls='--', lw=1.5)
        ax3.set_title('Summary v9.3', fontweight='bold')

        ax4 = fig.add_subplot(gs[1, :])
        cmap = plt.get_cmap('tab20')
        for i, (result, subj) in enumerate(zip(fold_results, train_subjects)):
            lbl = f"{subj['id'][:6]} {result['window_auc']:.3f}"
            ax4.plot(t_ms, result['aucs'], alpha=0.7, lw=1, color=cmap(i / max(1,n-1)), label=lbl)
        ax4.axhline(0.5, color='k', ls='--', lw=1.5)
        ax4.axvspan(ERP_WIN_START*1000/FS, ERP_WIN_END*1000/FS, alpha=0.15, color='green')
        ax4.set_title('Per-Subject AUC Time-Course (v9.3)', fontweight='bold')
        ax4.set_xlabel('Time (ms)')
        ax4.set_ylabel('AUC')
        ax4.set_ylim(0.40, 0.65)
        ax4.legend(fontsize=7, ncol=5, loc='upper right')

        plt.suptitle(f"LOSO CV v9.3 — Window-AUC={np.mean(win_aucs):.4f}", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.show()
        print(f" Plot saved → {save_path}")
    except Exception as e:
        print(f" Plot error (non-critical): {e}")

# ─────────────────────────────────────────────────────────────────────────────
# LOSO Cross-Validation
# ─────────────────────────────────────────────────────────────────────────────
def run_loso_v93(train_subjects):
    print("\n" + "="*70)
    print(" LOSO Cross-Validation — v9.3 (Mixture-of-Experts, Trial-Level)")
    print("="*70)

    print("\nStep 0: Precomputing subject caches (preprocessed EEG + class ERPs)...")
    subject_cache = []
    for i, s in enumerate(train_subjects):
        t0 = time.time()
        cache = preprocess_subject_for_cache(s)
        subject_cache.append(cache)
        print(f" [{i+1:2d}/{len(train_subjects)}] {s['id']}: eeg_proc={cache['eeg_proc'].shape} ({time.time()-t0:.1f}s)")

    fold_results = []
    oof_trial_probs = []
    oof_labels = []
    n = len(train_subjects)

    for val_i in range(n):
        t_fold = time.time()
        val_subj = train_subjects[val_i]
        fold_template = compute_fold_erp_template_from_cache(subject_cache, val_i)

        # Build training set blocks
        tr_blocks_list = {'erp': [], 'spec': [], 'riem': [], 'sent': []}
        y_tr_list = []
        for j, s in enumerate(train_subjects):
            if j == val_i:
                continue
            blocks, labels, _ = extract_subject_trial_features(s, erp_template=fold_template,
                                                               precomputed_proc=subject_cache[j]['eeg_proc'])
            for k in tr_blocks_list:
                tr_blocks_list[k].append(blocks[k])
            y_tr_list.append(labels)
        X_tr_blocks = {k: np.vstack(v).astype(np.float32) for k, v in tr_blocks_list.items()}
        y_tr = np.concatenate(y_tr_list)
        y_tr_bin = (y_tr == 2).astype(int)

        # Validation blocks
        X_val_blocks, y_val, _ = extract_subject_trial_features(val_subj, erp_template=fold_template,
                                                                precomputed_proc=subject_cache[val_i]['eeg_proc'])
        y_val_bin = (y_val == 2).astype(int)

        moe = MoETrialClassifier().fit(X_tr_blocks, y_tr)
        pred = moe.predict_trial_proba(X_val_blocks)
        p_trial = pred['p_final']
        trial_auc = roc_auc_score(y_val_bin, p_trial)

        probs_2d = tile_trial_probs(p_trial, N_TP)
        metric = window_auc_score(np.clip(probs_2d, 0.01, 0.99), y_val_bin)
        fold_results.append(metric)
        oof_trial_probs.append(p_trial)
        oof_labels.append(y_val)

        print(f" Fold {val_i+1:2d} | {val_subj['id'][:25]:25s} | trial_AUC={trial_auc:.4f} | wAUC={metric['window_auc']:.4f} | mAUC={metric['mean_auc']:.4f} ({time.time()-t_fold:.0f}s)")
        gc.collect()

    win_aucs = [r['window_auc'] for r in fold_results]
    mean_aucs = [r['mean_auc'] for r in fold_results]
    print(f"\n ╔═══════════════════════════════════════════════════════╗")
    print(f" ║ Window AUC  : {np.mean(win_aucs):.4f} ± {np.std(win_aucs):.4f}                 ║")
    print(f" ║ Mean AUC    : {np.mean(mean_aucs):.4f} ± {np.std(mean_aucs):.4f}                 ║")
    print(f" ║ Best fold   : {max(win_aucs):.4f}                              ║")
    print(f" ╚═══════════════════════════════════════════════════════╝")

    plot_loso_results(fold_results, train_subjects, os.path.join(BASE, 'loso_results_v9_3.png'))
    return fold_results, oof_trial_probs, oof_labels

# ─────────────────────────────────────────────────────────────────────────────
# Final training + submission
# ─────────────────────────────────────────────────────────────────────────────
def fit_trial_platt(oof_trial_probs, oof_labels):
    p = np.concatenate(oof_trial_probs)[:, None]
    y = (np.concatenate(oof_labels) == 2).astype(int)
    cal = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
    cal.fit(p, y)
    return cal


def generate_submission_v93(train_subjects, test_subjects, output_path=OUTPUT, trial_calibrator=None):
    print("\n" + "="*70)
    print(" Final Training → Submission Generation (v9.3)")
    print("="*70)

    print("\nStep 1: Preparing subject caches...")
    train_cache = [preprocess_subject_for_cache(s) for s in train_subjects]
    global_template = compute_fold_erp_template_from_cache(train_cache, exclude_index=-1)

    print("Step 2: Extracting all training trial-level features...")
    tr_blocks_list = {'erp': [], 'spec': [], 'riem': [], 'sent': []}
    y_list = []
    for s, c in zip(train_subjects, train_cache):
        blocks, labels, _ = extract_subject_trial_features(s, erp_template=global_template, precomputed_proc=c['eeg_proc'])
        for k in tr_blocks_list:
            tr_blocks_list[k].append(blocks[k])
        y_list.append(labels)
    X_tr_blocks = {k: np.vstack(v).astype(np.float32) for k, v in tr_blocks_list.items()}
    y_tr = np.concatenate(y_list)

    print("Step 3: Fitting MoE on ALL training trials...")
    moe = MoETrialClassifier().fit(X_tr_blocks, y_tr)

    print("Step 4: Predicting test subjects...")
    rows = []
    for s in test_subjects:
        subj_id = s['subj_id']
        eeg = s['eeg'].copy()
        eeg, _, _ = zscore_subject(eeg)
        eeg = euclidean_alignment(eeg)
        eeg_proc = np.array([preprocess_trial(eeg[i]) for i in range(eeg.shape[0])], dtype=np.float32)
        X_te_blocks, _, _ = extract_subject_trial_features(s, erp_template=global_template, precomputed_proc=eeg_proc)
        pred = moe.predict_trial_proba(X_te_blocks)
        p_trial = pred['p_final']
        if trial_calibrator is not None:
            p_trial = trial_calibrator.predict_proba(p_trial[:, None])[:, 1]
        probs_2d = np.clip(tile_trial_probs(p_trial, N_TP), 0.01, 0.99)
        print(f" Subject {subj_id} ({s['id']}): trials={probs_2d.shape[0]} range=[{probs_2d.min():.3f}, {probs_2d.max():.3f}] mean={probs_2d.mean():.4f}")
        for tr in range(probs_2d.shape[0]):
            for tp in range(N_TP):
                rows.append({'id': f"{subj_id}_{tr}_{tp}", 'prediction': float(probs_2d[tr, tp])})

    df = pd.DataFrame(rows)
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
║ EEG Emotional Memory — Ultra Pipeline v9.3 (Mixture-of-Experts)        ║
╠══════════════════════════════════════════════════════════════════════════╣
║ Architecture: Sentinel + ERP + SPEC + RIEM + Meta-Gater                ║
║ Labels are TRIAL-level → predict one probability per trial             ║
║ Output tiled to 200 timepoints (metric-aligned consistency)            ║
║ No GRU / no per-timepoint TRE / no ITPC tiling                         ║
║ Fold-safe ERP template (LOSO-safe)                                     ║
╚══════════════════════════════════════════════════════════════════════════╝
""")
    total_start = time.time()
    print("STEP 1: Loading training data...")
    train_subjects = load_all_training(EMO_DIR, NEU_DIR)
    print("\nSTEP 2: Loading test data...")
    test_subjects = load_all_test(TEST_DIR)

    if RUN_LOSO:
        print("\nSTEP 3: LOSO Cross-Validation...")
        fold_results, oof_trial_probs, oof_labels = run_loso_v93(train_subjects)
        print(f"\n ► Mean Window-AUC = {np.mean([r['window_auc'] for r in fold_results]):.4f} ± {np.std([r['window_auc'] for r in fold_results]):.4f}")
        print("\nSTEP 4: Fitting trial-level Platt calibrator...")
        trial_cal = fit_trial_platt(oof_trial_probs, oof_labels)
    else:
        fold_results = []
        trial_cal = None

    print("\nSTEP 5: Final training + submission generation...")
    df = generate_submission_v93(train_subjects, test_subjects, OUTPUT, trial_calibrator=trial_cal)

    print("\nSTEP 6: Validating submission format...")
    assert 'id' in df.columns and 'prediction' in df.columns
    assert df['prediction'].between(0, 1).all(), "Predictions out of [0,1]!"
    parts = str(df.iloc[0]['id']).split('_')
    assert len(parts) == 3, f"ID format wrong: {df.iloc[0]['id']}"
    print(f" ✓ Total rows: {len(df):,}")
    print(f" ✓ Pred range: [{df.prediction.min():.4f}, {df.prediction.max():.4f}]")

    total_time = time.time() - total_start
    print(f"\n✓ DONE ({total_time/60:.1f} min total)")
