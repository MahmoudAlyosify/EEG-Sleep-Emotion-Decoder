
# ============================================================================
# EEG Emotional Memory — Ultra Pipeline v10.4b (OPTUNA WRAPPER)
# ============================================================================
# Strategy:
#   Use the proven v10.4 LR/MDM-focused pipeline as a base module, then apply
#   Optuna tuning to a constrained, high-ROI search space:
#       - LR C
#       - Zone LR C values
#       - K-best sizes
#       - LightGBM helper params
#       - Backbone blend weights (LR/MDM/LGBM)
#       - Temporal alphas A/B/C
#
# Practical design:
#   - FAST search on representative LOSO folds
#   - FULL verify on all folds for best parameters
#   - Same paths and final submission format as your environment
# ============================================================================

import os, gc, time, subprocess, sys, warnings
warnings.filterwarnings('ignore')

for pkg in ['optuna']:
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', pkg, '-q'], check=False)
    except Exception:
        pass

import numpy as np
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
except Exception as e:
    raise RuntimeError('Optuna is required for v10.4b') from e

# Base pipeline import (same folder)
import EEG_Ultra_v10_4_lr_mdm_focus as base

# Config
RUN_OPTUNA = True
OPTUNA_TRIALS = 30      # raise to 40-60 if you want deeper search
FAST_FOLDS = [0, 1, 2, 5, 8, 11]
STUDY_NAME = 'eeg_v10_4b_optuna'
STUDY_DB = os.path.join(base.BASE, 'optuna_v10_4b.sqlite3')
FINAL_OUTPUT = os.path.join(base.BASE, 'submission_v10_4b_optuna.csv')


def apply_params(params):
    """Patch the base module globals/classes for one trial."""
    base.K_BEST_LR = int(params['k_backbone'])
    base.K_BEST_ZONE = int(params['k_zone'])
    # Patch LGBM helper params
    base.LGBM_PARAMS.update({
        'n_estimators': int(params['lgbm_n_estimators']),
        'num_leaves': int(params['lgbm_num_leaves']),
        'max_depth': int(params['lgbm_max_depth']),
        'learning_rate': float(params['lgbm_learning_rate']),
        'subsample': float(params['lgbm_subsample']),
        'colsample_bytree': float(params['lgbm_colsample_bytree']),
        'min_child_samples': int(params['lgbm_min_child_samples']),
        'reg_alpha': float(params['lgbm_reg_alpha']),
        'reg_lambda': float(params['lgbm_reg_lambda']),
    })
    # Blend weights (normalized later in backbone.blend)
    base.BEST_BLEND = {
        'lr': float(params['w_lr']),
        'mdm': float(params['w_mdm']),
        'lgbm': float(params['w_lgbm']),
    }
    base.BEST_ALPHAS = (
        float(params['alpha_A']),
        float(params['alpha_B']),
        float(params['alpha_C']),
    )

    # Monkey-patch __init__ to use tuned C/k values without rewriting the file
    orig_backbone_init = base.BackboneClassifier.__init__
    orig_mod_init = base.TemporalModulator.__init__

    def tuned_backbone_init(self, use_mdm=base.HAVE_PYRIEMANN):
        self.lr = base.TabularBinaryModel('lr', k_best=int(params['k_backbone']), C=float(params['lr_C']))
        self.lgbm = base.TabularBinaryModel('lgbm', k_best=None)
        self.use_mdm = use_mdm and base.HAVE_PYRIEMANN
        self.mdm = None
        self.weights = base.BEST_BLEND.copy()

    def tuned_mod_init(self):
        self.modA = base.TabularBinaryModel('lr', k_best=int(params['k_zone']), C=float(params['zoneA_C']))
        self.modB = base.TabularBinaryModel('lr', k_best=int(params['k_zone']), C=float(params['zoneB_C']))
        self.modC = base.TabularBinaryModel('lr', k_best=int(params['k_zone_c']), C=float(params['zoneC_C']))
        self.alphas = base.BEST_ALPHAS

    base.BackboneClassifier.__init__ = tuned_backbone_init
    base.TemporalModulator.__init__ = tuned_mod_init
    return orig_backbone_init, orig_mod_init


def restore_inits(orig_backbone_init, orig_mod_init):
    base.BackboneClassifier.__init__ = orig_backbone_init
    base.TemporalModulator.__init__ = orig_mod_init


def sample_trial_params(trial):
    return {
        'lr_C': trial.suggest_float('lr_C', 0.15, 1.50, log=True),
        'zoneA_C': trial.suggest_float('zoneA_C', 0.15, 1.50, log=True),
        'zoneB_C': trial.suggest_float('zoneB_C', 0.15, 1.50, log=True),
        'zoneC_C': trial.suggest_float('zoneC_C', 0.08, 1.00, log=True),
        'k_backbone': trial.suggest_int('k_backbone', 140, 280, step=20),
        'k_zone': trial.suggest_int('k_zone', 64, 128, step=16),
        'k_zone_c': trial.suggest_int('k_zone_c', 16, 48, step=8),
        'lgbm_n_estimators': trial.suggest_int('lgbm_n_estimators', 220, 520, step=40),
        'lgbm_num_leaves': trial.suggest_int('lgbm_num_leaves', 15, 63, step=8),
        'lgbm_max_depth': trial.suggest_int('lgbm_max_depth', 3, 6),
        'lgbm_learning_rate': trial.suggest_float('lgbm_learning_rate', 0.02, 0.08, log=True),
        'lgbm_subsample': trial.suggest_float('lgbm_subsample', 0.70, 0.95),
        'lgbm_colsample_bytree': trial.suggest_float('lgbm_colsample_bytree', 0.60, 0.90),
        'lgbm_min_child_samples': trial.suggest_int('lgbm_min_child_samples', 4, 12),
        'lgbm_reg_alpha': trial.suggest_float('lgbm_reg_alpha', 0.0, 0.5),
        'lgbm_reg_lambda': trial.suggest_float('lgbm_reg_lambda', 0.5, 2.0),
        'w_lr': trial.suggest_float('w_lr', 0.45, 0.90),
        'w_mdm': trial.suggest_float('w_mdm', 0.00, 0.30),
        'w_lgbm': trial.suggest_float('w_lgbm', 0.00, 0.20),
        'alpha_A': trial.suggest_float('alpha_A', 0.66, 0.84),
        'alpha_B': trial.suggest_float('alpha_B', 0.08, 0.22),
        'alpha_C': trial.suggest_float('alpha_C', 0.00, 0.08),
    }


def evaluate_subset(train_subjects, cache, csp_global, fold_indices):
    fold_scores, fold_trial_aucs = [], []
    subj_weights = base.compute_subject_weights_heuristic(train_subjects, cache)
    for j, i in enumerate(fold_indices):
        X_tr, y_tr, cov_tr, X_val, y_val, cov_val = base.build_fold_matrices(train_subjects, cache, i, csp_global)
        sw = base.expand_subject_weights(subj_weights, train_subjects, i)
        backbone = base.BackboneClassifier(use_mdm=base.HAVE_PYRIEMANN)
        backbone.weights = base.BEST_BLEND.copy()
        backbone.fit(X_tr['backbone'], y_tr, covs=cov_tr, sample_weight=sw)
        parts = backbone.predict_parts(X_val['backbone'], covs=cov_val)
        p_trial = backbone.blend(parts)
        mod = base.TemporalModulator().fit(X_tr['zoneA'], X_tr['zoneB'], X_tr['zoneC'], y_tr, sample_weight=sw)
        mod.alphas = base.BEST_ALPHAS
        zone_probs = mod.predict_zone_probs(X_val['zoneA'], X_val['zoneB'], X_val['zoneC'])
        probs_2d = mod.build_timecourse(p_trial, zone_probs)
        yb = (y_val == 2).astype(int)
        try:
            tr_auc = base.roc_auc_score(yb, p_trial)
        except Exception:
            tr_auc = 0.5
        m = base.window_auc_score(probs_2d, yb)
        fold_scores.append(m['window_auc'])
        fold_trial_aucs.append(tr_auc)
    return float(np.mean(fold_scores)), float(np.mean(fold_trial_aucs))


def objective_factory(train_subjects, cache, csp_global):
    def objective(trial):
        params = sample_trial_params(trial)
        orig_b, orig_m = apply_params(params)
        try:
            fast_wauc, fast_tauc = evaluate_subset(train_subjects, cache, csp_global, FAST_FOLDS)
            # small penalty if trial AUC is too weak vs window AUC; keep ranking stable
            score = fast_wauc + 0.10 * (fast_tauc - 0.5)
            trial.report(score, step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()
            return score
        finally:
            restore_inits(orig_b, orig_m)
            gc.collect()
    return objective


def run_full_verify(train_subjects, cache, csp_global, best_params):
    orig_b, orig_m = apply_params(best_params)
    try:
        print('\nRunning FULL LOSO verify on best Optuna params...')
        fold_results, oof_trial_probs, oof_labels, subj_weights = base.run_loso_v104(train_subjects, cache, csp_global)
        trial_cal = base.fit_trial_platt(oof_trial_probs, oof_labels) if base.USE_TRIAL_PLATT else None
        return fold_results, oof_trial_probs, oof_labels, subj_weights, trial_cal
    finally:
        restore_inits(orig_b, orig_m)
        gc.collect()


def run_final_submission(train_subjects, test_subjects, cache, csp_global, best_params, subj_weights, trial_cal):
    orig_b, orig_m = apply_params(best_params)
    try:
        return base.generate_submission_v104(train_subjects, test_subjects, cache, csp_global, trial_cal=trial_cal, subj_weights=subj_weights, output_path=FINAL_OUTPUT)
    finally:
        restore_inits(orig_b, orig_m)
        gc.collect()


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║ EEG Emotional Memory — Ultra Pipeline v10.4b                           ║
╠══════════════════════════════════════════════════════════════════════════╣
║ Optuna wrapper فوق v10.4 winner line                                   ║
║ Fast fold search + full LOSO verify                                    ║
║ Tunes: LR/zone C values, k-best, LGBM helper, blend, alphas           ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

    total_start = time.time()
    print('STEP 1: Loading training data...')
    train_subjects = base.load_all_training(base.EMO_DIR, base.NEU_DIR)
    print('\nSTEP 2: Loading test data...')
    test_subjects = base.load_all_test(base.TEST_DIR)
    print('\nSTEP 3: Building cache...')
    cache = base.build_cache(train_subjects)
    print('\nSTEP 4: Fitting global CSP...')
    eeg_all = np.vstack([s['eeg'] for s in train_subjects])
    lbl_all = np.concatenate([s['labels'] for s in train_subjects])
    csp_global = base.CSP(n_components=6).fit(eeg_all, lbl_all)
    del eeg_all, lbl_all
    gc.collect()

    best_params = DEFAULT_PARAMS.copy()
    if RUN_OPTUNA:
        print('\nSTEP 5: Optuna FAST tuning...')
        storage = f'sqlite:///{STUDY_DB}'
        study = optuna.create_study(
            study_name=STUDY_NAME,
            storage=storage,
            direction='maximize',
            load_if_exists=True,
            sampler=TPESampler(seed=42, multivariate=True),
            pruner=MedianPruner(n_startup_trials=6, n_warmup_steps=0)
        )
        study.optimize(objective_factory(train_subjects, cache, csp_global), n_trials=OPTUNA_TRIALS, gc_after_trial=True)
        best_params.update(study.best_trial.params)
        print('\nBest Optuna trial:')
        print(f' value = {study.best_value:.6f}')
        print(f' params = {study.best_trial.params}')
    else:
        print('\nSTEP 5: Optuna skipped — using defaults')

    print('\nSTEP 6: Full LOSO verify with best params...')
    fold_results, oof_trial_probs, oof_labels, subj_weights, trial_cal = run_full_verify(train_subjects, cache, csp_global, best_params)
    base.plot_loso_results(fold_results, os.path.join(base.BASE, 'loso_results_v10_4b_optuna.png'))
    print(f"\n ► Mean Window-AUC = {np.mean([r['window_auc'] for r in fold_results]):.4f} ± {np.std([r['window_auc'] for r in fold_results]):.4f}")

    print('\nSTEP 7: Final training + submission...')
    df = run_final_submission(train_subjects, test_subjects, cache, csp_global, best_params, subj_weights, trial_cal)

    print('\nSTEP 8: Validation...')
    assert 'id' in df.columns and 'prediction' in df.columns
    assert df['prediction'].between(0,1).all()
    parts = str(df.iloc[0]['id']).split('_')
    assert len(parts) == 3
    print(f" ✓ Total rows: {len(df):,}")
    print(f" ✓ Pred range: [{df.prediction.min():.4f}, {df.prediction.max():.4f}]")
    print(f"\n✓ DONE ({(time.time()-total_start)/60:.1f} min total)")


if __name__ == '__main__':
    main()
