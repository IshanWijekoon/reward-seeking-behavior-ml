"""Core modeling, nested CV, and XAI helpers for Objectives 2–4."""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
LABEL_ORDER = ["Low", "Moderate", "High"]
LABEL_TO_INT = {lab: i for i, lab in enumerate(LABEL_ORDER)}
INT_TO_LABEL = {i: lab for lab, i in LABEL_TO_INT.items()}

OUTER_FOLDS = 5
INNER_FOLDS = 5
N_ITER_SEARCH = 20


def resolve_project_root(cwd: Path | None = None) -> Path:
    cwd = (cwd or Path.cwd()).resolve()
    if (cwd / "Data" / "processed").exists():
        return cwd
    if (cwd.parent / "Data" / "processed").exists():
        return cwd.parent
    return cwd


def load_prepared(path: Path) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    df = pd.read_csv(path)
    if "risk_level" not in df.columns:
        raise ValueError(f"Missing risk_level in {path}")
    y = df["risk_level"].map(LABEL_TO_INT)
    if y.isna().any():
        bad = df.loc[y.isna(), "risk_level"].unique()
        raise ValueError(f"Unexpected risk_level values: {bad}")
    feature_cols = [c for c in df.columns if c != "risk_level"]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        X = X.fillna(X.median(numeric_only=True))
    return X, y.astype(int), feature_cols


def make_pipelines(random_state: int = RANDOM_SEED) -> dict[str, ImbPipeline]:
    """Build imblearn pipelines: scaler (when needed) -> SMOTE -> estimator."""
    lr = ImbPipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=random_state)),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=random_state,
                ),
            ),
        ]
    )
    svm = ImbPipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=random_state)),
            (
                "model",
                SVC(
                    kernel="rbf",
                    probability=True,
                    random_state=random_state,
                ),
            ),
        ]
    )
    rf = ImbPipeline(
        steps=[
            ("smote", SMOTE(random_state=random_state)),
            (
                "model",
                RandomForestClassifier(random_state=random_state, n_jobs=-1),
            ),
        ]
    )
    xgb = ImbPipeline(
        steps=[
            ("smote", SMOTE(random_state=random_state)),
            (
                "model",
                XGBClassifier(
                    objective="multi:softprob",
                    num_class=3,
                    eval_metric="mlogloss",
                    random_state=random_state,
                    n_jobs=-1,
                    tree_method="hist",
                ),
            ),
        ]
    )
    mlp = ImbPipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=random_state)),
            (
                "model",
                MLPClassifier(
                    random_state=random_state,
                    max_iter=400,
                    early_stopping=True,
                    validation_fraction=0.1,
                ),
            ),
        ]
    )
    return {
        "LogisticRegression": lr,
        "SVM": svm,
        "RandomForest": rf,
        "XGBoost": xgb,
        "NeuralNet": mlp,
    }


def param_distributions(random_state: int = RANDOM_SEED) -> dict[str, dict[str, Any]]:
    rng = np.random.RandomState(random_state)
    return {
        "LogisticRegression": {
            "model__C": np.logspace(-2, 2, 8),
        },
        "SVM": {
            "model__C": np.logspace(-1, 2, 6),
            "model__gamma": np.logspace(-3, 0, 6),
        },
        "RandomForest": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 8, 16, 24],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2"],
        },
        "XGBoost": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.03, 0.05, 0.1, 0.2],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
        },
        "NeuralNet": {
            "model__hidden_layer_sizes": [(64,), (128,), (128, 64)],
            "model__alpha": [1e-4, 1e-3, 1e-2],
            "model__learning_rate_init": [1e-3, 5e-4],
        },
    }


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }
    if y_proba is not None:
        try:
            out["roc_auc_ovr_macro"] = float(
                roc_auc_score(
                    y_true,
                    y_proba,
                    multi_class="ovr",
                    average="macro",
                )
            )
        except ValueError:
            out["roc_auc_ovr_macro"] = float("nan")
    return out


@dataclass
class NestedCVResult:
    model_name: str
    outer_f1_macro: list[float]
    outer_accuracy: list[float]
    outer_roc_auc: list[float]
    best_params_per_fold: list[dict[str, Any]]

    @property
    def summary(self) -> dict[str, float]:
        return {
            "model": self.model_name,
            "cv_f1_macro_mean": float(np.mean(self.outer_f1_macro)),
            "cv_f1_macro_std": float(np.std(self.outer_f1_macro)),
            "cv_accuracy_mean": float(np.mean(self.outer_accuracy)),
            "cv_accuracy_std": float(np.std(self.outer_accuracy)),
            "cv_roc_auc_mean": float(np.nanmean(self.outer_roc_auc)),
            "cv_roc_auc_std": float(np.nanstd(self.outer_roc_auc)),
        }


def run_nested_cv_for_model(
    name: str,
    pipeline: ImbPipeline,
    param_dist: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    outer_folds: int = OUTER_FOLDS,
    inner_folds: int = INNER_FOLDS,
    n_iter: int = N_ITER_SEARCH,
    random_state: int = RANDOM_SEED,
    n_jobs: int = -1,
    svm_max_outer_train: int = 2500,
) -> NestedCVResult:
    """Outer stratified CV with inner RandomizedSearchCV (nested)."""
    outer = StratifiedKFold(
        n_splits=outer_folds, shuffle=True, random_state=random_state
    )
    inner = StratifiedKFold(
        n_splits=inner_folds, shuffle=True, random_state=random_state
    )

    f1s, accs, aucs = [], [], []
    best_params: list[dict[str, Any]] = []

    X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    y_arr = y.values if isinstance(y, pd.Series) else np.asarray(y)

    for fold_i, (tr_idx, va_idx) in enumerate(outer.split(X_arr, y_arr), start=1):
        X_tr, X_va = X_arr[tr_idx], X_arr[va_idx]
        y_tr, y_va = y_arr[tr_idx], y_arr[va_idx]

        # RBF-SVM scales poorly; stratified subsample outer-train for tuning only
        if name == "SVM" and len(X_tr) > svm_max_outer_train:
            rng = np.random.RandomState(random_state + fold_i)
            # stratified subsample
            chosen = []
            for c in np.unique(y_tr):
                idx_c = np.where(y_tr == c)[0]
                n_c = int(round(svm_max_outer_train * (len(idx_c) / len(y_tr))))
                n_c = max(n_c, 1)
                pick = rng.choice(idx_c, size=min(n_c, len(idx_c)), replace=False)
                chosen.append(pick)
            sub = np.concatenate(chosen)
            X_tune, y_tune = X_tr[sub], y_tr[sub]
        else:
            X_tune, y_tune = X_tr, y_tr

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring="f1_macro",
            cv=inner,
            random_state=random_state + fold_i,
            n_jobs=n_jobs,
            refit=True,
            verbose=0,
        )
        search.fit(X_tune, y_tune)
        # Refit best params on full outer-train for unbiased outer-val estimate
        best_pipe = search.best_estimator_
        best_pipe.fit(X_tr, y_tr)
        pred = best_pipe.predict(X_va)
        proba = best_pipe.predict_proba(X_va)
        metrics = classification_metrics(y_va, pred, proba)
        f1s.append(metrics["f1_macro"])
        accs.append(metrics["accuracy"])
        aucs.append(metrics.get("roc_auc_ovr_macro", float("nan")))
        best_params.append(search.best_params_)
        print(
            f"  [{name}] outer fold {fold_i}/{outer_folds}: "
            f"F1_macro={metrics['f1_macro']:.4f} Acc={metrics['accuracy']:.4f}"
        )

    return NestedCVResult(
        model_name=name,
        outer_f1_macro=f1s,
        outer_accuracy=accs,
        outer_roc_auc=aucs,
        best_params_per_fold=best_params,
    )


def effective_n_iter(model_name: str, n_iter: int, n_samples: int) -> int:
    """Keep RBF-SVM tractable on larger datasets while retaining nested CV."""
    if model_name == "SVM" and n_samples > 1500:
        return min(n_iter, 8)
    return n_iter


def run_all_nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_iter: int = N_ITER_SEARCH,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, dict[str, NestedCVResult]]:
    pipes = make_pipelines(random_state)
    dists = param_distributions(random_state)
    rows = []
    details: dict[str, NestedCVResult] = {}
    n_samples = len(X)
    for name, pipe in pipes.items():
        model_iters = effective_n_iter(name, n_iter, n_samples)
        print(f"\n=== Nested CV: {name} (n_iter={model_iters}) ===")
        # SVM is expensive; keep full protocol but serialise to reduce memory pressure
        n_jobs = 1 if name == "SVM" else -1
        res = run_nested_cv_for_model(
            name,
            pipe,
            dists[name],
            X,
            y,
            n_iter=model_iters,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        details[name] = res
        rows.append(res.summary)
    summary = pd.DataFrame(rows).sort_values("cv_f1_macro_mean", ascending=False)
    return summary, details


def tune_and_fit_best(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_iter: int = N_ITER_SEARCH,
    random_state: int = RANDOM_SEED,
) -> RandomizedSearchCV:
    pipe = make_pipelines(random_state)[model_name]
    dist = param_distributions(random_state)[model_name]
    inner = StratifiedKFold(
        n_splits=INNER_FOLDS, shuffle=True, random_state=random_state
    )
    n_jobs = 1 if model_name == "SVM" else -1
    model_iters = effective_n_iter(model_name, n_iter, len(X_train))

    X_arr = X_train.values
    y_arr = y_train.values
    # Match nested-CV SVM tractability for final retune
    if model_name == "SVM" and len(X_arr) > 2500:
        rng = np.random.RandomState(random_state)
        chosen = []
        for c in np.unique(y_arr):
            idx_c = np.where(y_arr == c)[0]
            n_c = int(round(2500 * (len(idx_c) / len(y_arr))))
            n_c = max(n_c, 1)
            pick = rng.choice(idx_c, size=min(n_c, len(idx_c)), replace=False)
            chosen.append(pick)
        sub = np.concatenate(chosen)
        X_tune, y_tune = X_arr[sub], y_arr[sub]
    else:
        X_tune, y_tune = X_arr, y_arr

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=dist,
        n_iter=model_iters,
        scoring="f1_macro",
        cv=inner,
        random_state=random_state,
        n_jobs=n_jobs,
        refit=True,
        verbose=0,
    )
    search.fit(X_tune, y_tune)
    # Always refit winning pipeline on the full training set
    search.best_estimator_.fit(X_arr, y_arr)
    return search


def evaluate_holdout(
    estimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    pred = estimator.predict(X_test.values)
    proba = estimator.predict_proba(X_test.values)
    metrics = classification_metrics(y_test.values, pred, proba)
    return metrics, pred, proba


def plot_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    out_path: Path,
):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_ORDER,
        yticklabels=LABEL_ORDER,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()


def plot_roc_ovr(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str,
    out_path: Path,
):
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    plt.figure(figsize=(6.5, 5))
    for i, lab in enumerate(LABEL_ORDER):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        plt.plot(fpr, tpr, label=f"{lab} (AUC={auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()


def select_best_model(cv_summary: pd.DataFrame) -> str:
    """Primary: macro-F1; tie-break: ROC-AUC."""
    ranked = cv_summary.sort_values(
        ["cv_f1_macro_mean", "cv_roc_auc_mean"], ascending=False
    )
    return str(ranked.iloc[0]["model"])


def get_estimator(search: RandomizedSearchCV):
    return search.best_estimator_.named_steps["model"]


def transform_for_model(pipeline: ImbPipeline, X: np.ndarray) -> np.ndarray:
    """Apply scaler if present (not SMOTE) for explanation space."""
    if "scaler" in pipeline.named_steps:
        return pipeline.named_steps["scaler"].transform(X)
    return X


def run_shap(
    pipeline: ImbPipeline,
    model_name: str,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: list[str],
    out_prefix: Path,
    *,
    max_background: int = 100,
    max_explain: int = 200,
):
    """Global SHAP explanations; explainer chosen by model family."""
    rng = np.random.RandomState(RANDOM_SEED)
    bg_idx = rng.choice(
        len(X_background), size=min(max_background, len(X_background)), replace=False
    )
    ex_idx = rng.choice(
        len(X_explain), size=min(max_explain, len(X_explain)), replace=False
    )
    X_bg = X_background[bg_idx]
    X_ex = X_explain[ex_idx]

    model = pipeline.named_steps["model"]
    X_bg_t = transform_for_model(pipeline, X_bg)
    X_ex_t = transform_for_model(pipeline, X_ex)

    if model_name in {"RandomForest", "XGBoost"}:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_ex_t)
    elif model_name == "LogisticRegression":
        explainer = shap.LinearExplainer(model, X_bg_t)
        shap_values = explainer.shap_values(X_ex_t)
    else:
        # SVM / NeuralNet — KernelExplainer on predict_proba (class 2 / High as focus + multi)
        def f(data):
            # Inverse: model expects scaled space already for SVM/MLP
            return model.predict_proba(data)

        explainer = shap.KernelExplainer(f, X_bg_t)
        shap_values = explainer.shap_values(
            X_ex_t, nsamples=100
        )

    # shap_values may be list (per class) or array
    plt.figure()
    if isinstance(shap_values, list):
        # mean abs across classes for bar overview using class High (index 2) summary
        shap.summary_plot(
            shap_values[2],
            X_ex_t,
            feature_names=feature_names,
            show=False,
            plot_type="bar",
        )
        plt.title("SHAP mean |value| (class High)")
        plt.tight_layout()
        plt.savefig(Path(str(out_prefix) + "_shap_bar_high.png"), dpi=150, bbox_inches="tight")
        plt.show()

        plt.figure()
        shap.summary_plot(
            shap_values[2],
            X_ex_t,
            feature_names=feature_names,
            show=False,
        )
        plt.title("SHAP beeswarm (class High)")
        plt.tight_layout()
        plt.savefig(
            Path(str(out_prefix) + "_shap_beeswarm_high.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.show()
    else:
        # multioutput array (n_samples, n_features, n_classes) in newer shap
        if getattr(shap_values, "ndim", 0) == 3:
            sv_high = shap_values[:, :, 2]
        else:
            sv_high = shap_values
        shap.summary_plot(
            sv_high, X_ex_t, feature_names=feature_names, show=False, plot_type="bar"
        )
        plt.title("SHAP mean |value| (class High)")
        plt.tight_layout()
        plt.savefig(Path(str(out_prefix) + "_shap_bar_high.png"), dpi=150, bbox_inches="tight")
        plt.show()

        plt.figure()
        shap.summary_plot(sv_high, X_ex_t, feature_names=feature_names, show=False)
        plt.title("SHAP beeswarm (class High)")
        plt.tight_layout()
        plt.savefig(
            Path(str(out_prefix) + "_shap_beeswarm_high.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.show()

    return shap_values


def run_lime(
    pipeline: ImbPipeline,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    feature_names: list[str],
    out_dir: Path,
    dataset_tag: str,
):
    """Explain one correctly classified instance per class when possible."""
    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=LABEL_ORDER,
        discretize_continuous=True,
        mode="classification",
        random_state=RANDOM_SEED,
    )

    def predict_proba(data: np.ndarray) -> np.ndarray:
        return pipeline.predict_proba(data)

    for class_id, class_name in enumerate(LABEL_ORDER):
        matches = np.where((y_test == class_id) & (y_pred == class_id))[0]
        if len(matches) == 0:
            matches = np.where(y_test == class_id)[0]
        if len(matches) == 0:
            print(f"No instances for class {class_name}; skipping LIME.")
            continue
        idx = int(matches[0])
        exp = explainer.explain_instance(
            X_test[idx],
            predict_proba,
            num_features=min(10, len(feature_names)),
            top_labels=3,
        )
        fig = exp.as_pyplot_figure(label=class_id)
        fig.suptitle(f"LIME — {dataset_tag} — true/pred={class_name} (idx={idx})")
        out_path = out_dir / f"{dataset_tag}_lime_{class_name.lower()}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Saved LIME explanation: {out_path.name}")


def save_json(obj: dict, path: Path):
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
