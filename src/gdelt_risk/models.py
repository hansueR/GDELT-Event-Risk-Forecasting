from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_classifier(model_type="logistic", random_seed=42, C=1.0):
    if model_type == "logistic":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        C=C,
                        penalty="l2",
                        solver="lbfgs",
                        random_state=random_seed,
                    ),
                ),
            ]
        )
    if model_type == "random_forest":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=6,
                        min_samples_leaf=10,
                        class_weight="balanced_subsample",
                        random_state=random_seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    if model_type == "hist_gradient_boosting":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=200,
                        learning_rate=0.05,
                        l2_regularization=1.0,
                        random_state=random_seed,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown model_type: {model_type}")


def predict_score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)
