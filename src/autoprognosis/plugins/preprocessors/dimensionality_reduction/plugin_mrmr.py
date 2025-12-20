# stdlib
from typing import Any, List, Optional

# third party
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif

# autoprognosis absolute
import autoprognosis.plugins.preprocessors.base as base
import autoprognosis.plugins.core.params as params
import autoprognosis.utils.serialization as serialization


class MRMRSelector(BaseEstimator, TransformerMixin):
    """Minimal redundancy maximal relevance (mRMR) feature selector.

    Relevance is estimated with `mutual_info_classif` while redundancy is computed
    as the average absolute Pearson correlation with the already selected features.
    """

    def __init__(self, n_features: int = 4, random_state: int = 0) -> None:
        if n_features < 1:
            raise ValueError("n_features must be >= 1")

        self.n_features = n_features
        self.random_state = random_state

        self.selected_features_: List[str] = []
        self.feature_names_: List[str] = []
        self.relevance_: Optional[pd.Series] = None
        self.corr_matrix_: Optional[pd.DataFrame] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MRMRSelector":
        df = pd.DataFrame(X).reset_index(drop=True)
        y_series = pd.Series(y).reset_index(drop=True)

        self.feature_names_ = list(df.columns)
        k = min(self.n_features, len(self.feature_names_))

        relevance = mutual_info_classif(
            df, y_series, random_state=self.random_state, discrete_features="auto"
        )
        relevance = pd.Series(relevance, index=self.feature_names_)

        corr = df.corr(method="pearson").abs().fillna(0.0)

        selected: List[str] = []
        candidates = self.feature_names_.copy()

        for _ in range(k):
            best_feature = None
            best_score = -np.inf

            for feat in candidates:
                relevance_score = relevance[feat]
                redundancy_penalty = 0.0
                if selected:
                    redundancy_penalty = corr.loc[feat, selected].mean()
                    if np.isnan(redundancy_penalty):
                        redundancy_penalty = 0.0

                score = relevance_score - redundancy_penalty

                if score > best_score:
                    best_score = score
                    best_feature = feat

            if best_feature is None:
                break

            selected.append(best_feature)
            candidates.remove(best_feature)

        self.selected_features_ = selected
        self.relevance_ = relevance
        self.corr_matrix_ = corr

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_features_:
            raise RuntimeError("MRMRSelector was not fitted.")

        df = pd.DataFrame(X)
        missing = [col for col in self.selected_features_ if col not in df.columns]
        if missing:
            raise RuntimeError(f"Missing columns in transform: {missing}")

        return df[self.selected_features_].copy()

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series, **kwargs: Any
    ) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def get_support(self) -> List[bool]:
        if not self.feature_names_:
            raise RuntimeError("MRMRSelector was not fitted.")

        support = [feat in self.selected_features_ for feat in self.feature_names_]
        return support


class MRMRPlugin(base.PreprocessorPlugin):
    """AutoPrognosis plugin wrapper for the mRMR selector."""

    def __init__(self, random_state: int = 0, n_features: int = 4) -> None:
        super().__init__()
        self.random_state = random_state
        self.n_features = n_features
        self.model: Optional[MRMRSelector] = None
        self.selected_features_: List[str] = []

    @staticmethod
    def name() -> str:
        return "mrmr"

    @staticmethod
    def subtype() -> str:
        return "dimensionality_reduction"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        # Expose n_features so AutoPrognosis can tune k.
        # Bounds use the generic components interval helper (1 .. features_count).
        cmin, cmax = base.PreprocessorPlugin.components_interval(*args, **kwargs)
        return [params.Integer("n_features", cmin, cmax)]

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> "MRMRPlugin":
        self.model = MRMRSelector(
            n_features=min(self.n_features, X.shape[1]),
            random_state=self.random_state,
        )
        self.model.fit(X, y)
        self.selected_features_ = getattr(self.model, "selected_features_", [])

        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("mRMR plugin must be fitted before calling transform.")

        return self.model.transform(X)

    def save(self) -> bytes:
        return serialization.save_model(
            {
                "random_state": self.random_state,
                "n_features": self.n_features,
                "model": self.model,
                "selected_features_": self.selected_features_,
            }
        )

    @classmethod
    def load(cls, buff: bytes) -> "MRMRPlugin":
        args = serialization.load_model(buff)
        selected_features = args.get("selected_features_", [])
        obj = cls(
            random_state=args.get("random_state", 0),
            n_features=args.get("n_features", 4),
        )
        obj.model = args.get("model")
        obj.selected_features_ = selected_features
        return obj


plugin = MRMRPlugin
