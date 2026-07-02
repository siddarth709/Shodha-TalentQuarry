import warnings

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class DenseEmbedder:
    def __init__(self, n_components: int = 128, max_features: int = 20000):
        self.n_components = n_components
        self.max_features = max_features
        self.vectorizer = None
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.embeddings_ = None
        self._fitted = False

    def fit(self, texts: list[str]) -> "DenseEmbedder":
        min_df = 2 if len(texts) >= 20 else 1
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=min_df,
            sublinear_tf=True,
        )
        tfidf = self.vectorizer.fit_transform(texts)
        max_safe = min(self.n_components, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
        if max_safe < 2:
            self.svd = None
            self.embeddings_ = normalize(tfidf.toarray())
        else:
            if max_safe < self.n_components:
                self.svd = TruncatedSVD(n_components=max_safe, random_state=42)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=RuntimeWarning,
                )
                reduced = self.svd.fit_transform(tfidf)
            self.embeddings_ = normalize(reduced)
        self._fitted = True
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("DenseEmbedder.fit() must be called before transform()")
        tfidf = self.vectorizer.transform(texts)
        if self.svd is None:
            return normalize(tfidf.toarray())
        reduced = self.svd.transform(tfidf)
        return normalize(reduced)


def candidate_text(candidate: dict) -> str:
    parts = []
    for role in candidate.get("career_history", []):
        desc = role.get("description") or ""
        title = role.get("title") or ""
        parts.append(f"{title} {desc} {desc}")
    skills = candidate.get("skills", [])
    skill_names = " ".join(s.get("name") or " " for s in skills)
    parts.append(skill_names)

    profile = candidate.get("profile", {})
    parts.append(profile.get("headline") or "")
    parts.append(profile.get("summary") or "")
    return " ".join(p for p in parts if p)
