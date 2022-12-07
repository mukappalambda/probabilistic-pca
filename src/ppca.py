"""
An Implementation of Probabilistic Principal Component Analysis
"""
import numpy as np
from scipy.stats import multivariate_normal


class PPCA:
    """
    Probabilistic PCA Model
    Ref: Bishop, C. M. & Tipping, M. E. (2001). Probabilistic Principal Component Analysis.
    """

    def __init__(self, latent_dim: int) -> None:
        self.latent_dim = latent_dim
        self.x = None
        self.n_samples_ = None
        self.n_features_ = None
        self.mu_mle_ = None
        self.sample_variance_ = None

    def __repr__(self):
        return (
            f"PPCA(n_samples={self.n_samples_};"
            f"n_features={self.n_features_};"
            f"latent_dim={self.latent_dim})"
        )

    def fit(self, x: np.ndarray):
        self.x = x
        self.n_samples_ = x.shape[0]
        self.n_features_ = x.shape[1]
        self.mu_mle_ = self.x.mean(axis=0)
        self.sample_variance_ = self.sample_variance()
        return self

    def sample_variance(self) -> np.ndarray:
        S = (1 / self.n_samples_) * (self.x - self.mu_mle_).T @ (self.x - self.mu_mle_)
        assert S.shape == (self.n_features_, self.n_features_)
        return S

    def eigen_decomposition(self):
        vals, vecs = np.linalg.eigh(self.sample_variance_)
        assert np.allclose(self.sample_variance_ @ vecs, vecs @ np.diag(vals))
        idx = vals.argsort()[::-1]
        vals, vecs = vals[idx], vecs[:, idx]
        assert np.allclose(self.sample_variance_ @ vecs, vecs @ np.diag(vals))
        return vals, vecs

    def model_params(self):
        vals, vecs = self.eigen_decomposition()
        variance_mle = (
            abs(1 / (self.n_features_ - self.latent_dim))
            * vals[-self.latent_dim :].sum()
        )
        W_mle = vecs[:, : self.latent_dim] @ np.diag(
            (vals[: self.latent_dim] - variance_mle) ** 0.5
        )
        C_mle = W_mle @ W_mle.T + variance_mle * np.identity(self.n_features_)
        return variance_mle, W_mle, C_mle

    def marginal(self):
        self.variance_mle, self.W_mle, self.C_mle = self.model_params()
        return multivariate_normal(
            mean=self.mu_mle_,
            cov=self.C_mle,
            allow_singular=True,
        )

    def sample(self, size: int):
        return self.marginal().rvs(size=size)
