import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import linalg
from sklearn import cluster

EPS = np.finfo(float).eps

def log_multivariate_normal_density(X, means, covars, covariance_type='diag'):
    log_multivariate_normal_density_dict = {
        'spherical': _log_multivariate_normal_density_spherical,
        'tied': _log_multivariate_normal_density_tied,
        'diag': _log_multivariate_normal_density_diag,
        'full': _log_multivariate_normal_density_full}
    return log_multivariate_normal_density_dict[covariance_type](X, means, covars)

def distribute_covar_matrix_to_match_covariance_type(tied_cv, covariance_type, n_components):
    if covariance_type == 'spherical':
        cv = np.tile(tied_cv.mean() * np.ones(tied_cv.shape[1]), (n_components, 1))
    elif covariance_type == 'tied':
        cv = tied_cv
    elif covariance_type == 'diag':
        cv = np.tile(np.diag(tied_cv), (n_components, 1))
    elif covariance_type == 'full':
        cv = np.tile(tied_cv, (n_components, 1, 1))
    else:
        raise ValueError("covariance_type must be one of 'spherical', 'tied', 'diag', 'full'")
    return cv

def _covar_mstep_spherical(*args):
    cv = _covar_mstep_diag(*args)
    return np.tile(cv.mean(axis=1)[:, np.newaxis], (1, cv.shape[1]))

def _covar_mstep_diag(gmm, X, responsibilities, weighted_X_sum, norm, min_covar):
    avg_X2 = np.dot(responsibilities.T, X * X) * norm
    avg_means2 = gmm.means_ ** 2
    avg_X_means = gmm.means_ * weighted_X_sum * norm
    return avg_X2 - 2 * avg_X_means + avg_means2 + min_covar

def _covar_mstep_tied(gmm, X, responsibilities, weighted_X_sum, norm, min_covar):
    n_features = X.shape[1]
    avg_X2 = np.dot(X.T, X)
    avg_means2 = np.dot(gmm.means_.T, weighted_X_sum)
    return (avg_X2 - avg_means2 + min_covar * np.eye(n_features)) / X.shape[0]

def _covar_mstep_full(gmm, X, responsibilities, weighted_X_sum, norm, min_covar):
    n_features = X.shape[1]
    cv = np.empty((gmm.n_components, n_features, n_features))
    for c in range(gmm.n_components):
        post = responsibilities[:, c]
        np.seterr(under='ignore')
        avg_cv = np.dot(post * X.T, X) / (post.sum() + 10 * EPS)
        mu = gmm.means_[c][np.newaxis]
        cv[c] = (avg_cv - np.dot(mu.T, mu) + min_covar * np.eye(n_features))
    return cv

_covar_mstep_funcs = {'spherical': _covar_mstep_spherical,
                      'diag': _covar_mstep_diag,
                      'tied': _covar_mstep_tied,
                      'full': _covar_mstep_full}

def _log_multivariate_normal_density_spherical(X, means, covars):
    cv = covars.copy()
    if covars.ndim == 1:
        cv = cv[:, np.newaxis]
    if covars.shape[1] == 1:
        cv = np.tile(cv, (1, X.shape[-1]))
    return _log_multivariate_normal_density_diag(X, means, cv)

def _log_multivariate_normal_density_tied(X, means, covars):
    n_samples, n_dim = X.shape
    icv = pinvh(covars)
    lpr = -0.5 * (n_dim * np.log(2 * np.pi) + np.log(linalg.det(covars) + 0.1) + np.sum(X * np.dot(X, icv), 1)[:, np.newaxis] - 2 * np.dot(np.dot(X, icv), means.T) + np.sum(means * np.dot(means, icv), 1))
    return lpr

def _log_multivariate_normal_density_diag(X, means, covars):
    n_samples, n_dim = X.shape
    lpr = -0.5 * (n_dim * np.log(2 * np.pi) + np.sum(np.log(covars), 1) + np.sum((means ** 2) / covars, 1) - 2 * np.dot(X, (means / covars).T) + np.dot(X ** 2, (1.0 / covars).T))
    return lpr

def _log_multivariate_normal_density_full(X, means, covars, min_covar=1.e-7):
    n_samples, n_dim = X.shape
    nmix = len(means)
    log_prob = np.empty((n_samples, nmix))
    for c, (mu, cv) in enumerate(zip(means, covars)):
        try:
            cv_chol = linalg.cholesky(cv, lower=True)
        except linalg.LinAlgError:
            cv_chol = linalg.cholesky(cv + min_covar * np.eye(n_dim), lower=True)
        cv_log_det = 2 * np.sum(np.log(np.diagonal(cv_chol)))
        cv_sol = linalg.solve_triangular(cv_chol, (X - mu).T, lower=True).T
        log_prob[:, c] = - .5 * (np.sum(cv_sol ** 2, axis=1) + n_dim * np.log(2 * np.pi) + cv_log_det)
    return log_prob

def logsumexp(arr, axis=0):
    arr = np.rollaxis(arr, axis)
    vmax = arr.max(axis=0)
    out = np.log(np.sum(np.exp(arr - vmax), axis=0))
    out += vmax
    return out

def pinvh(a, cond=None, rcond=None, lower=True):
    a = np.asarray_chkfinite(a)
    s, u = linalg.eigh(a, lower=lower)
    if rcond is not None:
        cond = rcond
    if cond in [None, -1]:
        t = u.dtype.char.lower()
        factor = {'f': 1E3, 'd': 1E6}
        cond = factor[t] * np.finfo(t).eps
    above_cutoff = (abs(s) > cond * np.max(abs(s)))
    psigma_diag = np.zeros_like(s)
    psigma_diag[above_cutoff] = 1.0 / s[above_cutoff]
    return np.dot(u * psigma_diag, np.conjugate(u).T)

class GMM():
    def __init__(self, n_components=1, covariance_type='diag', thresh=1e-2, min_covar=1e-3, n_iter=100, n_init=1, params='wmc', init_params='wmc'):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.thresh = thresh
        self.min_covar = min_covar
        self.n_iter = n_iter
        self.n_init = n_init
        self.params = params
        self.init_params = init_params
        if not covariance_type in ['spherical', 'tied', 'diag', 'full']: 
            raise ValueError('Invalid value for covariance_type: %s' % covariance_type)
        if n_init < 1:
            raise ValueError('GMM estimation requires at least one run')
        self.weights_ = np.ones(self.n_components) / self.n_components
        self.converged_ = False

    def _get_covars(self):
        if self.covariance_type == 'full':
            return self.covars_
        elif self.covariance_type == 'diag':
            return [np.diag(cov) for cov in self.covars_]
        elif self.covariance_type == 'tied':
            return [self.covars_] * self.n_components
        elif self.covariance_type == 'spherical':
            return [np.diag(cov) for cov in self.covars_]

    def score_samples(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        if X.size == 0:
            return np.array([]), np.empty((0, self.n_components))
        if X.shape[1] != self.means_.shape[1]:
            raise ValueError('The shape of X  is not compatible with self')
        lpr = (log_multivariate_normal_density(X, self.means_, self.covars_, self.covariance_type) + np.log(self.weights_))
        logprob = logsumexp(lpr, axis=1)
        responsibilities = np.exp(lpr - logprob[:, np.newaxis])
        return logprob, responsibilities

    def score(self, X):
        logprob, _ = self.score_samples(X)
        return logprob

    def predict(self, X):
        logprob, responsibilities = self.score_samples(X)
        return responsibilities.argmax(axis=1)

    def predict_proba(self, X):
        logprob, responsibilities = self.score_samples(X)
        return responsibilities

    def fit(self, X):
        X = np.asarray(X, dtype=np.float)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        if X.shape[0] < self.n_components:
            raise ValueError('GMM estimation with %s components, but got only %s samples' % (self.n_components, X.shape[0]))
        max_log_prob = -np.infty
        print self.init_params
        for _ in range(self.n_init):
            if 'm' in self.init_params or not hasattr(self, 'means_'):
                self.means_ = cluster.KMeans(n_clusters=self.n_components).fit(X).cluster_centers_
            if 'w' in self.init_params or not hasattr(self, 'weights_'):
                self.weights_ = np.tile(1.0 / self.n_components, self.n_components)
            if 'c' in self.init_params or not hasattr(self, 'covars_'):
                cv = np.cov(X.T) + self.min_covar * np.eye(X.shape[1])
                if not cv.shape:
                    cv.shape = (1, 1)
                self.covars_ = distribute_covar_matrix_to_match_covariance_type(cv, self.covariance_type, self.n_components)
            # EM algorithms
            log_likelihood = []
            self.converged_ = False
            for i in range(self.n_iter):
                # Expectation step
                curr_log_likelihood, responsibilities = self.score_samples(X)
                log_likelihood.append(curr_log_likelihood.sum())
                # Check for convergence
                if i > 0 and abs(log_likelihood[-1] - log_likelihood[-2]) < self.thresh:
                    self.converged_ = True
                    break
                # Maximization step
                self._do_mstep(X, responsibilities, self.params, self.min_covar)
            # if the results are better, keep it
            if self.n_iter:
                if log_likelihood[-1] > max_log_prob:
                    max_log_prob = log_likelihood[-1]
                    best_params = {'weights': self.weights_, 'means': self.means_, 'covars': self.covars_}
        # check the existence of an init param that was not subject to likelihood computation issue
        if np.isneginf(max_log_prob) and self.n_iter:
            raise RuntimeError("EM algorithm was never able to compute a valid likelihood given initial parameters. Try different init parameters (or increasing n_init) or check for degenerate data.")
        if self.n_iter:
            self.covars_ = best_params['covars']
            self.means_ = best_params['means']
            self.weights_ = best_params['weights']
        return self

    def _do_mstep(self, X, responsibilities, params, min_covar=0):
        weights = responsibilities.sum(axis=0)
        weighted_X_sum = np.dot(responsibilities.T, X)
        inverse_weights = 1.0 / (weights[:, np.newaxis] + 10 * EPS)
        if 'w' in params:
            self.weights_ = (weights / (weights.sum() + 10 * EPS) + EPS)
        if 'm' in params:
            self.means_ = weighted_X_sum * inverse_weights
        if 'c' in params:
            covar_mstep_func = _covar_mstep_funcs[self.covariance_type]
            self.covars_ = covar_mstep_func(
                self, X, responsibilities, weighted_X_sum, inverse_weights,
                min_covar)
        return weights

    def _n_parameters(self):
        ndim = self.means_.shape[1]
        if self.covariance_type == 'full':
            cov_params = self.n_components * ndim * (ndim + 1) / 2.
        elif self.covariance_type == 'diag':
            cov_params = self.n_components * ndim
        elif self.covariance_type == 'tied':
            cov_params = ndim * (ndim + 1) / 2.
        elif self.covariance_type == 'spherical':
            cov_params = self.n_components
        mean_params = ndim * self.n_components
        return int(cov_params + mean_params + self.n_components - 1)

    def bic(self, X):
        return (-2 * self.score(X).sum() + self._n_parameters() * np.log(X.shape[0]))

    def aic(self, X):
        return - 2 * self.score(X).sum() + 2 * self._n_parameters()

n_samples = 500
np.random.seed(0)
C = np.array([[0., -0.1], [1.7, .4]])
X = np.r_[np.dot(np.random.randn(n_samples, 2), C), .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]
gmm = GMM(n_components=2, covariance_type='spherical')
gmm.fit(X)
clf, title = gmm, 'GMM'
splot = plt.subplot(1, 1, 1)
Y_ = clf.predict(X)
for i, (mean, covar, color) in enumerate(zip(clf.means_, clf._get_covars(), itertools.cycle(['r', 'g']))):
    v, w = linalg.eigh(covar)
    u = w[0] / linalg.norm(w[0])
    plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)
    angle = np.arctan(u[1] / u[0])
    angle = 180 * angle / np.pi
    ell = mpl.patches.Ellipse(mean, v[0]*3, v[1]*3, 180 + angle, color=color)
    ell.set_clip_box(splot.bbox)
    ell.set_alpha(0.5)
    splot.add_artist(ell)
plt.xlim(-10, 10)
plt.ylim(-3, 6)
plt.xticks(())
plt.yticks(())
plt.title(title)
plt.show()
