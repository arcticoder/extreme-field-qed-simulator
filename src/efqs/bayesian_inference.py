"""
Bayesian inference for anomalous coupling constraints.

This module implements posterior distribution sampling for κ-parameters
given observational data (or null results). Instead of point estimates,
we compute full posterior P(κ|data, model) via:

    P(κ|data) ∝ P(data|κ) × P(κ)

Key capabilities:
1. Likelihood functions for strain measurements with detector noise
2. Prior distributions (log-uniform, physical motivated)
3. MCMC sampling (emcee) for posterior exploration
4. Credible interval computation (68%, 95%, 99.7%)
5. Discovery reach curves: κ_95% vs. experimental parameters
6. Null-result upper limits with proper Bayesian treatment

References:
- Gregory, "Bayesian Logical Data Analysis for the Physical Sciences" (2005)
- Trotta, "Bayes in the sky", Contemp. Phys. 49, 71 (2008)
- Abbott et al. (LIGO), "GW150914 parameter estimation", Phys. Rev. Lett. 116, 241102 (2016)
"""

import numpy as np
from typing import Optional, Tuple, Dict, Callable, List
from dataclasses import dataclass
import warnings

# Optional imports (will gracefully degrade if not available)
try:
    import emcee
    HAS_EMCEE = True
except ImportError:
    HAS_EMCEE = False
    warnings.warn("emcee not installed; MCMC sampling unavailable. Install: pip install emcee")

try:
    from scipy.stats import norm, uniform
    from scipy.integrate import quad
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not installed; some statistical functions unavailable")


@dataclass
class PosteriorSample:
    """Container for MCMC posterior samples."""
    samples: np.ndarray  # Shape: (n_samples, n_params)
    log_prob: np.ndarray  # Log posterior values
    acceptance_fraction: float
    param_names: List[str]
    
    def percentile(self, param_idx: int, percentiles: List[float]) -> np.ndarray:
        """Compute percentiles for a parameter."""
        return np.percentile(self.samples[:, param_idx], percentiles)
    
    def credible_interval(self, param_idx: int, level: float = 0.95) -> Tuple[float, float]:
        """
        Compute equal-tailed credible interval.
        
        Args:
            param_idx: Parameter index
            level: Credible level (0.95 = 95%)
        
        Returns:
            (lower, upper) bounds
        """
        alpha = 1 - level
        lower = self.percentile(param_idx, [100 * alpha / 2])[0]
        upper = self.percentile(param_idx, [100 * (1 - alpha / 2)])[0]
        return (lower, upper)
    
    def median(self, param_idx: int) -> float:
        """Median value (50th percentile)."""
        return self.percentile(param_idx, [50])[0]
    
    def summary(self, param_idx: int = 0) -> Dict[str, float]:
        """Statistical summary for a parameter."""
        samples = self.samples[:, param_idx]
        return {
            'median': float(np.median(samples)),
            'mean': float(np.mean(samples)),
            'std': float(np.std(samples)),
            'ci_68': self.credible_interval(param_idx, 0.68),
            'ci_95': self.credible_interval(param_idx, 0.95),
            'ci_997': self.credible_interval(param_idx, 0.997),
        }


def log_likelihood_gaussian(
    h_obs: float,
    h_pred_func: Callable[[float], float],
    kappa: float,
    sigma_noise: float
) -> float:
    """
    Gaussian likelihood for strain measurement.
    
    Args:
        h_obs: Observed strain (could be 0 for null result)
        h_pred_func: Function kappa -> h_predicted
        kappa: Coupling parameter value
        sigma_noise: RMS noise level (from detector ASD integrated over bandwidth)
    
    Returns:
        Log-likelihood ln P(h_obs | kappa)
    
    Model: h_obs = h_pred(kappa) + noise, noise ~ N(0, sigma_noise^2)
    """
    h_pred = h_pred_func(kappa)
    residual = h_obs - h_pred
    return -0.5 * (residual / sigma_noise)**2 - np.log(sigma_noise * np.sqrt(2 * np.pi))


def log_likelihood_upper_limit(
    h_upper: float,
    h_pred_func: Callable[[float], float],
    kappa: float
) -> float:
    """
    Likelihood for upper-limit constraint (no detection).
    
    Args:
        h_upper: 95% CL upper limit on strain
        h_pred_func: Function kappa -> h_predicted
        kappa: Coupling parameter value
    
    Returns:
        Log-likelihood
    
    Model: If h_pred(kappa) > h_upper, likelihood = 0 (ruled out)
           If h_pred(kappa) <= h_upper, likelihood = uniform (consistent)
    """
    h_pred = h_pred_func(kappa)
    if h_pred > h_upper:
        return -np.inf  # Ruled out
    else:
        return 0.0  # Uniform within allowed region


def log_prior_log_uniform(kappa: float, kappa_min: float, kappa_max: float) -> float:
    """
    Log-uniform (Jeffreys) prior: P(κ) ∝ 1/κ over [kappa_min, kappa_max].
    
    Appropriate when κ spans many orders of magnitude and we have no
    preferred scale a priori.
    
    Args:
        kappa: Parameter value
        kappa_min, kappa_max: Prior bounds
    
    Returns:
        Log-prior ln P(κ)
    """
    if kappa_min <= kappa <= kappa_max:
        return -np.log(kappa) - np.log(np.log(kappa_max / kappa_min))
    else:
        return -np.inf


def log_prior_uniform(kappa: float, kappa_min: float, kappa_max: float) -> float:
    """
    Uniform prior: P(κ) = const over [kappa_min, kappa_max].
    
    Args:
        kappa: Parameter value
        kappa_min, kappa_max: Prior bounds
    
    Returns:
        Log-prior ln P(κ)
    """
    if kappa_min <= kappa <= kappa_max:
        return -np.log(kappa_max - kappa_min)
    else:
        return -np.inf


def log_posterior(
    kappa: float,
    h_obs: float,
    h_pred_func: Callable[[float], float],
    sigma_noise: float,
    prior_type: str = 'log_uniform',
    kappa_min: float = 1e-50,
    kappa_max: float = 1e50
) -> float:
    """
    Log-posterior: ln P(κ|data) = ln P(data|κ) + ln P(κ) + const.
    
    Args:
        kappa: Coupling parameter
        h_obs: Observed strain
        h_pred_func: Prediction function κ -> h
        sigma_noise: Noise RMS
        prior_type: 'log_uniform' or 'uniform'
        kappa_min, kappa_max: Prior bounds
    
    Returns:
        Log-posterior value
    """
    # Prior
    if prior_type == 'log_uniform':
        lp = log_prior_log_uniform(kappa, kappa_min, kappa_max)
    elif prior_type == 'uniform':
        lp = log_prior_uniform(kappa, kappa_min, kappa_max)
    else:
        raise ValueError(f"Unknown prior_type: {prior_type}")
    
    if not np.isfinite(lp):
        return -np.inf
    
    # Likelihood
    ll = log_likelihood_gaussian(h_obs, h_pred_func, kappa, sigma_noise)
    
    return lp + ll


def sample_posterior_mcmc(
    h_obs: float,
    h_pred_func: Callable[[float], float],
    sigma_noise: float,
    prior_type: str = 'log_uniform',
    kappa_min: float = 1e-50,
    kappa_max: float = 1e50,
    n_walkers: int = 32,
    n_steps: int = 5000,
    n_burn: int = 1000,
    initial_kappa: Optional[float] = None
) -> PosteriorSample:
    """
    Sample posterior distribution using MCMC (emcee).
    
    Args:
        h_obs: Observed strain (0 for null result)
        h_pred_func: Function κ -> h_predicted
        sigma_noise: Detector noise RMS
        prior_type: 'log_uniform' or 'uniform'
        kappa_min, kappa_max: Prior bounds
        n_walkers: Number of MCMC walkers (recommend ≥ 2×n_params)
        n_steps: Steps per walker
        n_burn: Burn-in steps to discard
        initial_kappa: Starting guess (None = geometric mean of bounds)
    
    Returns:
        PosteriorSample object with samples and statistics
    """
    if not HAS_EMCEE:
        raise ImportError("emcee required for MCMC sampling. Install: pip install emcee")
    
    # Initial positions for walkers
    if initial_kappa is None:
        if prior_type == 'log_uniform':
            initial_kappa = np.sqrt(kappa_min * kappa_max)  # Geometric mean
        else:
            initial_kappa = 0.5 * (kappa_min + kappa_max)  # Arithmetic mean
    
    # Scatter walkers around initial guess (in log-space for log-uniform prior)
    if prior_type == 'log_uniform':
        log_kappa_init = np.log10(initial_kappa)
        log_kappa_std = 0.1 * (np.log10(kappa_max) - np.log10(kappa_min))
        p0 = 10**(log_kappa_init + log_kappa_std * np.random.randn(n_walkers, 1))
    else:
        kappa_std = 0.1 * (kappa_max - kappa_min)
        p0 = initial_kappa + kappa_std * np.random.randn(n_walkers, 1)
    
    # Clip to prior bounds
    p0 = np.clip(p0, kappa_min, kappa_max)
    
    # Define log-probability function for emcee
    def log_prob_emcee(theta):
        kappa = theta[0]
        return log_posterior(kappa, h_obs, h_pred_func, sigma_noise,
                           prior_type, kappa_min, kappa_max)
    
    # Run MCMC
    sampler = emcee.EnsembleSampler(n_walkers, 1, log_prob_emcee)
    sampler.run_mcmc(p0, n_steps, progress=False)
    
    # Extract samples (discard burn-in)
    samples = sampler.get_chain(discard=n_burn, flat=True)  # Shape: (n_samples, 1)
    log_prob = sampler.get_log_prob(discard=n_burn, flat=True)
    
    acceptance = np.mean(sampler.acceptance_fraction)
    
    return PosteriorSample(
        samples=samples,
        log_prob=log_prob,
        acceptance_fraction=acceptance,
        param_names=['kappa']
    )


def null_result_upper_limit(
    h_pred_func: Callable[[float], float],
    h_sensitivity: float,
    confidence_level: float = 0.95,
    prior_type: str = 'log_uniform',
    kappa_min: float = 1e-50,
    kappa_max: float = 1e50,
    n_samples: int = 10000
) -> float:
    """
    Bayesian upper limit on κ from null result (no detection).
    
    Given no detection with sensitivity h_sensitivity, compute κ_upper such that:
        P(κ < κ_upper | no detection) = confidence_level
    
    Args:
        h_pred_func: Function κ -> h_predicted
        h_sensitivity: 5σ sensitivity of detector
        confidence_level: Desired CL (0.95 = 95%)
        prior_type: 'log_uniform' or 'uniform'
        kappa_min, kappa_max: Prior bounds
        n_samples: Number of samples for numerical integration
    
    Returns:
        κ_upper: Upper limit on coupling
    
    Method: Compute posterior P(κ|no detection) ∝ P(no detection|κ) × P(κ)
            where P(no detection|κ) = 1 if h_pred(κ) < h_sensitivity, else 0
    """
    # Sample κ values from prior
    if prior_type == 'log_uniform':
        # Log-uniform sampling
        log_kappa_samples = np.random.uniform(
            np.log10(kappa_min), np.log10(kappa_max), n_samples
        )
        kappa_samples = 10**log_kappa_samples
        # Prior weight: P(κ) ∝ 1/κ
        prior_weights = 1.0 / kappa_samples
    else:
        # Uniform sampling
        kappa_samples = np.random.uniform(kappa_min, kappa_max, n_samples)
        prior_weights = np.ones(n_samples)
    
    # Likelihood: 1 if consistent with non-detection, 0 if ruled out
    h_pred = np.array([h_pred_func(k) for k in kappa_samples])
    likelihood = (h_pred < h_sensitivity).astype(float)
    
    # Posterior weights (unnormalized)
    posterior_weights = likelihood * prior_weights
    
    # Normalize
    posterior_weights /= np.sum(posterior_weights)
    
    # Sort by κ and compute cumulative distribution
    sort_idx = np.argsort(kappa_samples)
    kappa_sorted = kappa_samples[sort_idx]
    cdf = np.cumsum(posterior_weights[sort_idx])
    
    # Find κ where CDF = confidence_level
    idx_upper = np.searchsorted(cdf, confidence_level)
    if idx_upper >= len(kappa_sorted):
        return kappa_max  # Upper limit exceeds prior range
    else:
        return kappa_sorted[idx_upper]


def discovery_reach_curve(
    param_values: np.ndarray,
    h_pred_funcs: List[Callable[[float], float]],
    h_sensitivity: float,
    confidence_level: float = 0.95,
    prior_type: str = 'log_uniform',
    kappa_min: float = 1e-50,
    kappa_max: float = 1e50
) -> np.ndarray:
    """
    Compute discovery reach: κ_upper vs. experimental parameter.
    
    Args:
        param_values: Array of parameter values (e.g., E₀, waist, Q)
        h_pred_funcs: List of prediction functions (one per param_value)
        h_sensitivity: Detector sensitivity (assumed constant)
        confidence_level: Upper limit CL
        prior_type: Prior distribution type
        kappa_min, kappa_max: Prior bounds
    
    Returns:
        Array of κ_upper values (same length as param_values)
    
    Example:
        E0_values = np.logspace(14, 16, 20)
        h_funcs = [lambda k: compute_strain(E0=E0, kappa=k) for E0 in E0_values]
        kappa_reach = discovery_reach_curve(E0_values, h_funcs, h_sens=1e-22)
    """
    kappa_upper = []
    
    for h_func in h_pred_funcs:
        k_up = null_result_upper_limit(
            h_func, h_sensitivity, confidence_level,
            prior_type, kappa_min, kappa_max
        )
        kappa_upper.append(k_up)
    
    return np.array(kappa_upper)


def evidence_ratio(
    h_obs: float,
    h_pred_func_1: Callable[[float], float],
    h_pred_func_2: Callable[[float], float],
    sigma_noise: float,
    prior_type: str = 'log_uniform',
    kappa_min: float = 1e-50,
    kappa_max: float = 1e50,
    n_samples: int = 10000
) -> float:
    """
    Compute Bayes factor (evidence ratio) for model comparison.
    
    Compares two models with different h_pred functions (e.g., different ansätze).
    
    Args:
        h_obs: Observed strain
        h_pred_func_1, h_pred_func_2: Prediction functions for models 1 and 2
        sigma_noise: Detector noise
        prior_type, kappa_min, kappa_max: Prior specification
        n_samples: Monte Carlo samples for evidence integration
    
    Returns:
        Bayes factor B_12 = P(data|model1) / P(data|model2)
        B > 1: model 1 preferred
        B < 1: model 2 preferred
        |ln B| < 1: inconclusive
    
    Reference: Kass & Raftery, J. Am. Stat. Assoc. 90, 773 (1995)
    """
    if not HAS_SCIPY:
        raise ImportError("scipy required for evidence calculation")
    
    # Sample κ from prior
    if prior_type == 'log_uniform':
        log_kappa = np.random.uniform(np.log10(kappa_min), np.log10(kappa_max), n_samples)
        kappa_samples = 10**log_kappa
    else:
        kappa_samples = np.random.uniform(kappa_min, kappa_max, n_samples)
    
    # Compute likelihoods
    log_like_1 = np.array([
        log_likelihood_gaussian(h_obs, h_pred_func_1, k, sigma_noise)
        for k in kappa_samples
    ])
    log_like_2 = np.array([
        log_likelihood_gaussian(h_obs, h_pred_func_2, k, sigma_noise)
        for k in kappa_samples
    ])
    
    # Marginal likelihood (evidence) via importance sampling
    # log(Evidence) ≈ log(mean(exp(log_likelihood)))
    evidence_1 = np.mean(np.exp(log_like_1 - np.max(log_like_1))) * np.exp(np.max(log_like_1))
    evidence_2 = np.mean(np.exp(log_like_2 - np.max(log_like_2))) * np.exp(np.max(log_like_2))
    
    bayes_factor = evidence_1 / evidence_2
    
    return bayes_factor


# ============================================================================
# Utility functions for common use cases
# ============================================================================

def quick_constraint_from_null(
    h_baseline: float,
    h_anomalous_per_kappa: float,
    h_threshold: float,
    confidence_level: float = 0.95
) -> float:
    """
    Quick κ-upper-limit from null result (no MCMC, analytic approximation).
    
    Assumes: h_total = h_baseline + κ × h_anomalous_per_kappa
             Null result: h_total < h_threshold
    
    Args:
        h_baseline: Strain from known physics (EM + GR)
        h_anomalous_per_kappa: Anomalous strain per unit κ
        h_threshold: Detector threshold (5σ sensitivity)
        confidence_level: Desired CL (typically 0.95)
    
    Returns:
        κ_upper: Conservative upper limit
    
    Note: This is a simplified version. For rigorous bounds, use full MCMC.
    """
    if h_anomalous_per_kappa <= 0:
        return np.inf  # No constraint possible
    
    # Null result means h_total < h_threshold
    # h_baseline + κ × h_anomalous < h_threshold
    # κ < (h_threshold - h_baseline) / h_anomalous
    
    kappa_upper = (h_threshold - h_baseline) / h_anomalous_per_kappa
    
    # Add safety factor for confidence level (rough approximation)
    # For 95% CL, use ~2σ = 0.95 × point estimate
    safety_factor = confidence_level
    
    return safety_factor * kappa_upper


def plot_posterior_corner(posterior: PosteriorSample, filename: Optional[str] = None):
    """
    Create corner plot of posterior distribution.
    
    Requires: pip install corner
    
    Args:
        posterior: PosteriorSample object
        filename: Optional save path (None = show plot)
    """
    try:
        import corner
    except ImportError:
        raise ImportError("corner package required. Install: pip install corner")
    
    fig = corner.corner(
        posterior.samples,
        labels=posterior.param_names,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12}
    )
    
    if filename:
        fig.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig
