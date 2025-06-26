import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def bvalue_goodness(M: np.ndarray, m_step: float = 0.01, threshold: float = 90,
                    p_lost_max: float = 50, plot: bool = False):
    """
    Estimate b-value and magnitude of completeness (Mc) using the
    Goodness-of-Fit method from Wiemer and Wyss (2000).
    Based on a code written by Grzegorz Kwiatek.

    Parameters
    ----------
    M : np.ndarray
        Array of earthquake magnitudes.
    m_step : float, optional
        Magnitude bin width. Default is 0.01.
    threshold : float, optional
        Goodness-of-fit threshold (%) to accept the b-value fit. Default is 90.
    p_lost_max : float, optional
        Maximum acceptable percentage of lost events below Mc. Default is 50.
    plot : bool, optional
        If True, plot the cumulative distribution and fit. Default is False.

    Returns
    -------
    b_value : float
        Estimated b-value.
    b_error : float
        2-sigma uncertainty in b-value (Shi & Bolt, 1982).
    m_min : float
        Estimated magnitude of completeness (Mc).
    fit_percent : int
        Maximum fit percentage across tested Mc values.
    retained_ratio : float
        Fraction of events retained above Mc.
    cum : dict
        Dictionary containing:
        - 'R' : Magnitude bins
        - 'N' : Cumulative counts above bins
        - 'Nlimt' : Fitted exponential model values (if computed)

    References
    ----------
    Wiemer, S., & Wyss, M. (2000). Bulletin of the Seismological Society of America. https://doi.org/10.1785/0119990114

    Shi, Y., & Bolt, B. A. (1982). Bulletin of the Seismological Society of America, 72(5) https://doi.org/10.1785/BSSA0720051677
    """
    M_RAW = np.copy(M)
    R = np.arange(min(M), max(M), m_step)
    dm = R[1] - R[0]

    N = np.array([np.sum(M >= r) for r in R])
    cum = {'N': N, 'R': R}

    MC, FIT, BETA = np.zeros_like(R), np.zeros_like(R), np.zeros_like(R)

    for k, mc in enumerate(R):
        if np.sum(M >= mc) == 0:
            continue
        mc_shift = mc - dm / 2
        M_above = M[M >= mc_shift]
        if len(M_above) == 0:
            continue
        BETA[k] = 1 / (np.mean(M_above) - mc_shift)
        N_pred = np.exp(-BETA[k] * (R - mc_shift)) * len(M_above)
        subset = R >= mc_shift
        FIT[k] = 100 - np.sum(np.abs(N_pred[subset] - N[subset])) / np.sum(N[subset]) * 100
        MC[k] = mc

    try:
        m_index = np.where(FIT >= threshold)[0][0]
    except IndexError:
        return -999, -999, -999, int(np.max(FIT)), -999, cum

    m_min = MC[m_index]
    n_events = np.sum(M_RAW >= m_min)
    retained_ratio = n_events / len(M_RAW)

    if retained_ratio < (100 - p_lost_max) / 100:
        return -999, -999, -999, int(np.max(FIT)), retained_ratio, cum

    beta_value = BETA[m_index] * (n_events - 1) / n_events
    b_value = beta_value / np.log(10)
    b_error = np.log(10) * b_value ** 2 * np.sqrt(np.var(M_RAW[M_RAW >= m_min]) / n_events)

    cum['Nlimt'] = np.exp(-beta_value * (R - m_min)) * n_events

    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(8, 6))
        ax[0].plot(R, N, 'k', label='Observed')
        ax[0].set_yscale('log')
        ax[0].set_xlabel('Magnitude')
        ax[0].set_ylabel('Cumulative number of events')

        if 'Nlimt' in cum:
            ax[0].plot(R[R >= m_min], cum['Nlimt'][R >= m_min], 'b', label='Fit')
            ax[0].scatter(R[R >= m_min], N[R >= m_min], color='red', s=10, label='Used for fit')
            ax[0].legend()

        ax[1].plot(R, FIT, 'k')
        ax[1].axvline(m_min, color='red')
        ax[1].axhline(threshold, color='red', linestyle='--')
        ax[1].set_xlabel('Magnitude')
        ax[1].set_ylabel('Fit (%)')

        fig.suptitle(f'b = {b_value:.2f} ± {2 * b_error:.2f} (2σ), Mc = {m_min:.2f}, N = {len(M)}, '
                     f'N(M ≥ Mc) = {n_events}, Retained = {retained_ratio:.2f}')
        plt.tight_layout()
        plt.show()

    return b_value, b_error, m_min, int(np.max(FIT)), retained_ratio, cum


if __name__ == '__main__':
    catalog = pd.read_csv('earthquake_catalog_curated.csv')
    mags = catalog['magnitude'].to_numpy()

    b_val, b_err, m_c, fit, retained_ratio, cum = bvalue_goodness(
        mags, m_step=0.005, threshold=93, p_lost_max=90, plot=False)