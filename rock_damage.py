import numpy as np
import pandas as pd

def compute_potency(magnitude: np.ndarray) -> np.ndarray:
    """
    Compute seismic potency based on local magnitude.

    Parameters
    ----------
    magnitude : np.ndarray or pd.Series
        Local magnitude values (ML).

    Returns
    -------
    potency : np.ndarray
        Seismic potency.

    The damage volume is proportional to the seismic potency,
    with different empirical scaling laws used for M < 3.5 and M >= 3.5.

      Notes
    -----
    The method follows:
    - Trugman and Ben-Zion (2024, The Seismic Record, https://doi.org/10.1785/0320240022A)

    """
    potency = np.where(
        magnitude < 3.5,
        10 ** (1.06018 * magnitude - 3.81636),
        10 ** (0.03310 * magnitude + 0.14673 * magnitude ** 2 - 2.01898)
    )
    return potency


def compute_normvoldmg(catalog: pd.DataFrame) -> pd.DataFrame:
    """
    Compute normalized damage volume based on seismic potency.

    Parameters
    ----------
    catalog : pd.DataFrame
        DataFrame with a 'magnitude' column.

    Returns
    -------
    catalog : pd.DataFrame
        Modified DataFrame with:
        - 'radius'
        - 'area'
        - 'potency'
        - 'damage'
        - 'damage_normalized'

          Notes
    -----
    The method follows:
    - Jamtveit et al. (2018, Nature, https://doi.org/10.1038/s41586-018-0045-y)
    - Ben-Zion and Zaliapin (2019, Earth and Planetary Science Letters, https://doi.org/10.1016/j.epsl.2019.02.006)

    """
    if 'magnitude' not in catalog.columns:
        raise ValueError("Input catalog must contain a 'magnitude' column.")

    deltaeps = 1e-4
    gamma = 1 / 500

    M = catalog['magnitude']
    potency = compute_potency(M)
    catalog['potency'] = potency

    # Damage radius from potency
    catalog['radius'] = ((7 / 16) * 1e-5 * potency / deltaeps) ** (1 / 3)
    catalog['area'] = np.pi * catalog['radius'] ** 2

    # Rock damage proportional to seismic potency
    catalog['damage'] = (7 / 16) * gamma * np.pi * 1e-5 * potency / deltaeps
    catalog['damage_normalized'] = catalog['damage'] / catalog['damage'].sum()

    return catalog

if __name__ == '__main__':
    catalog = pd.read_csv('earthquake_catalog_curated.csv')
    catalog = compute_normvoldmg(catalog)

    print(catalog[['magnitude', 'potency', 'radius', 'damage', 'damage_normalized']])
