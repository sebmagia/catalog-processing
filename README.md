# Seismic Potency and b-Value Estimation Tools

This repository provides two Python scripts for reproducing the results of the Nunez Jara et al. paper: Unraveling the spatiotemporal fault activation in a complex fault system: the run-up to the 2023 MW 7.8 Kahramanmaraş earthquake, Türkiye, currently in review.




## Contents

rock_damage.py: Estimates seismic potency and rock damage volume from magnitude using empirical relationships.
bvalue_analysis.py: Computes magnitude of completeness and b-value using the Wiemer & Wyss (2000) Goodness-of-Fit method.
earthquake_catalog_curated.csv: Example dataset with a "magnitude" column used for demonstration.

---

##  Example Usage

### Run damage volume estimation:


python rock_damage.py



### Run b-value and Mc estimation:


python bvalue_analysis.py





## Requirements

Install required dependencies with:


pip install -r requirements.txt


Dependencies:
- numpy
- pandas
- matplotlib

---

## License

This project is shared under the MIT License. You are free to use, modify, and redistribute the code with proper attribution.

---

