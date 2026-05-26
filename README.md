# Dataset Drift Detector

Detects statistical distribution shifts between two CSV datasets. Useful for catching training/serving skew in ML pipelines — outputs a single HTML report.

## Tests used

| Column type | Test | Flags drift when |
|-------------|------|-----------------|
| Numerical | Kolmogorov-Smirnov | p < 0.05 |
| Numerical | Population Stability Index (PSI) | PSI ≥ 0.1 |
| Categorical | Chi-Squared | p < 0.05 |

PSI: `< 0.1` no drift · `0.1–0.2` moderate · `> 0.2` significant

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
# generates drift_report.html by default
python drift_detector.py --ref reference.csv --curr current.csv

# custom output path
python drift_detector.py --ref reference.csv --curr current.csv --output report.html
```

## Demo

```bash
python generate_sample_data.py
python drift_detector.py --ref sample_data/reference.csv --curr sample_data/current.csv --output report.html
```
