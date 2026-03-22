# Ariel Data Challenge 2025

Kaggle competition solution for predicting exoplanet atmospheric spectra from raw ESA Ariel detector data.

## Approach

**Metric:** Gaussian Log-Likelihood — requires accurate means *and* well-calibrated per-wavelength uncertainties.

**Pipeline:**
1. **Calibration** — ADC inversion, hot/dead pixel masking, dark subtraction, CDS, time binning, flat field correction (based on the official ADC2025 calibration notebook)
2. **Feature engineering** — FGS1-detrended AIRS light curves; per-wavelength statistics, transit geometry, out-of-transit residuals, and spectral neighbourhood context
3. **Models** — LightGBM (mean + q16/q84 quantile regression) + ExtraTreesRegressor, 5-fold CV at planet level
4. **Ensemble** — OOF-optimised softmax blend of LGBM and ET predictions
5. **Uncertainty** — quantile-based σ combined with fold ensemble variance, then per-wavelength scale calibration against OOF residuals

## Structure

```
ariel_pipeline.ipynb   # end-to-end notebook (calibration → submission)
README.md
```

## Requirements

```
numpy pandas scipy astropy lightgbm scikit-learn tqdm matplotlib
```

## Usage

Set `CFG['data_path']` and `CFG['output_path']` at the top of the notebook, then run all cells. `submission.csv` and `diagnostics.png` are written to `output_path`.

Set `device='cpu'` in `lgbm_params` if no GPU is available.
