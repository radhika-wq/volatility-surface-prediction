### 🚀 Implied Volatility Surface Reconstruction: 3-Bucket TTE QP Ensemble (v13)

This project implements a rigorous cross-sectional quantitative framework designed to reconstruct missing data points on the NIFTY 50 options chain Implied Volatility (IV) surface. The methodology guarantees **zero temporal lookahead bias** by executing independent, separate mathematical fits for each unique timestamp. This eliminates numerical instabilities near contract settlement while strictly preserving formatting and smile regularities.

---

# NIFTY IV Surface Reconstruction

## Overview

This project focuses on reconstructing missing **Implied Volatility (IV)** values in the NIFTY 50 options chain. The objective is to recover a realistic and consistent volatility surface by leveraging relationships between option strikes, option types, and time to expiry.

---

## Problem Statement

Given a partially observed implied volatility surface containing:

- NIFTY Call (CE) and Put (PE) options
- Multiple strike prices
- Multiple timestamps
- Missing IV observations

The goal is to predict missing IV values while preserving the characteristics of the volatility smile and term structure.

---

## Financial Motivation

Implied volatility surfaces exhibit several well-known market structures:

- **Volatility Smile** – IV varies across strike prices.
- **Volatility Skew** – Different behavior across calls and puts.
- **Term Structure** – IV evolves as expiry approaches.
- **Surface Continuity** – Nearby strikes tend to have related volatility levels.

A robust reconstruction method should preserve these properties while accurately filling missing observations.

---

## Methodology

### 1. Data Processing

- Parsed NIFTY option symbols
- Extracted strike prices from option names
- Separated Call (CE) and Put (PE) surfaces
- Computed Time-To-Expiry (TTE) for every observation

### 2. Volatility Smile Reconstruction Models

#### Quadratic Polynomial Fit

Models the volatility smile using a second-order polynomial:

```python
IV = a + bK + cK²
```

where `K` is the strike price.

#### Variance Space Polynomial Fit

Instead of fitting IV directly, the model fits:

```python
IV²
```

across strikes and converts back to implied volatility.

#### PCHIP Interpolation

Piecewise Cubic Hermite Interpolation (PCHIP) is used to reconstruct missing values while preserving local smile structure and avoiding spline overshooting.

#### Adaptive Smile Reconstruction

Combines:

- PCHIP interpolation for interior missing strikes
- Polynomial extrapolation for smile wings

#### Log-Wing Extrapolation

Uses log-linear extrapolation in the smile wings while preserving interior structure through interpolation.

#### Total Variance Reconstruction

Near expiry, implied volatility becomes unstable.

Instead of fitting IV directly:

```python
Total Variance = IV² × T
```

is reconstructed and converted back to implied volatility.

This improves stability for short-dated options.

---

## Ensemble Framework

Rather than relying on a single model, multiple reconstructed surfaces are generated and combined.

The final IV estimate is obtained through a weighted ensemble of:

- Polynomial Smile
- Variance Smile
- PCHIP Smile
- Adaptive Smile
- Log-Wing Smile
- Total Variance Smile

---

## Cross-Validation Strategy

A stratified masking framework was used:

1. Existing IV observations were temporarily hidden.
2. Missing values were reconstructed.
3. Prediction error was measured against the original values.

This process allows objective evaluation of different reconstruction methods.

---

## Expiry-Aware Weight Optimization

The dataset was divided into three regimes:

### Normal Period

```text
TTE > 1950 minutes
```

### Mid-Expiry Period

```text
60 < TTE ≤ 1950 minutes
```

### Final Expiry Stretch

```text
TTE ≤ 60 minutes
```

Separate ensemble weights were optimized for each regime using constrained optimization.

---

## Technologies Used

- Python
- Pandas
- NumPy
- SciPy
- Scikit-Learn
- Jupyter Notebook

---

## Project Structure

```text
NIFTY-IV-Surface-Reconstruction/
│
├── nifty_iv_surface.ipynb
├── solve_v13.py
├── README.md
```

---

## Key Concepts

- Implied Volatility
- Volatility Smile
- Volatility Skew
- Surface Reconstruction
- Total Variance Modelling
- Cross-Sectional Interpolation
- Ensemble Learning
- Constrained Optimization
- Options Analytics

---

## Results

Implemented and compared multiple volatility surface reconstruction techniques and combined them through an optimized ensemble framework to generate a smooth and financially consistent implied volatility surface.

---

