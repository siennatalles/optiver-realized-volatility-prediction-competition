# Optiver Realized Volatility Prediction Competition

Independent entry in Optiver's Kaggle Code competition: predicting short-term stock volatility from order book and trade data.

## Problem

Given order book and trade data from a fixed 10-minute window, predict the realized volatility of the next 10-minute window, across 429,000 windows spanning 112 stocks.

## Approach

Computed weighted average price (WAP) and log returns to calculate realized volatility from order book snapshots. Engineered book, trade, and market-wide features: realized volatility over smaller windows (past 5 and 2 minutes), bid-ask spread, trading activity and volume, and average realized volatility across all stocks in the same window. Compared naive baseline (past realized volatility), Linear Regression, and LightGBM models, using grouped time-window cross-validation splits so the same 10-minute window never appeared in both training and validation.

## Result

Reduced RMSPE from 0.341 (naive baseline) to 0.249, with LightGBM as the best-performing model.

## Data

Uses the [Optiver Realized Volatility Prediction](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction) dataset from Kaggle (not included in this repo, per competition rules). Download the competition files from Kaggle to run the notebook.

## Contents

- `optiver_volatility_prediction.ipynb` — full analysis: naive baseline, feature engineering, Linear Regression, LightGBM, and model comparison

## Tools

Python, Pandas, NumPy, Scikit-learn (Linear Regression, GroupKFold), LightGBM
