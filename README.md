
# CPU Benchmark Score Predictor

A machine learning web app that predicts CPU PassMark multi-thread benchmark 
scores from hardware specifications.

## Live Demo
[Open App](https://cpu-benchmark-predictor-a8u8nlutckczmxxv24q6tw.streamlit.app/)

## Overview

CPU benchmark scores are only known after physically running a test on the hardware.
This project builds a model that estimates a CPU's PassMark score using only its
hardware specifications — cores, single thread performance, price, TDP, and release year.

The model achieves an R² of 0.9833 and an RMSE of 2,331 benchmark points on
the held-out test set.

## Dataset

Source: CPU Benchmarks Compilation — Kaggle
Records: 3,825 CPUs from PassMark Performance Test

## Project Structure

- Notebook 1 — Data Loading and Exploration
- Notebook 2 — Data Cleaning
- Notebook 3 — Feature Engineering and Modeling
- app/ — Streamlit web application

## Methodology

- Exploratory data analysis to understand distributions and data quality issues
- Log transformation of the target variable to handle right skew
- Feature engineering — price per core, TDP per core, thread to core ratio
- Trained and compared three models — Linear Regression, Random Forest, XGBoost
- SHAP analysis to identify the most influential features

## Key Findings

- threadMark and cores are the two strongest predictors of benchmark performance
- Tree based models reduce prediction error by roughly 7x compared to Linear Regression
- Price and release year have moderate positive effects on benchmark scores
- Socket type has minimal predictive power once core specs are accounted for

## Tech Stack

Python, pandas, scikit-learn, XGBoost, SHAP, Streamlit

## Author
Tran Bao Quan
