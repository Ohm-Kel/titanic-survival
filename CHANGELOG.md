# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Woman-child-group model (`titanic.groups`): women survive and men do not, except where a passenger's travel group (shared ticket or family) shows otherwise. It is now the model that `titanic-train` saves and `titanic-predict` submits.
- Repeated 10 × 5-fold cross-validation in `titanic-train`.

### Removed
- Gradient boosting grid search from `titanic-train`; gradient boosting is still compared in `titanic-experiments`.

## [0.1.0] - 2026-10-21

### Added
- Kaggle data download script.
- Exploratory data analysis notebook.
- Leakage-safe pipeline: imputation, scaling, one-hot encoding and Title feature.
- Feature ablation and model comparison (`titanic-experiments`).
- Gradient boosting tuned with GridSearchCV, out-of-fold evaluation and confusion matrix (`titanic-train`).
- Kaggle submission command (`titanic-predict`); public leaderboard score 0.761.
- Fixture-based pytest suite with coverage, and GitHub Actions CI on Python 3.10–3.12.