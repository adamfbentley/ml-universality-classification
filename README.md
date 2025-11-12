# ML Universality Classification

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)

A machine learning project for classifying surface growth universality classes. Uses Random Forest and SVM classifiers to distinguish between ballistic deposition, Edwards-Wilkinson, and KPZ growth dynamics based on simulated surface features.

## Overview

This project explores whether machine learning can identify different universality classes in surface growth models. I simulate three types of growth processes (ballistic deposition, Edwards-Wilkinson, and KPZ), extract features from the resulting surfaces, and train classifiers to predict which model generated each surface.

## Features

- Physics simulations for three growth models
- Feature extraction from surface height profiles
- Random Forest and SVM classification
- Visualization of results and feature importance

## Requirements

```
numpy
scikit-learn
matplotlib
scipy
```

## Usage

Train the model:
```python
python train_model.py
```

Run classification on new data:
```python
python classifier.py
```

## Project Structure

```
ml-universality-classification/
├── train_model.py               # Model training script
├── classifier.py                 # Classification script
├── src/                          # Core modules
│   ├── physics_simulation.py
│   ├── feature_extraction.py
│   ├── ml_training.py
│   └── analysis.py
├── results/                      # Output figures
└── README.md
```

## Results

The classifiers achieve good accuracy in distinguishing between the three universality classes, with Random Forest generally outperforming SVM. Feature importance analysis shows that scaling exponents and roughness measures are the most discriminative features.

## Notes

This is an exploratory project completed as part of my undergraduate thesis work. The simulations are somewhat simplified compared to full research implementations, but provide a solid foundation for understanding ML applications in statistical physics.

## License

MIT
