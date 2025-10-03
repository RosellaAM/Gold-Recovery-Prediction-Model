# Gold Recovery Prediction Model - Mineral Processing Optimization

[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Machine-Learning-blueviolet)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Pandas](https://img.shields.io/badge/Pandas-1.0%252B-brightgreen)](https://pandas.pydata.org/)

**Gold Recovery Prediction Model** is a machine learning solution for Zyfra that predicts gold recovery rates from raw ore during the purification process. The project addresses the challenge of optimizing gold extraction efficiency in mining operations through regression models that anticipate recovery percentages at both rougher and final stages. The system analyzes historical operational parameters and laboratory measurements to build predictive models that can serve as the foundation for an automated control system, eliminating non-profitable parameters and maximizing production yield.

## 🚀 Results
The final Random Forest Regressor model demonstrated:
* 12.82% total sMAPE on test data.
* 13.54% sMAPE for Rougher stage (initial concentration).
* 12.58% sMAPE for Final stage (purification).
* Strong generalization capability with minimal overfitting.
* Consistent performance between validation (12.45%) and test sets (12.82%).

## 💼 Business Impact
* **Process Optimization**: Enables real-time adjustment of operational parameters.
* **Resource Efficiency**: Reduces energy consumption and operational costs.
* **Increased Recovery**: Maximizes gold extraction from raw materials.
* **Data-Driven Operations**: Foundation for automated control systems.
* **Sustainable Mining**: More efficient use of resources in gold purification.

## 🎯 Core Skills
* Industrial Data Analysis: Processing complex manufacturing process data with 87+ features.
* Feature Engineering: Strategic alignment of training (87 features) and production (53 features) datasets.
* Data Validation: Verification of gold recovery calculation integrity and data consistency.
* Process Understanding: Analysis of multi-stage purification (Rougher → Primary Cleaning → Final).
* Regression Modeling: Implementation of Random Forest, Decision Tree, and Linear Regression.
* Hyperparameter Optimization: Fine-tuning with 15 max depth and 150 estimators.
* Custom Metric Implementation: sMAPE (Symmetric Mean Absolute Percentage Error) for industrial accuracy measurement.
* Statistical Analysis: Metal concentration evolution, particle size distribution, and process consistency.
* Model Robustness: Validation against overfitting with consistent train/test performance.

## 🛠️ Tech Stack
* **Machine Learning** → Scikit-learn, Random Forest, Decision Tree, Linear Regression
* **Backend** → Python 3.8+, Pandas, NumPy, SciPy
* **Visualization** → Matplotlib, Seaborn
* **Development** → Jupyter Notebooks

## Local Execution
1. Clone the repository:

git clone https://github.com/RosellaAM/Gold-Recovery-Prediction-Model.git

2. Install dependencies:

pip install -r requirements.txt

3. Run analysis:

  jupyter notebook notebooks/zyfra_gold_recovery_model.ipynb
