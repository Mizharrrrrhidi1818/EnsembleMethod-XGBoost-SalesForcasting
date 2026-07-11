# 📈 Sales Forecasting with XGBoost

An end-to-end machine learning project for predicting retail sales using **XGBoost**. This repository features a deep dive into the algorithm's mechanics by implementing XGBoost from scratch using NumPy and comparing it against the industry-standard `xgboost` library.

## 📖 Overview
In modern business analytics, accurately forecasting sales performance is critical for inventory management and strategic planning. This project explores two distinct predictive tasks:
1. **Regression:** Predicting continuous sales values for revenue forecasting.
2. **Classification:** Predicting whether an order belongs to a high- or low-sales category to enable proactive resource allocation.

## 🗂️ Dataset
The project utilizes the **[Sales Forecasting Dataset](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting)** from Kaggle. 
- **Features:** Transactional retail data including temporal data, geographical locations, customer segments, and product categories.
- **Target:** `Sales` (continuous) and `Sales_binary` (discretized high/low sales).

## 🛠️ Methodology
1. **Data Cleaning:** Imputation of missing postal codes and removal of duplicate records.
2. **Feature Engineering:** Aggregation of transactions by `Order ID`, extraction of temporal features (weekday, month), and One-Hot Encoding of categorical variables.
3. **Modeling:** 
   - **Custom Implementation:** A from-scratch mathematical implementation of XGBoost using NumPy to expose the underlying second-order Taylor expansion and gradient boosting mechanics.
   - **Library Implementation:** Using the official `xgboost.XGBClassifier` and `XGBRegressor`.
4. **Evaluation:** RMSE for regression; Accuracy, Precision, Recall, and F1-Score for classification.

## 📊 Key Results
| Task | Model | Metric | Score |
| :--- | :--- | :--- | :--- |
| **Regression** | Custom XGBoost | RMSE | **0.4872** |
| **Regression** | XGBoost Library | RMSE | **0.4872** |
| **Classification** | Custom XGBoost | Accuracy | **71.39%** |
| **Classification** | XGBoost Library | Accuracy | **70.31%** |

*Note: The custom implementation achieved a more balanced trade-off between precision and recall for the minority class (High Sales) compared to the library's default parameters.*

## 📂 Project Structure
```text
├── data/                  # Raw and processed datasets
├── notebooks/             # Jupyter notebooks containing the analysis
├── src/                   # Modularized Python scripts
├── images/                # Visualizations and plots
├── .gitignore             # Files to ignore in Git
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
