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
The workflow follows a structured, reproducible machine learning pipeline:
1. **Data Acquisition:** Download the transactional sales dataset from Kaggle.
2. **Preprocessing & Cleaning:** Handle missing values, remove duplicates, convert data types, and drop non-predictive identifiers.
3. **Feature Engineering:** Aggregate transactions by order, create temporal features (weekday, month), discretize sales into interpretable bins, and one-hot encode categorical variables.
4. **Target Formulation:** 
   - *Regression:* Use raw aggregated sales as a continuous target.
   - *Classification:* Convert sales bins into a binary label (0 = Low Sales, 1 = High Sales).
5. **Model Development:** 
   - Build a custom XGBoost implementation from scratch to expose the underlying mathematics.
   - Train the official `xgboost.XGBClassifier` and `XGBRegressor` for benchmarking.
6. **Evaluation:** Compare predictions using domain-appropriate metrics (RMSE for regression; Accuracy, Precision, Recall, and F1-Score for classification).
7. **Analysis & Interpretation:** Contrast mathematical theory with empirical results, discuss trade-offs, and outline production considerations.

##  Data Exploration and Preprocessing
### Dataset Description
The dataset used in this study is the **"sales-forecasting"** dataset available on Kaggle (uploaded by *rohitsahoo*). It contains transactional retail data with the following key features:
- **Identifiers:** `Row ID`, `Order ID`, `Customer ID`, `Product ID`.
- **Temporal Data:** `Order Date`, `Ship Date`.
- **Categorical Features:** `Ship Mode`, `Customer Name`, `Segment`, `Country`, `City`, `State`, `Postal Code`, `Region`, `Category`, `Sub-Category`, `Product Name`.
- **Target Variable:** `Sales` (continuous numerical value).

### Data Cleaning (Missing values, duplicate)
- **Missing Values:** An initial inspection revealed missing values exclusively in the `Postal Code` column (11 missing entries). To address this, domain knowledge was applied: the missing postal codes were imputed with a known Vermont ZIP code (`05401` for Burlington, VT) based on the corresponding `City` and `State` values. After imputation, the dataset contained zero missing values.
- **Duplicates:** The dataset contained duplicate rows. To ensure data integrity and prevent data leakage or biased learning, all duplicate records were identified and removed using the `drop_duplicates()` function.

### Feature Engineering
Feature engineering was crucial to transform raw transactional data into a format suitable for modeling:
- **Aggregation by Order:** Since a single order can contain multiple products, transactions were grouped by `Order ID`. The `Sales` column was summed to represent the total order value. Other features like `Order Date`, `Ship Mode`, `Segment`, and `Region` were aggregated by taking their first occurrence. Unique `Category` and `Sub-Category` values were joined into strings.
- **Temporal Features:** Extracted `Order_Weekday` (e.g., Monday, Tuesday) and `Order_Month` (e.g., Jan, Feb) from the `Order Date` to capture seasonal and weekly purchasing patterns.
- **Target Discretization (for Classification):** The continuous `Sales` variable was discretized into bins: `[0,100)`, `[100,300)`, `[300,600)`, and `[600+,10k]`. A binary target `Sales_binary` was then created, where orders falling into the higher value bins (`[300,600)` and `[600+,10k]`) were labeled as `1` (High Sales), and the rest as `0` (Low Sales).
- **Encoding:** Categorical features (`Order_Weekday`, `Order_Month`, `Ship Mode`, `Segment`, `Region`, `Category`) were transformed using One-Hot Encoding (`pd.get_dummies`) to make them compatible with the XGBoost algorithm.

After dropping any resulting NaNs, the final feature matrix `X` had a shape of **(4916, 33)**. The target distribution for the binary classification task was **3143 instances for class 0** (Low Sales) and **1773 instances for class 1** (High Sales).

### Training and Test
The engineered dataset was split into training and testing sets to evaluate model performance on unseen data. 
- **Features (X):** The one-hot encoded matrix containing temporal, geographical, and product-related features.
- **Targets (y):** 
  - For Regression: The continuous aggregated `Sales` values.
  - For Classification: The binary `Sales_binary` labels.
Both the custom NumPy-based XGBoost models and the official Scikit-learn API XGBoost models were trained on the training split and subsequently evaluated on the test split.

## 📊 Discussion and Results

###  XGBoost Regression
**Mathematical Formulation:** 
XGBoost builds trees sequentially to correct the residuals of previous trees. The objective function combines a loss function (Mean Squared Error) and a regularization term penalizing tree complexity:
$$L(\theta) = \sum_{i=1}^n l(y_i, \hat{y}_i^{(t)}) + \sum_{k=1}^t \Omega(f_k)$$
It uses a second-order Taylor expansion to optimize the loss, calculating optimal leaf weights ($w_j^* = -G_j / (H_j + \lambda)$) and split gains based on gradients ($g_i$) and Hessians ($h_i$).

**Evaluation Metric:** 
Root Mean Squared Error (RMSE) was used, which penalizes larger errors quadratically, making it sensitive to outliers (important in sales forecasting).

**Results:** 
- **Custom Implementation RMSE:** `0.4872`
- **Scikit-learn (Library) RMSE:** `0.4872`

The custom implementation perfectly matched the mathematical output of the optimized library. This validates the correctness of the from-scratch mathematical formulation for regression and proves that the core gradient boosting mechanics are identical regardless of the implementation language.

### XGBoost Classification
**Mathematical Formulation:** 
For classification, the loss function shifts to Binary Cross-Entropy (Log Loss):
$$l(y_i, \hat{y}_i) = -[y_i \log(p_i) + (1 - y_i) \log(1 - p_i)]$$
The gradients and Hessians are recalculated based on the sigmoid function mapping raw logits to probabilities ($g_i = p_i - y_i$, $h_i = p_i(1 - p_i)$). The tree-growing mechanics remain identical, showcasing XGBoost's unified optimization framework.

**Evaluation Metrics:** 
Accuracy, Precision, Recall, and F1-Score.

**Results:**
| Model | Accuracy | Class 0 (Low) Precision/Recall | Class 1 (High) Precision/Recall |
| :--- | :--- | :--- | :--- |
| **Custom Implementation** | **0.7139** | 0.84 / 0.69 | 0.57 / **0.76** |
| **XGBoost Library** | **0.7031** | 0.69 / **0.97** | 0.82 / **0.21** |


The custom implementation achieved a slightly higher overall accuracy (71.39% vs 70.31%) and provided a much more balanced trade-off between precision and recall for the minority class (High Sales). 
- The **Custom Model** successfully identified 76% of all high-sales orders (Recall = 0.76), which is highly valuable for business operations (e.g., ensuring high-value orders get priority shipping).
- The **Library Model** heavily favored predicting Class 0, achieving 97% recall for Low Sales but only identifying 21% of High Sales orders. This indicates that the library's default parameters are highly biased toward the majority class in this specific run. To fix this in production, one would typically apply techniques like `scale_pos_weight` or SMOTE to handle the class imbalance.

## 📂 Project Structure
```text
├── data/                  # Raw and processed datasets
├── notebooks/             # Jupyter notebooks containing the analysis
├── src/                   # Modularized Python scripts
├── images/                # Visualizations and plots
├── .gitignore             # Files to ignore in Git
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
