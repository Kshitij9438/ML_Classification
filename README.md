
---


# ML_Classification: Customer Churn Prediction

## 📌 Project Overview
This project investigates **customer churn prediction** in the telecom sector using machine learning classification models. Retaining customers is significantly more cost-effective than acquiring new ones, making churn prediction a high-value application.  

The project compares multiple models — **Logistic Regression, Support Vector Machines (SVM), and Random Forest Classifiers** — while placing strong emphasis on **handling class imbalance** to ensure fair and actionable results.

---

## 📂 Repository Structure
``` markdown

ML\_Classification/
│── Classification.ipynb                        # Jupyter Notebook with implementation
│── Comprehensive\_Classification\_Project\_Report.pdf  # Detailed report with methodology & results

```

---

## 🗂 Dataset
- **Source**: Telco Customer Churn dataset (~7,043 customer records, 21 features).  
- **Target Variable**: `Churn` (binary: Yes/No).  
- **Features**: Customer demographics, subscription details, billing information, tenure, and charges.  
- **Imbalance**: ~73% non-churn vs. 27% churn.  

### Data Preparation
- Dropped unique identifiers (`customerID`).  
- Handled missing values in `TotalCharges` via numeric coercion and imputation.  
- Encoded categorical variables (binary mapping / one-hot encoding).  
- Standardized numerical features.  
- Applied two imbalance-handling strategies:  
  - **Class weights** within algorithms.  
  - **SMOTE** oversampling.  

---

## ⚙️ Methodology
1. **Exploratory Data Analysis (EDA)** – Identified drivers of churn.  
2. **Data Preprocessing** – Cleaning, encoding, scaling, balancing.  
3. **Model Training** – Logistic Regression, SVM, Random Forest under different imbalance strategies.  
4. **Evaluation Metrics** – Accuracy, Precision, Recall, F1-score (with focus on churn class).  
5. **Comparison & Recommendations** – Based on use-case requirements (interpretability vs. balance vs. recall).  

---

## 📈 Results

| Model               | Imbalance Handling | Accuracy | Precision (Churn) | Recall (Churn) | F1-score | Remarks |
|---------------------|-------------------|----------|-------------------|----------------|----------|---------|
| Logistic Regression | Class Weights     | 0.759    | 0.535             | 0.719          | 0.613    | Strong recall, interpretable |
| Logistic Regression | SMOTE             | 0.756    | 0.530             | 0.717          | 0.609    | Similar to weights, added complexity |
| SVM                 | None              | 0.780    | 0.560             | 0.520          | 0.540    | Good accuracy, under-detects churners |
| SVM                 | Class Weights     | 0.740    | 0.500             | 0.710          | 0.590    | Higher recall, lower precision |
| Random Forest       | None              | 0.800    | 0.680             | 0.500          | 0.580    | High accuracy, biased to majority |
| Random Forest       | Class Weights     | 0.780    | 0.620             | 0.690          | 0.650    | Balanced trade-off, strong F1 |

---

## ✅ Model Recommendations
- **Interpretability**: Logistic Regression with class weights – coefficients provide direct insights into churn drivers.  
- **Balanced Trade-off**: Random Forest with class weights – best overall balance of precision and recall.  
- **Maximizing Recall**: SVM with class weights – effective for aggressive retention campaigns where missing churners is very costly.  

---

## 💡 Key Insights
- Handling **class imbalance is critical**; unadjusted models underpredict churners.  
- **SVM** offers strong recall (useful for aggressive retention) but sacrifices precision.  
- **Random Forest with class weights** provides the most consistent balance across metrics.  
- **Logistic Regression** remains valuable for its simplicity and interpretability.  

---

## 🚀 Next Steps
- **Threshold optimization**: Align probability cutoffs with business cost trade-offs.  
- **Hybrid approach**: Use Random Forest for prediction + Logistic Regression for interpretability dashboards.  
- **Feature engineering**: Include behavioral features (e.g., complaint history, service usage).  
- **Advanced models**: Explore Gradient Boosting and XGBoost.  
- **Deployment**: Integrate final model into CRM systems for real-time churn detection.  

---

## 🛠 Tech Stack
- **Languages**: Python  
- **Libraries**: Pandas, NumPy, Scikit-learn, Imbalanced-learn, Matplotlib, Seaborn  
- **Tools**: Jupyter Notebook  

---

## 📜 License
This project is released under the **MIT License**.  

---

## 🙌 Acknowledgments
- Dataset: [Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)  
- Developed as part of a comprehensive machine learning classification project.  


---
