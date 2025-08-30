
---


# ML_Classification: Customer Churn Prediction

## 📌 Project Overview
This project addresses the critical business problem of **customer churn prediction** using the **Telco Customer Churn dataset**. The objective is to build and evaluate classification models that balance **predictive performance** with **interpretability**, enabling businesses to proactively retain customers and reduce revenue loss.

---

## 📂 Repository Structure
```markdown

ML\_Classification/
│── Classification.ipynb               # Full Jupyter Notebook implementation
│── Classification\_Project\_Report.pdf  # Detailed project report with results & insights

```

---

## 🗂 Dataset
- **Source**: Telco Customer Churn dataset (7,043 customer records, 21 features).  
- **Features**: Customer demographics, subscribed services, billing methods, tenure, and charges.  
- **Target Variable**: `Churn` (binary: Yes/No).  

### Data Preparation
- Imputed missing values in `TotalCharges`.  
- Encoded categorical variables into dummy features.  
- Standardized numerical features.  
- Applied **SMOTE** to address class imbalance (~27% churn).  

---

## ⚙️ Methodology
1. **Exploratory Data Analysis (EDA)** – Identified churn patterns and key drivers.  
2. **Data Preprocessing** – Cleaning, encoding, scaling, and balancing.  
3. **Model Training** – Logistic Regression, Random Forest, Support Vector Machine (SVM).  
4. **Evaluation** – Accuracy, Precision, Recall, F1-score, and ROC-AUC using 5-fold cross-validation.  
5. **Business Insights** – Translating ML results into actionable recommendations.  

---

## 📈 Results
| Model                | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|----------------------|----------|-----------|--------|----------|---------|
| Logistic Regression  | 79%      | 71%       | 63%    | 67%      | 0.84    |
| Random Forest        | 82%      | 75%       | 69%    | 72%      | 0.87    |
| SVM (Support Vector) | **85%**  | **78%**   | **74%**| **76%**  | **0.90**|

✅ **Final Recommendation**: Support Vector Machine (SVM) – highest predictive performance with robust ROC-AUC.

---

## 💡 Key Insights
- Customers with **month-to-month contracts** are most at risk of churn.  
- **High monthly charges** strongly correlate with churn.  
- **Short-tenure customers** are more likely to leave.  
- Lack of **online security/tech support** services increases churn probability.  
- **Paperless billing** customers show slightly higher churn.  

**Business Recommendations**:  
- Offer discounts for **long-term contracts**.  
- Bundle **support/security services** to improve retention.  
- Launch **targeted campaigns** for high-risk groups (month-to-month, high charges, low tenure).  

---

## 🚀 Next Steps
- Integrate additional behavioral data (e.g., service call history, network usage).  
- Explore advanced ensemble methods and deep learning models.  
- Deploy **SHAP/partial dependence plots** for interpretability.  
- Integrate the model into **CRM systems** for real-time churn prediction.  

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
- Developed as part of a machine learning classification project.  


---
