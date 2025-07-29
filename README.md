# 🫀 Heart Disease Prediction using Machine Learning

This project builds a predictive system to detect the presence of **heart disease** using multiple machine learning algorithms.  
The models are compared using **Accuracy, Precision, Recall, and F1-Score**, along with a performance visualization.

---

## 📁 Dataset
- **Source:** [UCI Cleveland Heart Disease Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data)  
- **Total Attributes:** 13 features + target variable  
- **Target:**  
  - `1` → Patient has heart disease  
  - `0` → No heart disease  

---

## 📌 Feature Descriptions (and Their Importance)

| Feature       | Description | Medical Importance |
|---------------|-------------|-------------------|
| **age**       | Patient’s age in years | Risk of heart disease increases with age. |
| **sex**       | Gender (1 = male, 0 = female) | Men are generally at higher risk earlier than women. |
| **cp**        | Chest pain type (0–3) | Indicates angina type; typical angina is a strong risk indicator. |
| **trestbps**  | Resting blood pressure (mm Hg) | High BP is a significant risk factor for heart disease. |
| **chol**      | Serum cholesterol (mg/dl) | High cholesterol leads to arterial blockage and heart disease. |
| **fbs**       | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) | Diabetes and high sugar levels increase heart risk. |
| **restecg**   | Resting electrocardiographic results (0–2) | Detects abnormalities in heart rhythm or size. |
| **thalach**   | Maximum heart rate achieved | Lower values may indicate poor heart function. |
| **exang**     | Exercise induced angina (1 = yes, 0 = no) | Angina during exercise suggests blocked arteries. |
| **oldpeak**   | ST depression induced by exercise | Measures heart stress; higher depression indicates risk. |
| **slope**     | Slope of the ST segment (0–2) | Related to exercise ECG results; abnormal slopes indicate issues. |
| **ca**        | Number of major vessels (0–3) colored by fluoroscopy | Higher count usually correlates with less blockage. |
| **thal**      | Thalassemia test (3 = normal, 6 = fixed defect, 7 = reversible defect) | Detects blood and oxygen flow abnormalities. |
| **target**    | Heart disease presence (1 = disease, 0 = no disease) | Label for prediction. |

---

## 📊 Algorithms Used
- Support Vector Machine (SVM)
- Decision Tree
- K-Nearest Neighbours (KNN)
- Random Forest
- Gradient Boosting
- Voting Classifier (Ensemble)

---

## 📌 Key Features
- Cleaned and preprocessed dataset (handled missing values)  
- Standardized features using **StandardScaler**  
- Hyperparameter tuning with **GridSearchCV**  
- Evaluated models with multiple metrics  
- Visualized performance for easy comparison  

---

## 🔧 Tech Stack
- **Programming Language:** Python  
- **Libraries:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`

---

## 📈 Model Performance Summary

| Model             | Accuracy | Precision | Recall | F1-Score |
|-------------------|----------|-----------|--------|----------|
| **SVM**           | 0.85     | 0.88      | 0.79   | 0.83     |
| **Decision Tree** | 0.82     | 0.79      | 0.82   | 0.81     |
| **KNN**           | 0.85     | 0.85      | 0.82   | 0.84     |
| **Random Forest** | 0.85     | 0.88      | 0.79   | 0.83     |
| **Gradient Boosting** | 0.82 | 0.87      | 0.71   | 0.78     |
| **Voting Classifier** | 0.83 | 0.88      | 0.75   | 0.81     |

---

## 📊 Visualization
![Comparison of Model Performance Metrics](443357df-4287-41a6-a493-2958eda3de3d.png)

---

## 🚀 How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/heart-disease-prediction.git
   cd heart-disease-prediction

2. Install dependencies:
    pip install -r requirements.txt
3. Run the script:
    python heart_disease_prediction.py

---

## 🔮 Future Improvements
- Deploy with a **Flask/Django web interface**  
- Add **deep learning models** (ANN, CNN)  
- Use **SHAP/LIME** for model interpretability  
- Expand dataset for improved generalization  

---

## 📄 License
This project is licensed under the **MIT License**.

---

## 👤 Author
**Kadambala Eitish**  
📧 eitishkadambala@gmail.com 
🔗 [GitHub](https://github.com/Eitish24) | [LinkedIn](https://www.linkedin.com/in/kadambala-eitish0509/)

