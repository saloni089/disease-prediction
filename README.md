Disease Prediction Using Machine Learning

##  Overview
This project aims to predict diseases based on user-provided symptoms using various machine learning algorithms. A user-friendly web interface built with Streamlit allows users to input symptoms and receive predictions in real-time.

The model is trained on a structured dataset containing **132 symptoms** and around **41 possible diseases**, enabling reliable healthcare-related predictions.

---

## Technologies & Libraries Used

- **Python**
- `pandas` – data manipulation
- `numpy` – numerical operations
- `scikit-learn` – machine learning model building
- `streamlit` – to build interactive web interface
- `matplotlib` & `seaborn` – data visualization (optional)

📁 Dataset
The dataset was provided in CSV format by a training center mentor. It includes multiple rows of symptom combinations and the corresponding diagnosed disease.

---

## Machine Learning Models Used

- Random Forest Classifier 🌟 (Best performance)
- Support Vector Machine (SVM)
- Naive Bayes
- Decision Tree

###  Best Model:
- **Random Forest** gave the highest accuracy: **~95–97%**
- Achieved high **precision, recall, and F1-score** on test data.

---
##  Features

- Interactive **Streamlit UI** for disease prediction.
- Users can select symptoms via dropdowns and receive instant results.
- Compared performance of multiple ML algorithms.
- Clean, easy-to-understand visualizations (if used).

---

##  How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/disease-prediction.git
   cd disease-prediction

2. Install the required dependencies:

pip install -r requirements.txt

3. Run the Streamlit app:

streamlit run app.py
   
