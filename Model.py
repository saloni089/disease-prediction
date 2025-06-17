# 1. Libraries import
import pandas as pd
import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier,_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 2. Data loading function
df = pd.read_csv("/kaggle/input/disease-prediction/Training.csv")
print(df.head())

# 3. Preprocessing function
unique_diseases = df['prognosis'].nunique()
print(f"Total unique diseases: {unique_diseases}")

#List all unique diseases
disease_list = df['prognosis'].unique()
print("List of diseases:\n", disease_list)

#Count occurrences of each disease
disease_counts = df['prognosis'].value_counts()

#Display the count for each disease
print("Number of people per disease:\n", disease_counts)

# 4. Training function
x = df.iloc[:,:-1]
y = df.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 24)
print(f"Train: {x_train.shape}, {y_train.shape}")
print(f"Test:{x_test.shape}, {y_test.shape}")

#initializing models
models = {
    "SVC":SVC(),
    "Gaussian NB":GaussianNB(),
    "Random Forest":RandomForestClassifier(random_state=18)
}
# Defining scoring metric for k-fold cross validation
def cv_scoring(estimator, x, y):
    return accuracy_score(y, estimator.predict(x))
