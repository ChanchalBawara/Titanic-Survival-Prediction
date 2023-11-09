import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
dataset = pd.read_csv("data/train_data.csv", encoding="latin-1")
dataset = dataset.rename(columns=lambda x: x.strip().lower())

# Data preprocessing
dataset = dataset[
    ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked", "survived"]
]
dataset["sex"] = dataset["sex"].map({"male": 0, "female": 1})
dataset["age"] = pd.to_numeric(dataset["age"], errors="coerce")
dataset["age"] = dataset["age"].fillna(np.mean(dataset["age"]))

# Dummy variables
embarked_dummies = pd.get_dummies(dataset["embarked"])
dataset = pd.concat([dataset, embarked_dummies], axis=1)
dataset = dataset.drop(["embarked"], axis=1)

X = dataset.drop(["survived"], axis=1)
y = dataset["survived"]

# Scaling features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Logistic Regression model
logistic_model = LogisticRegression(C=1)
logistic_model.fit(X_train, y_train)

# Model evaluation
y_pred = logistic_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save the model and scaler
import pickle

with open("model/titanic_survival_ml_model.sav", "wb") as model_file:
    pickle.dump(logistic_model, model_file)

with open("model/scaler.sav", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)
