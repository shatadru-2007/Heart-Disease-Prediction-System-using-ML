import numpy as nmp
import pandas as pnd
import matplotlib.pyplot as mppp
import seaborn as sbn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from xgboost import XGBClassifier
import joblib

data = pnd.read_csv("heart.csv")
print("Dataset Shape:", data.shape)

x = data.drop("target", axis=1)
y = data["target"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

lr = LogisticRegression(max_iter=1000)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42
)

svm = SVC(
    kernel='rbf',
    C=10,
    gamma='scale',
    probability=True
)

xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric='logloss',
    random_state=42
)

model = VotingClassifier(
    estimators=[
        ('lr', lr),
        ('rf', rf),
        ('svm', svm),
        ('xgb', xgb)
    ],
    voting='soft'
)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("\nEnsemble (with XGBoost) Accuracy:",
      accuracy_score(y_test, y_pred) * 100, "%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

mppp.figure(figsize=(5,4))
sbn.heatmap(cm, annot=True, fmt='d', cmap='Blues')
mppp.title("Confusion Matrix - XGBoost Ensemble")
mppp.xlabel("Predicted")
mppp.ylabel("Actual")
mppp.show()

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully.")