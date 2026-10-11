import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("Iris.csv").drop(columns=["Id"], errors="ignore").dropna()
X, y = df.drop(columns=["Species"]), df["Species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=2000))])
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print(classification_report(y_test, pred))
print("Confusion matrix class order:", list(model.classes_))
print(confusion_matrix(y_test, pred))
sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=X.columns)
print("Example predicted species:", model.predict(sample)[0])
