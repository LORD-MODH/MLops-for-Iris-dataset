from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

import joblib
model = joblib.load("models/iris_model.pkl")

iris = load_iris()

X, y = iris.data, iris.target
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

if accuracy < 0.9:
    raise ValueError("Model accuracy is below 90%.")