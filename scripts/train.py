from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import joblib

iris = load_iris()
X, y = iris.data, iris.target

model = SVC(max_iter = 100 , kernel='linear',random_state = 42)  # 'linear' kernel is similar to Logistic Regression behavior
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state = 42)
model.fit(X_train, y_train)

joblib.dump(model, "models/iris_model.pkl")
print("Model trained and saved.")