from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

data = load_iris()
X, y = data.data, data.target

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "iris_model.pkl")
