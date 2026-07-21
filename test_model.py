import joblib

model = joblib.load("models/titanic_svm_model.pkl")

print(type(model))
print(hasattr(model, "predict_proba"))