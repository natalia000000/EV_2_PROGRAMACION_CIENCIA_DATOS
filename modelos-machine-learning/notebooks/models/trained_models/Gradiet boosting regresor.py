import joblib

def save_model(model, path="models/trained_models/gradient_boosting.pkl"):
    joblib.dump(model, path)
    print(f"Modelo guardado en {path}")
    