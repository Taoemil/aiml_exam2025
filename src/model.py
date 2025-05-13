import joblib
import numpy as np

# Load model from file
model = joblib.load("models/lr_cv_tuned.joblib")

def predict_toxicity(comment: str):
    """
    Run a comment through the BoW model and return prediction and confidence scores.
    """
    pred = model.predict([comment])[0]
    proba = model.predict_proba([comment])[0]
    
    label = "toxic" if pred == 1 else "non-toxic"
    confidence = np.max(proba)
    
    return label, confidence