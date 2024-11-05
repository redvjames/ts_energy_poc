import joblib

def load_model():
    model = joblib.load('model.pkl')
    return model