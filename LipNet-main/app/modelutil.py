import os
from model import build_lipnet_model

def load_model(model_type='overlapped'):
    model = build_lipnet_model()

    if model_type == 'overlapped':
        weight_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'overlapped-weights368.h5')
    elif model_type == 'unseen':
        weight_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'unseen-weights178.h5')
    else:
        raise ValueError("model_type must be 'overlapped' or 'unseen'")

    weight_path = os.path.abspath(weight_path)
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"Weight file not found: {weight_path}")

    model.load_weights(weight_path)
    return model