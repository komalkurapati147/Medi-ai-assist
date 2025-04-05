import pandas as pd
import joblib
from sklearn.preprocessing import MultiLabelBinarizer

# 🔄 Load the trained model and encoders
model = joblib.load("disease_model.pkl")
mlb = joblib.load("symptom_encoder.pkl")
label_encoder = joblib.load("disease_label_encoder.pkl")


# ✅ Function to predict disease from input symptoms
def predict_disease(symptoms_input):
    """
    Predict disease from a list of input symptoms.
    Args:
        symptoms_input (list): e.g., ["itching", "skin_rash", "nodal_skin_eruptions"]
    """
    # Preprocess: lowercase, strip, and remove extra spaces
    cleaned = [s.strip().lower() for s in symptoms_input]

    # 🔢 Encode the symptoms using the fitted MultiLabelBinarizer
    input_encoded = mlb.transform([cleaned])

    # 🔍 Predict the encoded label
    prediction_encoded = model.predict(input_encoded)

    # 🔤 Decode the label back to disease name
    predicted_disease = label_encoder.inverse_transform(prediction_encoded)[0]

    return predicted_disease

# 🧪 Example use
if __name__ == "__main__":
    # You can change these to test your own cases
    input_symptoms = ["itching", "skin_rash", "nodal_skin_eruptions"]
    result = predict_disease(input_symptoms)
    print(f"Predicted Disease: {result}")
