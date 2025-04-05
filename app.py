from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load models and encoders
model = joblib.load("disease_model.pkl")
label_encoder = joblib.load("disease_label_encoder.pkl")
symptom_encoder = joblib.load("symptom_encoder.pkl")

# Load the Disease precaution CSV
precaution_df = pd.read_csv(r"C:\Users\kurap\Documents\Medical\archive\Disease precaution.csv")

# Convert into a dictionary for quick lookup
precaution_dict = {}
for _, row in precaution_df.iterrows():
    disease = row["Disease"]
    precautions = [row[f"Precaution_{i}"] for i in range(1, 5) if pd.notnull(row[f"Precaution_{i}"])]
    precaution_dict[disease.strip().lower()] = precautions

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        symptoms = data.get("symptoms")

        if not symptoms or not isinstance(symptoms, list):
            return jsonify({"error": "Please provide a list of symptoms."}), 400

        input_vector = symptom_encoder.transform([symptoms])
        predicted_class_index = model.predict(input_vector)[0]
        predicted_disease = label_encoder.inverse_transform([predicted_class_index])[0]

        # Get precautions from dataset
        precautions = precaution_dict.get(predicted_disease.strip().lower(), ["No specific precautions found."])

        return jsonify({
            "prediction": predicted_disease,
            "precautions": precautions,
            "symptoms_received": symptoms
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
