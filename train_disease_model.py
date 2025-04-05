import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1️⃣ Load the cleaned dataset
df = pd.read_csv("cleaned_disease_symptoms.csv")

# 2️⃣ Convert 'All_Symptoms' string to list of symptoms
# This is required for multi-label binarization (like one-hot encoding for symptoms)
df["Symptom_List"] = df["All_Symptoms"].apply(lambda x: x.split(", "))

# 3️⃣ Use MultiLabelBinarizer to convert symptom lists into binary features
mlb = MultiLabelBinarizer()
X = pd.DataFrame(mlb.fit_transform(df["Symptom_List"]), columns=mlb.classes_)

# 4️⃣ Encode disease labels into numeric format
le = LabelEncoder()
y = le.fit_transform(df["Disease"])

# 5️⃣ Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6️⃣ Create and train the Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7️⃣ Predict on the test set
y_pred = model.predict(X_test)

# 8️⃣ Evaluate the model
print("✅ Model trained successfully!")
print(f"\n🎯 Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 9️⃣ Save the model and encoders for future use (like in an app or web interface)
joblib.dump(model, "disease_model.pkl")
joblib.dump(le, "disease_label_encoder.pkl")
joblib.dump(mlb, "symptom_encoder.pkl")
print("\n💾 Model and encoders saved to disk!")
