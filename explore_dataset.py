
import pandas as pd
path = r"C:\Users\kurap\Documents\Medical\archive\DiseaseAndSymptoms.csv"
df = pd.read_csv(path)
print("✅ Dataset loaded!\n")
print("📊 Shape:", df.shape)
print("\n📄 Columns:", df.columns.tolist())
print("\n🔍 Sample Data:\n", df.head())
print("\n❓ Missing Values:\n", df.isnull().sum())
df.fillna("None", inplace=True)
symptom_columns = [f"Symptom_{i}" for i in range(1, 18)]
df["All_Symptoms"] = df[symptom_columns].apply( lambda x: ', '.join([symptom.strip() for symptom in x if symptom != "None"]), axis=1 )
print("\n✅ Preprocessing complete! Here's a preview:\n")
print(df[["Disease", "All_Symptoms"]].head())
df[["Disease", "All_Symptoms"]].to_csv("cleaned_disease_symptoms.csv", index=False)
print("\n📁 Cleaned data saved to cleaned_disease_symptoms.csv")