import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import streamlit as st

# Load the dataset
file_path = '/workspaces/ml--web-application-Streamlit/data/Diabetes.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
st.write("Dataset Preview:")
st.dataframe(df.head())

# Separate features and target variable
X = df.drop(columns='diabetes')
y = df['diabetes']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train the model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

st.write(f"Model Accuracy: {accuracy}")
st.write("Classification Report:\n", report)
st.write("Confusion Matrix:\n", conf_matrix)

# Save the model
joblib.dump(model, 'model.pkl')

# Streamlit app for user input and prediction
st.title("Diabetes Prediction App")
st.write("Enter the features to predict diabetes:")

# Create input fields for each feature based on your dataset
# Replace 'feature1', 'feature2', etc., with actual feature names
pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0, format="%.2f")
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.2f")
age = st.number_input("Age", min_value=0)

# Prediction
if st.button("Predict"):
    # Scale the input data
    user_data = scaler.transform([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
    
    # Make prediction
    prediction = model.predict(user_data)
    prediction_prob = model.predict_proba(user_data)[0][1]
    
    # Display result
    st.write("Prediction:", "Diabetes" if prediction[0] == 1 else "No Diabetes")
    st.write(f"Prediction Probability: {prediction_prob:.2f}")
