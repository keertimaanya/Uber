import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.calibration import calibration_curve

# Title
st.title("🚖 Uber Ride Cancellation Prediction")

# Generate synthetic dataset
st.write("📊 **Model Training & Data Overview**")
data_size = 90000
np.random.seed(42)

df = pd.DataFrame({
    'user_rating': np.random.uniform(1.0, 5.0, data_size),
    'driver_rating': np.random.uniform(1.0, 5.0, data_size),
    'time_of_day': np.random.choice(['morning', 'afternoon', 'evening', 'night'], data_size),
    'ride_demand': np.random.choice(['low', 'medium', 'high'], data_size),
    'past_cancellations': np.random.randint(0, 5, data_size),
    'cancellation': np.random.choice([0, 1], data_size, p=[0.7, 0.3])
})

# Encode categorical variables
df = pd.get_dummies(df, columns=['time_of_day', 'ride_demand'], drop_first=True)

# Split dataset
X = df.drop(columns=['cancellation'])
y = df['cancellation']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_prob = model.predict_proba(X_test)[:, 1]
y_prob = np.clip(y_prob, 0, 0.9)  # Restrict probabilities to 0-90%
y_pred = (y_prob >= 0.3).astype(int)  # Adjust threshold
accuracy = accuracy_score(y_test, y_pred)

# Show model accuracy
st.write(f"✅ **Model Accuracy: {accuracy * 100:.2f}%**")

# Feature Importance Plot
st.subheader("Feature Importance")
feature_importances = model.feature_importances_
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(X.columns, feature_importances)
plt.xticks(rotation=45, ha='right')
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance")
st.pyplot(fig)

# ROC Curve
st.subheader("ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
ax.plot([0, 1], [0, 1], linestyle='--', label='Random Model')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
st.pyplot(fig)

# User Input for Prediction
st.sidebar.header("Enter Ride Details:")
user_rating = st.sidebar.slider("User Rating", 1.0, 5.0, 4.5, step=0.1)
driver_rating = st.sidebar.slider("Driver Rating", 1.0, 5.0, 4.5, step=0.1)
time_of_day = st.sidebar.selectbox("Time of Day", ['morning', 'afternoon', 'evening', 'night'])
ride_demand = st.sidebar.selectbox("Ride Demand", ['low', 'medium', 'high'])
past_cancellations = st.sidebar.slider("Past Cancellations", 0, 5, 0)

# Convert categorical input
time_of_day_encoded = [int(time_of_day == 'afternoon'), int(time_of_day == 'evening'), int(time_of_day == 'night')]
ride_demand_encoded = [int(ride_demand == 'medium'), int(ride_demand == 'high')]

# Prepare input array
input_data = [user_rating, driver_rating, past_cancellations] + time_of_day_encoded + ride_demand_encoded
input_data = np.array(input_data).reshape(1, -1)

# Prediction
prediction = model.predict(input_data)[0]

# Show Result
if prediction == 1:
    st.error("❌ Ride is likely to be CANCELED!")
else:
    st.success("✅ Ride is likely to be COMPLETED!")

st.write("🔹 **Prediction is based on ML model analysis & historical trends.**")

# Run Streamlit
# streamlit run app.py
