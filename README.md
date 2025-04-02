
# Uber Ride Cancellation Prediction

This project predicts whether an Uber ride will be **canceled or completed** based on factors like user rating, driver rating, time of day, ride demand, and past cancellations. A **Random Forest Classifier** is trained for predictions, and an interactive **Streamlit web app** allows real-time testing.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Deployment on Streamlit Cloud](#deployment-on-streamlit-cloud)
- [Key Features](#key-features)
- [Visualizations](#visualizations)
- [Conclusion](#conclusion)
- [Contributing](#contributing)
- [Contact](#contact)

---

## Project Overview
The project analyzes Uber ride cancellations using **Machine Learning** and provides an **interactive web-based prediction tool**. The model learns from historical ride data and predicts if a ride will be canceled.

### Features:
- Predicts **ride cancellations** based on real-world-like factors.
- Provides **data insights** via visualizations.
- Web app built with **Streamlit** for real-time predictions.

---

## Tech Stack
- **Programming Language**: Python
- **Libraries Used**:
  - **Data Handling**: `pandas`, `numpy`
  - **Machine Learning**: `scikit-learn`
  - **Visualization**: `matplotlib`
  - **Web App Deployment**: `streamlit`
  - **Model Saving**: `joblib`

---

## Dataset
- **Number of Records**: 90,000 (synthetic)
- **Features:**
  - `user_rating` → Passenger's Uber rating (1.0 - 5.0)
  - `driver_rating` → Driver's Uber rating (1.0 - 5.0)
  - `time_of_day` → Time of the ride (`morning`, `afternoon`, `evening`, `night`)
  - `ride_demand` → Ride demand (`low`, `medium`, `high`)
  - `past_cancellations` → Number of past canceled rides
  - `cancellation` → **Target Variable** (0 = Not Canceled, 1 = Canceled)

---

## Model Performance
- **Algorithm Used**: Random Forest Classifier
- **Metrics**:
  - **Accuracy**: ~85%
  - **AUC-ROC Score**: ~0.88
- **Feature Importance Analysis**: Identifies key factors impacting ride cancellations.

---

## Installation
Follow these steps to set up the project locally:

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/uber-ride-cancellation.git
cd uber-ride-cancellation
```

### Step 2: Create & Activate Virtual Environment (Recommended)
#### On Windows:
```bash
python -m venv env
env\Scripts\activate
```

#### On Mac/Linux:
```bash
python3 -m venv env
source env/bin/activate
```

### Step 3: Install Dependencies
#### Using `requirements.txt`:
```bash
pip install -r requirements.txt
```

#### Or install manually:
```bash
pip install pandas numpy joblib scikit-learn matplotlib streamlit
```

---

## Usage
### Step 1: Run the Streamlit App
```bash
streamlit run app.py
```

### Step 2: Interact with the Web App
- Enter values for **user rating, driver rating, time of day, ride demand, and past cancellations**.
- Click **Predict** to see if the ride is likely to be canceled.

---

## Key Features
- **Predicts ride cancellations** using machine learning.
- **Interactive UI** built with Streamlit.
- **Feature importance analysis** for better insights.
- **Real-time data input** for live predictions.

---

## Visualizations
The project includes:
- **Feature Importance Graphs**
- **Calibration Curve**
- **ROC Curve**

---

## Conclusion
This project demonstrates the power of machine learning in predicting ride cancellations. By using historical data, we can make data-driven decisions to improve ride efficiency.

---
