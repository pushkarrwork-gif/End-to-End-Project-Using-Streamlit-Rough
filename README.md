# End-to-End Machine Learning Project: Diabetes Prediction App 🩺

Welcome to the End-to-End Diabetes Prediction project! This repository contains a fully functional rough machine learning web application built using **Python**, **Scikit-Learn**, and **Streamlit**. The application explores medical data and predicts whether a patient is likely to have diabetes based on specific diagnostic measurements.

## 📋 Project Overview

The primary goal of this project is to build an interactive end-to-end pipeline that takes raw medical data, processes it, trains multiple predictive models, and deploys the best-performing model to a user-friendly web interface.

The web app is divided into three functional pages:

1. **Main Page**: Displays raw dataset snapshots and statistical summaries.

2. **Page 1 (Visualizations)**: Features exploratory data analysis (EDA) with line and area charts mapping Age vs. BMI and Age vs. Insulin.

3. **Page 2 (Prediction Engine)**: An interactive form where users can input their medical metrics to receive a real-time diabetes prediction.

## 🛠️ Tech Stack & Libraries

* **Language:** Python 

* **Web Framework:** Streamlit (`app.py`)

* **Data Manipulation & Analysis:** Pandas, NumPy

* **Machine Learning:** Scikit-Learn (Support Vector Machine, Logistic Regression, Random Forest, StandardScaler)

* **Model Serialization:** Pickle (`pickle`)

## 📊 Data Analysis & Modeling Workflow

As a Data Analyst, ensuring data quality and selecting the right model is crucial. Here is the workflow executed in the background (referencing the Jupyter Notebook pipeline):

1. **Exploratory Data Analysis (EDA):**

   * Processed the Pima Indians Diabetes dataset (`diabetes.csv`) consisting of 768 records and 9 features.

   * Checked for missing values and observed class distributions (500 non-diabetic vs. 268 diabetic).

2. **Data Preprocessing:**

   * Separated independent variables (`X`) and the target variable (`Outcome`).

   * Applied `StandardScaler` to normalize features, ensuring that variables like Insulin (spanning up to 846) don't dominate metrics like Diabetes Pedigree Function.

   * Used a stratified 80-20 train-test split (`stratify=Y`) to maintain class proportions.

3. **Model Selection & Evaluation:**

   * **Logistic Regression:** Achieved \~78.6% training accuracy and \~75.9% test accuracy.

   * **Random Forest:** Showed severe overfitting (100% training accuracy but dropped to \~74.0% test accuracy).

   * **Support Vector Classifier (SVC):** Selected as the optimal model (using a `linear` kernel). It provided the best balance of variance and bias, achieving a **\~78.6% training accuracy** and **\~77.2% testing accuracy**.

4. **Deployment:**

   * Saved the SVC model (`model.pkl`) and the standard scaler (`std_scaler_model.pkl`) for inference in the Streamlit app.

## 🚀 How to Run the App Locally

To run this project on your local machine, follow these steps:

### 1. Clone the repository

```
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

```

### 2. Create a virtual environment (Recommended)

```
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

```

### 3. Install dependencies

```
pip install pandas numpy scikit-learn streamlit

```

*(Note: If you have a `requirements.txt`, you can run `pip install -r requirements.txt`)*

### 4. Run the Streamlit Application

```
streamlit run app.py

```

## 📁 Repository Structure

* `app.py`: The main Streamlit application code containing the UI and prediction logic.

* `model.pkl`: The serialized Support Vector Machine (SVC) trained model.

* `std_scaler_model.pkl`: The serialized StandardScaler object used to normalize user inputs.

* `diabetes.csv` & `diabetes_short.csv`: The datasets used for model training and app visual rendering.

* `notebook.ipynb` *(if uploaded)*: The Jupyter Notebook containing the initial data analysis, preprocessing, and model training steps.
