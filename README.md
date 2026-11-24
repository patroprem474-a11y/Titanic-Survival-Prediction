# 🚢 Titanic Survival Prediction

A Machine Learning project that predicts whether a Titanic passenger survived using Logistic Regression and an interactive Streamlit web application.

## 📌 Project Overview

This project analyzes the Titanic dataset and uses Machine Learning to predict passenger survival based on passenger information.

It includes data cleaning, preprocessing, model training, evaluation, and interactive predictions through a Streamlit dashboard.

## ✨ Features

- 📊 Dataset Overview and Analysis
- 🧹 Missing Value Handling
- 🔍 Duplicate Detection and Removal
- 🤖 Logistic Regression Model
- ⚖️ SMOTE for Class Imbalance Handling
- 📈 Model Evaluation and Accuracy
- 📉 Confusion Matrix and Classification Report
- 🔮 Interactive Passenger Survival Prediction
- 💾 Download Trained ML Model

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## 📂 Project Structure

```text
Titanic-Survival-Prediction/
│
├── app.py
├── titanic_model.py
├── titanic_dataset.csv
├── dataquality.ipynb
└── README.md
```

## ⚙️ Installation

**1. Clone the repository**

```bash
git clone https://github.com/patroprem474-a11y/Titanic-Survival-Prediction.git
```

**2. Open project folder**

```bash
cd Titanic-Survival-Prediction
```

**3. Install required libraries**

```bash
pip install streamlit pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib
```

## ▶️ Run the Application

```bash
streamlit run app.py
```

Open your browser at:

http://localhost:8501

Upload `titanic_dataset.csv` to start using the application.

## 🧠 Machine Learning Workflow

1. Load Titanic dataset.
2. Handle missing values.
3. Remove duplicate records.
4. Perform feature preprocessing.
5. Split data into training and testing sets.
6. Train Logistic Regression model.
7. Evaluate model performance.
8. Predict passenger survival.

## 🎯 Target Variable

The target column is `survived`.

- `0` = Did not survive
- `1` = Survived

## 📊 Model Evaluation

The application provides:

- Training Accuracy
- Testing Accuracy
- Confusion Matrix
- Classification Report
- Precision, Recall and F1-score

## 👨‍💻 Author

**Prem**

GitHub: [@patroprem474-a11y](https://github.com/patroprem474-a11y)

---

⭐ If you find this project useful, consider giving it a star!

