# Credit Risk Prediction System

An end-to-end Machine Learning system to predict loan default risk using real-world financial data.

## 📥 Dataset

The raw dataset is too large to include in this repository.

👉 Download it from Kaggle:
https://www.kaggle.com/competitions/home-credit-default-risk/data

After downloading, place files in:
data/raw/

---

## 🚀 Features
- LightGBM-based model with strong performance on imbalanced data
- End-to-end ML pipeline using scikit-learn
- FastAPI backend for real-time predictions
- Streamlit UI for interactive demo
- SHAP-based model interpretability

---

## 📂 Project Structure

data/
  ├── raw/           # (not included due to size)
  ├── processed/     # cleaned dataset

models/
src/
api/
app/
notebooks/

---

## ⚙️ Installation

pip install -r requirements.txt

---

## ▶️ Run FastAPI

uvicorn api.app:app --reload

---

## ▶️ Run Streamlit

streamlit run app/streamlit_app.py

---

## 📊 Output
- Default probability
- Risk classification
- Risk level

---

## 📌 Tech Stack
- Python
- LightGBM
- Scikit-learn
- FastAPI
- Streamlit

---

## 👨‍💻 Author
Pratik Shukla