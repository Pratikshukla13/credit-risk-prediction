import streamlit as st
import requests

st.set_page_config(page_title="Credit Risk Predictor", layout="wide")

st.title("💳 Credit Default Risk Predictor")

# ---------------- INPUTS ---------------- #

# 💰 Financial
col1, col2 = st.columns(2)
with col1:
    income = st.number_input("Income", value=150000)
    credit = st.number_input("Credit Amount", value=500000)
    annuity = st.number_input("Annuity", value=25000)
    goods_price = st.number_input("Goods Price", value=450000)

with col2:
    children = st.number_input("Children", value=1)
    birth = st.number_input("Days Birth (-ve)", value=-12000)
    employed = st.number_input("Days Employed (-ve)", value=-2000)

# 💼 Employment
income_type = st.selectbox("Income Type", ["Working", "Commercial associate", "Pensioner"])
occupation = st.text_input("Occupation", "Laborers")
organization = st.text_input("Organization", "Business Entity Type 3")

# 🏠 Assets
col3, col4 = st.columns(2)
with col3:
    car = st.selectbox("Own Car", ["Y", "N"])
    realty = st.selectbox("Own Realty", ["Y", "N"])
    housing = st.text_input("Housing Type", "House / apartment")

with col4:
    education = st.text_input("Education", "Secondary / secondary special")
    family = st.text_input("Family Status", "Married")
    gender = st.selectbox("Gender", ["M", "F"])

contract = st.selectbox("Contract Type", ["Cash loans", "Revolving loans"])

# 🌍 Region
region = st.number_input("Region Population", value=0.02)
registration = st.number_input("Days Registration", value=-4000)
id_publish = st.number_input("Days ID Publish", value=-3000)

# 📊 External Scores
ext1 = st.number_input("EXT_SOURCE_1", value=0.5)
ext2 = st.number_input("EXT_SOURCE_2", value=0.6)
ext3 = st.number_input("EXT_SOURCE_3", value=0.7)

# 📅 Process
weekday = st.selectbox("Weekday", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
hour = st.number_input("Hour", value=10)

# ---------------- PAYLOAD ---------------- #

payload = {
    "AMT_INCOME_TOTAL": income,
    "AMT_CREDIT": credit,
    "AMT_ANNUITY": annuity,
    "AMT_GOODS_PRICE": goods_price,
    "CNT_CHILDREN": children,
    "DAYS_BIRTH": birth,
    "DAYS_EMPLOYED": employed,
    "NAME_INCOME_TYPE": income_type,
    "OCCUPATION_TYPE": occupation,
    "ORGANIZATION_TYPE": organization,
    "FLAG_OWN_CAR": car,
    "FLAG_OWN_REALTY": realty,
    "NAME_HOUSING_TYPE": housing,
    "NAME_EDUCATION_TYPE": education,
    "NAME_FAMILY_STATUS": family,
    "CODE_GENDER": gender,
    "NAME_CONTRACT_TYPE": contract,
    "REGION_POPULATION_RELATIVE": region,
    "DAYS_REGISTRATION": registration,
    "DAYS_ID_PUBLISH": id_publish,
    "EXT_SOURCE_1": ext1,
    "EXT_SOURCE_2": ext2,
    "EXT_SOURCE_3": ext3,
    "WEEKDAY_APPR_PROCESS_START": weekday,
    "HOUR_APPR_PROCESS_START": hour
}

# ---------------- BUTTONS ---------------- #

col5, col6 = st.columns(2)

# 🔥 PREDICT
with col5:
    if st.button("🚀 Predict"):
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)

        if response.status_code == 200:
            st.success("Prediction Result")
            st.json(response.json())
        else:
            st.error("API Error")
            st.text(response.text)

# 🔥 EXPLAIN
with col6:
    if st.button("🔍 Explain Prediction"):
        response = requests.post("http://127.0.0.1:8000/explain", json=payload)

        if response.status_code == 200:
            st.success("SHAP Explanation")
            result = response.json()

            st.write("### Default Probability:", result["default_probability"])
            st.write("### Risk Level:", result["risk_level"])

            st.write("### 🔥 Top Contributing Features")

            for feature, details in result["top_features"].items():
                st.write(
                    f"**{feature}** → {details['impact']} ({details['effect']})"
                )
        else:
            st.error("API Error")
            st.text(response.text)