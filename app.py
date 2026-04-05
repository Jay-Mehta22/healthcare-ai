import streamlit as st
import joblib
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Healthcare AI",
    layout="centered"
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🩺 Healthcare AI")
st.sidebar.info("""
Multi-Disease Prediction System

✔ Diabetes Prediction  
✔ Heart Disease Prediction  

Built using Machine Learning
""")

# ---------------- LOAD MODELS ----------------
diabetes_model = joblib.load("diabetes_model.pkl")
heart_model = joblib.load("heart_model.pkl")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 20px;
}
.stTabs [data-baseweb="tab"] {
    background-color: #1c1f26;
    padding: 10px 20px;
    border-radius: 10px;
    color: white;
}
.stTabs [aria-selected="true"] {
    background-color: #ff4b4b;
    color: white;
}
.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🩺 Healthcare AI Predictor")
st.caption("Predict Diabetes & Heart Disease Risk Instantly")

# ---------------- TABS ----------------
tab1, tab2 = st.tabs(["🩸 Diabetes", "❤️ Heart Disease"])

# =========================================================
# ===================== DIABETES TAB ======================
# =========================================================
with tab1:
    st.subheader("Diabetes Prediction")

    # INFO SECTION
    with st.expander("ℹ️ About Inputs"):
        st.write("Pregnancies → Number of times pregnant")
        st.write("Glucose → Blood sugar level")
        st.write("BloodPressure → Resting BP")
        st.write("BMI → Body Mass Index")

    col1, col2 = st.columns(2)

    with col1:
        preg = st.number_input("Pregnancies", 0, 20, step=1)
        glucose = st.number_input("Glucose", 0, 300, step=1)
        bp = st.number_input("Blood Pressure", 0, 200, step=1)
        skin = st.number_input("Skin Thickness", 0, 100, step=1)

    with col2:
        insulin = st.number_input("Insulin", 0, 900, step=1)
        bmi = st.number_input("BMI", 0.0, 70.0, step=0.1)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, step=0.01)
        age = st.number_input("Age", 1, 120, step=1)

    if st.button("🔍 Predict Diabetes"):
        with st.spinner("Analyzing..."):
            input_df = pd.DataFrame([{
                "Pregnancies": preg,
                "Glucose": glucose,
                "BloodPressure": bp,
                "SkinThickness": skin,
                "Insulin": insulin,
                "BMI": bmi,
                "DiabetesPedigreeFunction": dpf,
                "Age": age
            }])

            prediction = diabetes_model.predict(input_df)[0]
            prob = diabetes_model.predict_proba(input_df)[0][1]

        st.markdown("### Result")

        if prediction == 1:
            st.error(f"⚠️ High Risk ({prob*100:.2f}%)")
            st.markdown("""
            ### 🩺 Recommendations:
            - Reduce sugar intake
            - Exercise regularly (30 mins daily)
            - Maintain healthy weight
            - Monitor blood sugar levels
            - Consult a doctor
            """)
        else:
            st.success(f"✅ Low Risk ({prob*100:.2f}%)")

# =========================================================
# ===================== HEART TAB =========================
# =========================================================
with tab2:
    st.subheader("Heart Disease Prediction")

    # INFO SECTION
    with st.expander("ℹ️ About Inputs"):
        st.write("cp → Chest Pain Type")
        st.write("thal → Blood disorder type")
        st.write("fbs → Fasting Blood Sugar")
        st.write("oldpeak → ST depression")

    col1, col2 = st.columns(2)

    with col1:
        age_h = st.number_input("Age", 1, 120, step=1, key="h_age")
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
        trestbps = st.number_input("Resting BP", 80, 200)

        chol = st.number_input("Cholesterol", 100, 600)
        fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])

    with col2:
        restecg = st.selectbox("Rest ECG", [0, 1, 2])
        thalach = st.number_input("Max Heart Rate", 60, 220)
        exang = st.selectbox("Exercise Induced Angina", [0, 1])
        oldpeak = st.number_input("Oldpeak", 0.0, 6.0, step=0.1)

        slope = st.selectbox("Slope", [0, 1, 2])
        ca = st.selectbox("Major Vessels", [0, 1, 2, 3])
        thal = st.selectbox("Thal", [0, 1, 2, 3])

    if st.button("🔍 Predict Heart Disease"):
        with st.spinner("Analyzing..."):
            input_df = pd.DataFrame([{
                "age": age_h,
                "sex": sex,
                "cp": cp,
                "trestbps": trestbps,
                "chol": chol,
                "fbs": fbs,
                "restecg": restecg,
                "thalach": thalach,
                "exang": exang,
                "oldpeak": oldpeak,
                "slope": slope,
                "ca": ca,
                "thal": thal
            }])

            prediction = heart_model.predict(input_df)[0]
            prob = heart_model.predict_proba(input_df)[0][1]

        st.markdown("### Result")

        if prediction == 1:
            st.error(f"⚠️ High Risk ({prob*100:.2f}%)")
            st.markdown("""
            ### ❤️ Recommendations:
            - Avoid high cholesterol food
            - Do regular cardio exercise
            - Reduce stress levels
            - Monitor blood pressure
            - Consult a cardiologist
            """)
        else:
            st.success(f"✅ Low Risk ({prob*100:.2f}%)")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("🚀 Built with Streamlit | Healthcare AI Project")