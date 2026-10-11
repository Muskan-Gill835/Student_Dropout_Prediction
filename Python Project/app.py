import streamlit as st
import joblib
import pandas as pd
import numpy as np


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Student Dropout Prediction",
    page_icon="🎓",
    layout="wide"
)
# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False


# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():

    model = joblib.load("student_dropout_model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("label_encoder.pkl")

    return model, scaler, encoder


model, scaler, encoder = load_model()

 



# ---------------- SIDEBAR ----------------

st.sidebar.title("🎓 Student Dropout System")

if st.session_state.logged_in:
    page = st.sidebar.radio(
        "Select Page",
        [
            "Home",
            "User Dashboard",
            "Prediction", 
            "Sign Out"
        ]
    )
else:
    page = st.sidebar.radio(
        "Select Page",
        [
            "Sign In"
        ]
    )

# ---------------- SIGN IN ----------------

if page == "Sign In":

    st.title("🔐 Student Login")

    st.subheader("Enter your credentials")

    username = st.text_input("Username")

    password = st.text_input(
        "Password",
        type="password"
    )

    if st.button("Login"):

        if username == "admin" and password == "1234":

            st.session_state.logged_in = True

            st.success("Login Successful ✅")

            st.balloons()

            st.rerun()

        else:

            st.error("Invalid Username or Password ❌")

# ---------------- HOME PAGE ----------------

if page == "Home":

    st.title("🎓 Student Dropout Prediction System")

    st.markdown("""
    ## Welcome 👋

    This Machine Learning based application predicts student academic outcome.

    The model predicts:

    - ⚠️ Dropout
    - 📚 Enrolled
    - 🎓 Graduate

    based on student's academic, financial and personal information.
    """)


    st.success(
        "Machine Learning Model: Random Forest"
    )

if not st.session_state.logged_in:
    st.warning("Please Login First")
    st.stop()

# ---------------- USER DASHBOARD ----------------

elif page == "User Dashboard":

    st.title("📊 User Dashboard")

    st.write("### 👋 Welcome to Student Dropout Prediction System")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="User",
            value="Admin"
        )

    with col2:
        st.metric(
            label="Login Status",
            value="Active ✅"
        )

    with col3:
        st.metric(
            label="Predictions",
            value="Ready"
        )

    st.markdown("---")

    st.subheader("📌 Dashboard")

    st.success("✔ Login Successful")

    st.info("Use the Prediction page to predict whether a student is likely to Dropout, Enrolled, or Graduate.")

    st.subheader("📝 Instructions")

    st.write("""
- Enter all required student details.
- Click **Predict** to generate the result.
- The system will display the predicted academic outcome.
- Use the **Sign Out** page to securely log out.
""")

# ---------------- PREDICTION PAGE ----------------

elif page == "Prediction":

    st.title("🔮 Student Dropout Prediction")

    st.write("Enter Student Details")

    col1, col2 = st.columns(2)


    with col1:

        age = st.number_input(
            "Age at Enrollment",
            min_value=17,
            max_value=70,
            value=20
        )


        admission_grade = st.number_input(
            "Admission Grade",
            min_value=0.0,
            max_value=200.0,
            value=120.0
        )


        previous_grade = st.number_input(
            "Previous Qualification Grade",
            min_value=0.0,
            max_value=200.0,
            value=120.0
        )


        debtor = st.selectbox(
            "Debtor",
            ["No", "Yes"]
        )

        debtor = 1 if debtor == "Yes" else 0



    with col2:


        tuition = st.selectbox(
            "Tuition Fees Up To Date",
            ["No", "Yes"]
        )

        tuition = 1 if tuition == "Yes" else 0



        scholarship = st.selectbox(
            "Scholarship Holder",
            ["No", "Yes"]
        )

        scholarship = 1 if scholarship == "Yes" else 0



        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        gender = 1 if gender == "Male" else 0



    if st.button("🚀 Predict"):


        input_data = np.array([
            age,
            admission_grade,
            previous_grade,
            debtor,
            tuition,
            scholarship,
            gender
        ]).reshape(1,-1)


        # Scaling
        input_scaled = scaler.transform(input_data)


        # Prediction
        prediction = model.predict(input_scaled)


        # Convert label back
        result = encoder.inverse_transform(prediction)[0]


        st.success(
            f"Prediction: {result}"
        )


        # Confidence
        if hasattr(model, "predict_proba"):

            probability = model.predict_proba(input_scaled)

            confidence = np.max(probability)*100

            st.info(
                f"Confidence: {confidence:.2f}%"
            )
 

# ---------------- SIGN OUT ----------------

elif page == "Sign Out":

    st.title("🚪 Logout")

    st.write("Click below to logout.")

    if st.button("Logout"):

        st.session_state.logged_in = False

        st.success("Logged Out Successfully")

        st.rerun()