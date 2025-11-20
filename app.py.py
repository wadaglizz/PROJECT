import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the trained model
model = joblib.load("student_model.pkl")

st.title("Student Dropout Prediction")

# Input sliders
attendance = st.slider("Attendance %", 0, 100, 80)
cat_score = st.slider("CAT Score", 0, 100, 65)
exam_score = st.slider("Exam Score", 0, 100, 70)

# Prepare input data
input_data = pd.DataFrame({
    "attendance": [attendance],
    "CAT": [cat_score],
    "exam": [exam_score]
})

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    st.subheader(f"Prediction: {'Dropout' if prediction == 1 else 'No Dropout'}")
    st.write(f"Probability of dropout: {proba[1] * 100:.2f}%")

    # Bar chart of inputs
    fig, ax = plt.subplots()
    sns.barplot(x=input_data.columns, y=input_data.loc[0], ax=ax)
    ax.set_ylim(0, 100)
    ax.set_title("Student Metrics")
    st.pyplot(fig)

    # Scatter plot
    fig2, ax2 = plt.subplots()
    ax2.scatter(input_data["CAT"], input_data["exam"], s=120)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)
    ax2.set_xlabel("CAT Score")
    ax2.set_ylabel("Exam Score")
    ax2.set_title("CAT vs Exam Score")
    st.pyplot(fig2)
