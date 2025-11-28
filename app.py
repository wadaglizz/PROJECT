import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the trained model
model = joblib.load("student_model.pkl")

st.title("Student Dropout Prediction")

# --- Interactive Prediction Section ---
st.header("Predict Student Dropout")

# Input sliders
attendance = st.slider("Attendance %", 0, 100, 80)
cat_score = st.slider("CAT Score", 0, 100, 65)
exam_score = st.slider("Exam Score", 0, 100, 70)

# Prepare input for prediction
input_data = pd.DataFrame({
    "attendance": [attendance],
    "CAT": [cat_score],
    "exam": [exam_score]
})

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    st.subheader(f"Prediction: {'Dropout' if prediction == 1 else 'No Dropout'}")
    st.write(f"Probability of dropout: {proba[1]*100:.2f}%")

    # Bar chart of inputs
    fig, ax = plt.subplots()
    sns.barplot(x=input_data.columns, y=input_data.loc[0], palette="viridis", hue=input_data.columns, legend=False, ax=ax)
    ax.set_ylim(0, 100)
    ax.set_title("Student Metrics")
    st.pyplot(fig)

    # Scatter plot: CAT vs Exam
    fig2, ax2 = plt.subplots()
    ax2.scatter(input_data["CAT"], input_data["exam"], color='blue', s=100)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)
    ax2.set_xlabel("CAT Score")
    ax2.set_ylabel("Exam Score")
    ax2.set_title("CAT vs Exam Score")
    st.pyplot(fig2)

# --- Model Performance Section ---
st.header("Model Performance on Training Data")

# Recreate student_data for performance calculation
student_data_train = pd.DataFrame({
    'attendance': [80, 90, 70, 95, 60, 85, 75, 92, 68, 88],
    'CAT': [60, 70, 50, 80, 45, 65, 55, 75, 40, 72],
    'exam': [70, 80, 60, 85, 50, 72, 65, 88, 48, 78],
    'dropout': ['No', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No']
})

X_train = student_data_train[["attendance", "CAT", "exam"]]
y_train = student_data_train["dropout"]

# Map 'dropout' to numerical if it's 'object' type
if y_train.dtype == 'object':
    y_train = y_train.map({"No": 0, "Yes": 1})

y_pred_train = model.predict(X_train)

accuracy = accuracy_score(y_train, y_pred_train)
precision = precision_score(y_train, y_pred_train)
recall = recall_score(y_train, y_pred_train)
f1 = f1_score(y_train, y_pred_train)

st.write(f"**Accuracy:** {accuracy:.2f}")
st.write(f"**Precision:** {precision:.2f}")
st.write(f"**Recall:** {recall:.2f}")
st.write(f"**F1-Score:** {f1:.2f}")

st.warning("Note: These metrics are calculated on the training data and may not reflect performance on unseen data. Perfect scores often indicate potential overfitting given the small dataset.")

# --- Dashboard Explanation Section ---
st.header("How to Use This Dashboard")
st.markdown("""
This interactive dashboard allows you to predict the likelihood of student dropout based on key academic metrics.

**1. Adjust the Sliders:**
Use the sliders for 'Attendance %', 'CAT Score', and 'Exam Score' to input hypothetical student performance data. These values will instantly update the input for the prediction.

**2. Get Prediction:**
Click the 'Predict' button to see the model's prediction (Dropout or No Dropout) and the associated probability. The probability indicates the model's confidence in its prediction.

**3. Visualize Input:**
*   **Student Metrics Bar Chart:** Shows a bar chart of the input values you've selected, allowing for a quick overview of the student's profile.
*   **CAT vs Exam Score Scatter Plot:** Visualizes the relationship between the Continuous Assessment Test (CAT) score and the Exam score for the selected student.

**4. Model Performance:**
Below the interactive section, you'll find the model's performance metrics (Accuracy, Precision, Recall, F1-Score) calculated on the training data. This gives you an idea of how well the model performed on the data it was trained on.

This tool is designed to provide insights into potential student dropout and can be used to inform early intervention strategies.
""")
