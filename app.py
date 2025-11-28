%%writefile app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

# --- Helper function to ensure model and data exist ---
def ensure_model_and_data():
    if not os.path.exists("student_model.pkl") or not os.path.exists("student_sample.csv"):
        st.warning("Model or data not found. Re-creating them...")
        student_data = pd.DataFrame({
            'attendance': [80, 90, 70, 95, 60, 85, 75, 92, 68, 88],
            'CAT': [60, 70, 50, 80, 45, 65, 55, 75, 40, 72],
            'exam': [70, 80, 60, 85, 50, 72, 65, 88, 48, 78],
            'dropout': ['No', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No']
        })
        student_data.to_csv("student_sample.csv", index=False)

        X = student_data[["attendance", "CAT", "exam"]]
        y = student_data["dropout"].map({"No": 0, "Yes": 1})

        model_to_save = RandomForestClassifier(n_estimators=100, random_state=42)
        model_to_save.fit(X, y)
        joblib.dump(model_to_save, "student_model.pkl")
        st.success("Model and data re-created successfully!")

ensure_model_and_data()

# Load the trained model
model = joblib.load("student_model.pkl")

st.set_page_config(page_title="Student Dropout Prediction Dashboard", layout="wide")

st.title("Student Dropout Prediction Dashboard")

# --- Sidebar for navigation ---
st.sidebar.title("Navigation")
selection = st.sidebar.radio(
    "Go to",
    ['Interactive Prediction', 'EDA and Data Analysis', 'Model Performance Evaluation']
)

# --- Section: Interactive Prediction ---
if selection == 'Interactive Prediction':
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
    if st.button("Predict"):n        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]

        st.subheader(f"Prediction: {'Dropout' if prediction == 1 else 'No Dropout'}")
        st.write(f"Probability of dropout: {proba[1]*100:.2f}%")

        # Bar chart of inputs
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(x=input_data.columns, y=input_data.loc[0], palette="viridis", hue=input_data.columns, legend=False, ax=ax)
        ax.set_ylim(0, 100)
        ax.set_title("Student Metrics")
        st.pyplot(fig)

        # Scatter plot: CAT vs Exam
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.scatter(input_data["CAT"], input_data["exam"], color='blue', s=100)
        ax2.set_xlim(0, 100)
        ax2.set_ylim(0, 100)
        ax2.set_xlabel("CAT Score")
        ax2.set_ylabel("Exam Score")
        ax2.set_title("CAT vs Exam Score")
        st.pyplot(fig2)

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

    **4. Model Performance (View in 'Model Performance Evaluation' Section):**
    You can navigate to the 'Model Performance Evaluation' section using the sidebar to see detailed metrics and plots about how well the model performed on the data it was trained on.

    This tool is designed to provide insights into potential student dropout and can be used to inform early intervention strategies.
    """)

# --- Section: EDA and Data Analysis ---
elif selection == 'EDA and Data Analysis':
    st.header("Exploratory Data Analysis (EDA)")

    # Load student_sample.csv
    try:
        df_eda = pd.read_csv("student_sample.csv")
    except FileNotFoundError:
        st.error("student_sample.csv not found. Please ensure it's in the same directory as app.py.")
        st.stop()

    st.subheader("Descriptive Statistics")
    st.write(df_eda.describe())

    # Histogram of 'attendance'
    st.subheader("Distribution of Attendance")
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.histplot(df_eda['attendance'], bins=5, kde=True, ax=ax3)
    ax3.set_title('Histogram of Attendance')
    ax3.set_xlabel('Attendance %')
    ax3.set_ylabel('Frequency')
    st.pyplot(fig3)

    # Count plot of 'dropout'
    st.subheader("Dropout Count")
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    sns.countplot(x='dropout', data=df_eda, palette='coolwarm', ax=ax4)
    ax4.set_title('Count of Dropout vs. No Dropout')
    ax4.set_xlabel('Dropout Status')
    ax4.set_ylabel('Count')
    st.pyplot(fig4)

# --- Section: Model Performance Evaluation ---
elif selection == 'Model Performance Evaluation':
    st.header("Model Performance Evaluation on Training Data")

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
    y_proba_train = model.predict_proba(X_train)[:, 1]

    # Metrics
    st.subheader("Classification Metrics")
    accuracy = accuracy_score(y_train, y_pred_train)
    precision = precision_score(y_train, y_pred_train)
    recall = recall_score(y_train, y_pred_train)
    f1 = f1_score(y_train, y_pred_train)

    st.write(f"**Accuracy:** {accuracy:.2f}")
    st.write(f"**Precision:** {precision:.2f}")
    st.write(f"**Recall:** {recall:.2f}")
    st.write(f"**F1-Score:** {f1:.2f}")

    st.warning("Note: These metrics are calculated on the training data and may not reflect performance on unseen data. Perfect scores often indicate potential overfitting given the small dataset.")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_train, y_pred_train)
    fig5, ax5 = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Dropout', 'Dropout'],
                yticklabels=['No Dropout', 'Dropout'], ax=ax5)
    ax5.set_title('Confusion Matrix')
    ax5.set_xlabel('Predicted')
    ax5.set_ylabel('Actual')
    st.pyplot(fig5)

    # ROC Curve
    st.subheader("Receiver Operating Characteristic (ROC) Curve")
    fpr, tpr, thresholds = roc_curve(y_train, y_proba_train)
    roc_auc = auc(fpr, tpr)

    fig6, ax6 = plt.subplots(figsize=(7, 6))
    ax6.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax6.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax6.set_xlim([0.0, 1.0])
    ax6.set_ylim([0.0, 1.05])
    ax6.set_xlabel('False Positive Rate')
    ax6.set_ylabel('True Positive Rate')
    ax6.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax6.legend(loc='lower right')
    st.pyplot(fig6)
kamau
