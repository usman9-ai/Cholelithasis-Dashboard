import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from plotly import graph_objects as go

st.set_page_config(layout="wide")

# Load Dataset
def load_data():
    data = pd.read_excel('Model Training\colelithiasis_dataset.xlsx')  # Update with your dataset file path
    data.drop('Patient No.', axis=1, inplace=True)
    return data

# Initialize Session State
if "data" not in st.session_state:
    st.session_state.data = load_data()

def introduction_page():
    st.title("Introduction")
    st.markdown("""
    ## Project Overview
    This project analyzes the Colelithiasis dataset to perform exploratory data analysis (EDA) and prediction using pre-trained machine learning models. The goal is to provide insights into the data and make predictions efficiently.

    ## Objectives
    - Perform EDA to uncover patterns and insights.
    - Use pre-trained machine learning models for predictions.
    - Create an interactive Streamlit application.
    """)

def stats_page():
    st.title("Exploratory Data Analysis")

    # Dataset Overview
    st.subheader("Dataset Overview")
    st.dataframe(st.session_state.data.head())

    # Summary Statistics
    st.subheader("Summary Statistics")
    st.write(st.session_state.data.describe())

    # Correlation Matrix
    st.subheader("Correlation Analysis")

    # encode the target variable
    data = st.session_state.data.copy()
    data['Health_status'].replace({'healthy': 0, 'patient': 1}, inplace=True)

    # apply ordinal encoding to the categorical columns
    categorical_columns = ['Gender','Family history','Obese/non obese']
    encoder = joblib.load('Model Training\encoder.pkl')
    data[categorical_columns] = encoder.transform(data[categorical_columns])
    
    correlation = data.corr()
    plt.figure(figsize=(5, 3))
    # reduce the font size of the heatmap
    sns.set(font_scale=0.5)
    sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot(plt, use_container_width=False)

def eda_page():
    st.title("Exploratory Data Analysis")

    # Interactive Visualizations
    st.subheader("Visualizations")
    chart_type = st.selectbox("Choose Chart Type", ["Histogram", "Scatter Plot", "Box Plot"])

    if chart_type == "Histogram":
        column = st.selectbox("Choose Column for Visualization", st.session_state.data.columns)
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=st.session_state.data[column], name=column, marker_color="indigo"))
        fig.update_layout(
            title=dict(text="Histogram Analysis", x=0.5, font=dict(size=22)),
            xaxis_title=column,
            yaxis_title="Count",
            legend=dict(title="Legend", orientation="h", x=0.5, xanchor="center"),
            bargap=0.2,
            hovermode="x unified",
            template="plotly_dark"
        )
        st.plotly_chart(fig)

    elif chart_type == "Scatter Plot":
        x_col = st.selectbox("Choose X-axis Column", st.session_state.data.columns)
        y_col = st.selectbox("Choose Y-axis Column", st.session_state.data.columns)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=st.session_state.data[x_col],
            y=st.session_state.data[y_col],
            mode="markers",
            marker=dict(size=10, color="purple", line=dict(width=1, color="white")),
            name=f"{y_col} vs {x_col}"
        ))
        fig.update_layout(
            title=dict(text="Scatter Plot Analysis", x=0.5, font=dict(size=22)),
            xaxis_title=x_col,
            yaxis_title=y_col,
            legend=dict(title="Legend", orientation="h", x=0.5, xanchor="center"),
            hovermode="closest",
            template="plotly_dark"
        )
        st.plotly_chart(fig)

    elif chart_type == "Box Plot":
        column = st.selectbox("Choose Column for Visualization", st.session_state.data.columns)
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=st.session_state.data[column],
            name=column,
            boxmean="sd",
            marker_color="teal"
        ))
        fig.update_layout(
            title=dict(text="Boxplot Analysis", x=0.5, font=dict(size=22)),
            yaxis_title=column,
            legend=dict(title="Legend", orientation="h", x=0.5, xanchor="center"),
            hovermode="y",
            template="plotly_dark"
        )
        st.plotly_chart(fig)


def model_page():
    st.title("Model Evaluation")
    test_data = pd.read_excel(r'Model Training\test_data.xlsx')  


    # encode the target variable
    test_data['Health_status'].replace({'healthy': 0, 'patient': 1}, inplace=True)

    # apply ordinal encoding to the categorical columns
    categorical_columns = ['Gender','Family history','Obese/non obese']
    encoder = joblib.load('Model Training\encoder.pkl')

    X = test_data.drop( columns=['Health_status'])
    X[categorical_columns] = encoder.transform(X[categorical_columns])
    y = test_data['Health_status']

    # apply standard scalling to numberical features in X
    numerical_columns = [col_name for col_name in X.columns if col_name not in categorical_columns]
    scaler = joblib.load('Model Training\scaler.pkl')  
    X[numerical_columns] = scaler.transform(X[numerical_columns])

    # Model Selection
    st.text("Model Selection")
    model_choice = st.selectbox("Choose a Pre-trained Model", ["SVM - Linear", "SVM - Polynomial", "SVM - RBF",
                                                                 "Random Forest","Random Forest Boosted", "Logistic Regression", "GDA"])
 
    # Load pre-trained model
    model = None
    if model_choice == "SVM - Linear":
        model = joblib.load('Model Training\svm_model_linear.pkl')
    elif model_choice == "SVM - Polynomial":
        model = joblib.load('Model Training\svm_model_poly.pkl')
    elif model_choice == "SVM - RBF":
        model = joblib.load('Model Training\svm_model_rbf.pkl')
    elif model_choice == "Random Forest":
        model = joblib.load('Model Training\rf_model.pkl')
    elif model_choice == "Random Forest Boosted":
        model = joblib.load('Model Training\rf_boosted.pkl')
    elif model_choice == "Logistic Regression":
        model = joblib.load('Model Training\lr_model.pkl')
    elif model_choice == "GDA":
        model = joblib.load('Model Training\gda.pkl')


    if model:
        # Make Predictions
        y_pred = model.predict(X)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("### Predictions on the Test Data:")
            st.dataframe(pd.DataFrame({"Actual": y, "Predicted": y_pred}))

        with col2:
            st.subheader("Classification Report")
            report = classification_report(y, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose().reset_index()
            report_df.drop('support', axis=1, inplace=True)
            report_df.set_index(['index'], inplace=True)
            report_df.rename(index={'0.0': 'Negative', '1.0': 'Positive'}, inplace=True)
            report_df.iloc[report_df.index.get_loc('accuracy'), 0:2] = ''
            st.table(report_df)

        st.subheader("Confusion Matrix")
        conf_matrix = confusion_matrix(y, y_pred)
        # Generate text annotations for the confusion matrix
        text_annotations = np.array([[str(value) for value in row] for row in conf_matrix])

        col1, col2 = st.columns(2)
        with col1:
            # Create the heatmap using seaborn
            plt.figure(figsize=(3 , 3))
            sns.heatmap(conf_matrix, annot=text_annotations, fmt="", cmap="Blues", cbar=False, square=True)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Confusion Matrix")
            st.pyplot(plt)

 
def prediction_page():
    st.title("Get Your Diagnosis")
    st.subheader("Symptoms Entry Form")
    # Model Selection
    model_choice = st.selectbox("Choose a Pre-trained Model", ["SVM - Linear", "SVM - Polynomial", "SVM - RBF",
                                                                "Random Forest","Random Forest Boosted", "Logistic Regression", "GDA"])

    # Load pre-trained model
    model = None
    if model_choice == "SVM - Linear":
        model = joblib.load('Model Training\svm_model_linear.pkl')
    elif model_choice == "SVM - Polynomial":
        model = joblib.load('Model Training\svm_model_poly.pkl')
    elif model_choice == "SVM - RBF":
        model = joblib.load('Model Training\svm_model_rbf.pkl')
    elif model_choice == "Random Forest":
        model = joblib.load('Model Training\rf_model.pkl')
    elif model_choice == "Random Forest Boosted":
        model = joblib.load('Model Training\rf_boosted.pkl')
    elif model_choice == "Logistic Regression":
        model = joblib.load('Model Training\lr_model.pkl')
    elif model_choice == "GDA":
        model = joblib.load('Model Training\gda.pkl')

    with st.form(key="health_data_form"):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            # Categorical features with dropdown selection
            gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
            weight = st.number_input("Weight (kg)", min_value=0, step=1, key="weight")
            cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=0, step=1, key="cholesterol")
        with col2:
            family_history = st.selectbox("Family History of Illness", ["Yes", "No"], key="family_history")
            bmi = st.number_input("BMI", min_value=0.0, step=0.1, key="bmi")
            triglycerides = st.number_input("Triglycerides Level (mg/dL)", min_value=0, step=1, key="triglycerides")
            
        with col3:
            height = st.number_input("Height (cm)", min_value=0.0, step=0.1, key="height")
            obese_status = st.selectbox("Obese/Non Obese", ["Obese", "Non-Obese"], key="obese_status")
            ldl = st.number_input("LDL Level (mg/dL)", min_value=0.0, step=0.1, key="ldl")

        with col4:
            vldl = st.number_input("VLDL Level (mg/dL)", min_value=0.0, step=0.1, key="vldl")


        
        # Submit button
        submit_button = st.form_submit_button(label="Submit" )

    if submit_button:
        # Create a DataFrame directly with the user input data
        data = pd.DataFrame({
            "Gender": [gender],
            "Family history": [family_history],
            "Height": [height],
            "Weight": [weight],
            "BMI": [bmi],
            "Obese/non obese": [obese_status],
            "Cholesterol": [cholesterol],
            "Triglycerides": [triglycerides],
            "LDL level": [ldl],
            "VLDL level": [vldl]
        })
        

        columns = ['Gender', 'Family history', 'Height', 'Weight', 'BMI', 'Obese/non obese', 'Cholesterol', 'Triglycerides level', 'LDL level', 'VLDL level']
        data = data.reindex(columns=columns, fill_value=0)
        
        categorical_columns = ['Gender','Family history','Obese/non obese']
        numerical_columns = [col_name for col_name in data.columns if col_name not in categorical_columns]
        # Encoding categorical data
        encoder = joblib.load('Model Training\encoder.pkl')  
        data[categorical_columns] = encoder.transform(data[categorical_columns])

        # Scaling the numeric features
        scaler = joblib.load('Model Training\scaler.pkl')
        data[numerical_columns] = scaler.transform(data[numerical_columns])

    
        
        prediction = int(model.predict(data)[0])
        st.write(f"### Predicted Diagnosis: {'Positive' if prediction == 1 else 'Negative'}")
        

def conclusion_page():
    st.title("Conclusion")
    st.markdown("""
    ## Key Takeaways
    - Comprehensive EDA provides actionable insights into the data.
    - Pre-trained machine learning models allow efficient predictions.
    - The interactive app makes the analysis accessible and engaging.

    Thank you for exploring this project!
    """)

# Sidebar Navigation Menu with radio buttons for page selection
page = st.sidebar.radio("Navigation Menu", ["Introduction","Descriptive Statistics", "Data Analytics", "Model Evaluation", "Get Your Diagnosis", "Conclusion"])

if page == "Introduction":
    introduction_page()
elif page == "Descriptive Statistics":
    stats_page()
elif page == "Data Analytics":
    eda_page()
elif page == "Model Evaluation":
    model_page()
elif page == "Get Your Diagnosis":
    prediction_page()
elif page == "Conclusion":
    conclusion_page()
