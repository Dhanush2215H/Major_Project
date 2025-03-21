import os
import pickle
import joblib
import shap
import io
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import lime
import lime.lime_tabular
from sklearn.preprocessing import StandardScaler
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

    
# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# loading the saved models
diabetes_model = joblib.load('/Users/dhanushk/Python/Clg_Projects/Multi_disease_prediction/Saved_models/diabetes_model.joblib')

heart_disease_model = joblib.load('/Users/dhanushk/Python/Clg_Projects/Multi_disease_prediction/Saved_models/heart_model.joblib')

parkinsons_model = joblib.load('/Users/dhanushk/Python/Clg_Projects/Multi_disease_prediction/Saved_models/parkinsons_model.joblib')

# sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',

                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person'],
                           default_index=0)


# Diabetes Prediction Page
if selected == 'Diabetes Prediction':

    # page title
    st.title('Diabetes Prediction using ML')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')

    with col2:
        Glucose = st.text_input('Glucose Level')

    with col3:
        BloodPressure = st.text_input('Blood Pressure value')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value')

    with col2:
        Insulin = st.text_input('Insulin Level')

    with col3:
        BMI = st.text_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

    with col2:
        Age = st.text_input('Age of the Person')


    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction

    if st.button('Diabetes Test Result'):
        try:
            user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                        BMI, DiabetesPedigreeFunction, Age]

            user_input = [float(x) for x in user_input]

            diab_prediction = diabetes_model.predict([user_input])

            if diab_prediction[0] == 1:
                diab_diagnosis = 'The person is diabetic'
            else:
                diab_diagnosis = 'The person is not diabetic'
        except ValueError:
            st.error("Please enter valid numeric values.")

    st.success(diab_diagnosis)

    st.subheader("SHAP Feature Importance")

    with open("/Users/dhanushk/Python/Clg_Projects/Multi_disease_prediction/x_test_folder/Diabetes_X.pkl", "rb") as f:
        X = pickle.load(f)
    with open("/Users/dhanushk/Python/Clg_Projects/Multi_disease_prediction/x_test_folder/Diabetes_X_train.pkl", "rb") as f:
        X_train = pickle.load(f)
    with open("/Users/dhanushk/Python/Clg_Projects/Multi_disease_prediction/x_test_folder/Diabetes_X_test.pkl", "rb") as f:
        X_test = pickle.load(f)

    # Cache the SHAP computation to avoid recomputation
    @st.cache_data
    def compute_shap():
        explainer = shap.Explainer(diabetes_model, X_train)
        shap_values = explainer(X_test)
        return shap_values
    
    shap_values = compute_shap()

    # Generate SHAP summary plot
    fig, ax = plt.subplots(figsize=(9, 8))
    shap.summary_plot(shap_values, X_test, feature_names=X.columns, show=False)

    # Save the plot to a buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)

    # Display in Streamlit
    st.image(buffer, caption="SHAP Summary Plot", use_column_width=True)

    # Close the figure
    plt.close(fig)
    

    # LIME Explanation Section
    st.subheader("LIME Explanation for Your Input")
    
    if diab_diagnosis: 

        with open("/Users/dhanushk/Python/Clg_Projects/Multi_disease_prediction/x_test_folder/Diabetes_X_train.pkl", "rb") as f:
            x_train = pickle.load(f)

        feature_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
        # Create LIME explainer
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            np.array(x_train),  # Convert DataFrame to NumPy array
            feature_names=feature_names, 
            class_names=['Not Diabetic', 'Diabetic'], 
            discretize_continuous=True)

        # Convert user input to NumPy array
        user_input_np = np.array(user_input).reshape(1, -1)

        # Generate LIME explanation (Using `.decision_function()` for SVM)
        lime_exp = lime_explainer.explain_instance(
            user_input_np[0], diabetes_model.predict_proba, num_features=8
        )

        # Create LIME explanation plot
        fig, ax = plt.subplots(figsize=(8, 4))

        # Extract explanation values
        lime_values = lime_exp.as_list()
        features = [x[0] for x in lime_values]
        values = [x[1] for x in lime_values]

        # Use color coding: Blue (negative influence), Orange (positive influence)
        colors = ["blue" if val < 0 else "orange" for val in values]

        ax.barh(features, values, color=colors)
        ax.set_xlabel("Contribution to Prediction")
        ax.set_title("Local Explanation for Diabetes Prediction")

        # Show LIME plot in Streamlit
        st.pyplot(fig)

    

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':

    # page title
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')

    with col2:
        sex = st.text_input('Sex')

    with col3:
        cp = st.text_input('Chest Pain types')

    with col1:
        trestbps = st.text_input('Resting Blood Pressure')

    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')

    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')

    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')

    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')

    with col3:
        exang = st.text_input('Exercise Induced Angina')

    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')

    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')

    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')

    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction

    if st.button('Heart Disease Test Result'):

        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

        user_input = [float(x) for x in user_input]

        heart_prediction = heart_disease_model.predict([user_input])

        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'

    st.success(heart_diagnosis)

    # SHAP Explanation Section
    st.subheader("Feature Importance using SHAP")

    with open("/Users/dhanushk/Python/Clg_Projects/Multi_disease_prediction/x_test_folder/Heart_X_train_scaled.pkl", "rb") as f:
        X_train_scaled = pickle.load(f)
    with open("/Users/dhanushk/Python/Clg_Projects/Multi_disease_prediction/x_test_folder/Heart_X_test_scaled.pkl", "rb") as f:
        X_test_scaled = pickle.load(f)
    with open("/Users/dhanushk/Python/Clg_Projects/Multi_disease_prediction/x_test_folder/Heart_X.pkl", "rb") as f:
        X = pickle.load(f)

    @st.cache_data
    def compute_shap():
        #Generate SHAP values for test data (or a sample if large)
        background_data = shap.sample(X_train_scaled, 100)  # Take 100 random samples
        explainer = shap.KernelExplainer(heart_disease_model.decision_function, background_data)
        # Compute SHAP values for a sample of test data
        shap_values = explainer.shap_values(X_test_scaled[:100])
        return shap_values
    
    shap_values = compute_shap()
    # Generate SHAP summary plot
    fig, ax = plt.subplots(figsize=(9, 8))
    shap.summary_plot(shap_values, X_test_scaled[:100], feature_names=X.columns, show=False)

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)

    # Display in Streamlit
    st.image(buffer, caption="SHAP Summary Plot", use_column_width=True)

    # Close the figure
    plt.close(fig)

    # **LIME Explanation Section**
    st.subheader("LIME Explanation for Your Input")
    if heart_diagnosis:
        # Load dataset for LIME
        with open("/Users/dhanushk/Python/Clg_Projects/Multi_disease_prediction/x_test_folder/Heart_X_train_scaled.pkl", "rb") as f:
            X_train_scaled = pickle.load(f)
        with open("/Users/dhanushk/Python/Clg_Projects/Multi_disease_prediction/x_test_folder/Heart_X.pkl", "rb") as f:
            X = pickle.load(f)

        # LIME Explainer
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array(X_train_scaled),
            feature_names=X.columns.tolist(),
            class_names=['No Disease', 'Disease'],
            mode='classification'
        )

        # Generate LIME explanation for user input
        lime_exp = lime_explainer.explain_instance(
            np.array(user_input),
            heart_disease_model.predict_proba,
            num_features=5
        )

        # **Create LIME plot**
        fig, ax = plt.subplots(figsize=(8, 4))

        lime_values = lime_exp.as_list()
        features = [x[0] for x in lime_values]
        values = [x[1] for x in lime_values]

        colors = ["blue" if val < 0 else "orange" for val in values]
        ax.barh(features, values, color=colors)
        ax.set_xlabel("Contribution to Prediction")
        ax.set_title("LIME Explanation for Heart Disease Prediction")

        # Display LIME plot in Streamlit
        st.pyplot(fig)


# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":

    # page title
    st.title("Parkinson's Disease Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')

    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')

    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')

    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')

    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

    with col1:
        RAP = st.text_input('MDVP:RAP')

    with col2:
        PPQ = st.text_input('MDVP:PPQ')

    with col3:
        DDP = st.text_input('Jitter:DDP')

    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')

    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')

    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')

    with col3:
        APQ = st.text_input('MDVP:APQ')

    with col4:
        DDA = st.text_input('Shimmer:DDA')

    with col5:
        NHR = st.text_input('NHR')

    with col1:
        HNR = st.text_input('HNR')

    with col2:
        RPDE = st.text_input('RPDE')

    with col3:
        DFA = st.text_input('DFA')

    with col4:
        spread1 = st.text_input('spread1')

    with col5:
        spread2 = st.text_input('spread2')

    with col1:
        D2 = st.text_input('D2')

    with col2:
        PPE = st.text_input('PPE')

    # code for Prediction
    parkinsons_diagnosis = ''

    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):

        user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP,Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

        user_input = [float(x) for x in user_input]

        parkinsons_prediction = parkinsons_model.predict([user_input])

        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"

    st.success(parkinsons_diagnosis)


    # SHAP Explanation Section
    st.subheader("Feature Importance using SHAP")


    with open("/Users/dhanushk/Python/Clg_Projects/Multi_disease_prediction/x_test_folder/parkinsons_x_test.pkl", "rb") as f:
        x_test = pickle.load(f)
    with open("/Users/dhanushk/Python/Clg_Projects/Multi_disease_prediction/Feature_names_folder/parkinsons_feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    

    # Cache the SHAP computation to avoid recomputation
    @st.cache_data
    def compute_shap():
        # Generate SHAP values for test data (or a sample if large)
        parkinsons_explainer = shap.TreeExplainer(parkinsons_model)
        shap_values = parkinsons_explainer.shap_values(x_test)
        return shap_values
    
    shap_values = compute_shap()

    # Create SHAP summary plot and save it as an image buffer
    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(shap_values, x_test, feature_names, show=False)
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    st.image(buffer, caption="SHAP Summary Plot", use_column_width=True)

    plt.close(fig)


    # LIME Explanation Section
    st.subheader("LIME Explanation for Your Input")

    # Load train data for LIME explainer
    with open("/Users/dhanushk/Python/Clg_Projects/Multi_disease_prediction/x_test_folder/parkinsons_x_train.pkl", "rb") as f:
        x_train = pickle.load(f)

    # Create LIME explainer
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(x_train, feature_names=feature_names, class_names=['No Parkinsons', 'Parkinsons'], discretize_continuous=True)
    # Only generate LIME explanation if a prediction has been made
    if parkinsons_diagnosis:
        # Generate LIME explanation using user input instead of a test sample
        user_input_np = np.array(user_input).reshape(1, -1)
        lime_exp = lime_explainer.explain_instance(
            user_input_np[0], parkinsons_model.predict_proba, num_features=5
        )

        # Create a LIME explanation plot similar to Colab
        fig, ax = plt.subplots(figsize=(8, 4))

        # Extract explanation values
        lime_values = lime_exp.as_list()
        features = [x[0] for x in lime_values]
        values = [x[1] for x in lime_values]

        # Use blue color for consistency with Colab
        colors = ["blue" if val < 0 else "orange" for val in values]

        ax.barh(features, values, color=colors)
        ax.set_xlabel("Contribution to Prediction")
        ax.set_title("Local Explanation for Parkinson’s Prediction")

        # Show LIME plot in Streamlit
        st.pyplot(fig)