import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the model
@st.cache_resource
def load_model():
    try:
        return joblib.load('svc.pkl')
    except:
        st.error("Error: Model file 'svc.pkl' not found. Please ensure the model file is in the same directory.")
        return None

def preprocess_input(data):
    """Preprocess the input data to match model's expected format"""
    # Convert breast quadrant to categorical
    quadrant_map = {
        'Upper inner': 0,
        'Upper outer': 1,
        'Lower inner': 2,
        'Lower outer': 3
    }
    
    data[3] = quadrant_map[data[3]]  # Convert breast quadrant to numeric
    return [data]

def main():
    st.set_page_config(
        page_title="Breast Cancer Prediction Tool",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title('🏥 Breast Cancer Prediction Tool')
    st.markdown("""
    This tool uses machine learning to predict breast cancer diagnosis based on patient data.
    Please note that this is a screening tool and should not replace professional medical diagnosis.
    """)

    # Create two columns for input
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Patient Information")
        age = st.number_input('Age', 
                            min_value=13, 
                            max_value=100, 
                            value=40,
                            help="Patient's age (13-100 years)")
        
        menopause = st.selectbox(
            'Menopause Status',
            options=[0, 1],
            format_func=lambda x: 'Yes' if x == 1 else 'No',
            help="Whether the patient has undergone menopause"
        )
        
        history = st.selectbox(
            'Family History of Breast Cancer',
            options=[0, 1],
            format_func=lambda x: 'Yes' if x == 1 else 'No',
            help="Whether there is a family history of breast cancer"
        )

    with col2:
        st.subheader("Medical Data")
        tumor_size = st.number_input('Tumor Size (cm)', 
                                   min_value=1, 
                                   max_value=15, 
                                   value=2,
                                   help="Size of the tumor in centimeters")
        
        breast_quadrant = st.selectbox(
            'Breast Quadrant',
            ['Upper inner', 'Upper outer', 'Lower inner', 'Lower outer'],
            help="Location of the tumor in the breast"
        )

    # Add a divider
    st.markdown("---")

    # Prediction button
    if st.button('Generate Prediction', type='primary'):
        model = load_model()
        
        if model:
            # Prepare input data
            input_data = [age, menopause, tumor_size, breast_quadrant, history]
            processed_input = preprocess_input(input_data)
            
            try:
                # Make prediction
                prediction = model.predict(processed_input)
                probability = model.predict_proba(processed_input)
                
                # Show results in a nice format
                st.subheader('Prediction Results')
                
                # Create columns for results
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    if prediction[0] == 'Malignant':
                        st.error('📊 Prediction: Malignant')
                    else:
                        st.success('📊 Prediction: Benign')
                
                with res_col2:
                    st.info(f'🎯 Confidence: {max(probability[0])*100:.2f}%')
                
                # Display warning
                st.warning("""
                ⚠️ **Important Notice:**
                - This is a screening tool only
                - Results should be verified by healthcare professionals
                - Medical decisions should not be based solely on these predictions
                """)
                
                # Display input summary
                st.subheader("Input Summary")
                summary_data = {
                    "Parameter": ["Age", "Menopause Status", "Tumor Size", "Breast Quadrant", "Family History"],
                    "Value": [
                        age,
                        "Yes" if menopause == 1 else "No",
                        f"{tumor_size} cm",
                        breast_quadrant,
                        "Yes" if history == 1 else "No"
                    ]
                }
                st.table(pd.DataFrame(summary_data))
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")

    # Add information about the model
    with st.expander("About the Model"):
        st.markdown("""
        ### Model Information
        - **Type**: Support Vector Machine (SVM)
        - **Accuracy**: 90%
        - **Features Used**: Age, Menopause Status, Tumor Size, Breast Quadrant, Family History
        - **Dataset**: UCTH Breast Cancer Dataset (213 records)
        
        ### Disclaimer
        This tool is for educational and research purposes only. Always consult with healthcare professionals for medical advice and diagnosis.
        """)

if __name__ == '__main__':
    main() 