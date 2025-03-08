import streamlit as st
import pandas as pd
import io
import joblib
import numpy as np

@st.cache_resource
def load_model():
    model = joblib.load(open('model/lr_model.pkl','rb'))
    return model


def give_predictions(df):
    columns_to_take = ['Age', 'Gender (1=Male, 2=Female, 3=Other)',
       'Length of Hospital Stay (days)', 'Fever (1=Yes, 0=No)',
       'Dysuria (1=Yes, 0=No)', 'Urinary Frequency (1=Yes, 0=No)',
       'Flank Pain (1=Yes, 0=No)', 'Hematuria (1=Yes, 0=No)',
       'History of UTI (last 6 months) (1=Yes, 0=No)',
       'History of Urological Procedure (last 6 months) (1=Yes, 0=No)',
       'Catheterization in Past 1 Month (1=Yes, 0=No)',
       'ICU Stay (last 6 months) (1=Yes, 0=No)', 'Diabetes (1=Yes, 0=No)',
       'Immunosuppression (1=Yes, 0=No)', 'SGLT2 Inhibitor Use (1=Yes, 0=No)',
       'Urine Protein (1=Yes, 0=No)', 'Urine Sugar (1=Yes, 0=No)',
       'Urine Pus Cells (/hpf)', 'Urine RBCs (/hpf)',
       'Urine Nitrates (1=Yes, 0=No)', 'Urine Bacteria (1=Yes, 0=No)',
       'Complicated UTI (1=Yes, 0=No)', 'Hospital Admission (1=Yes, 0=No)',
       'ICU Admission (1=Yes, 0=No)', 'Outcome (1=Recovered, 0=Death)']
    
    mapper =    {0: 'Enterococcus spp.',
                1: 'Escherichia spp.',
                2: 'Klebsiella spp.',
                3: 'Pseudomonas spp.',
                4: 'Serratia/Citrobacter spp.',
                5: 'Staphylococcus spp.',
                6: 'Unidentified'}
    
    df = df[columns_to_take]
    model = load_model()
    predictions_with_confidence = model.predict_proba(df)
    predictions = np.argmax(predictions_with_confidence,axis=1)
    confidence_scores = np.round(np.max(predictions_with_confidence,axis=1)*100,2)
    
    df['prediction'] = predictions
    df['prediction'] = df['prediction'].map(mapper)
    df['confidence'] = confidence_scores
    return df

def convert_df_to_csv(df):
    output = io.BytesIO()
    df.to_csv(output, index=False)
    return output.getvalue()

def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()

st.title("UTI AI")

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1]
    df = pd.read_csv(uploaded_file) if file_type == "csv" else pd.read_excel(uploaded_file)
    
    if st.button("Predict"):
        with st.spinner("Generating predictions....."):
            processed_df = give_predictions(df)
        
        if file_type == "csv":
            output_data = convert_df_to_csv(processed_df)
            mime_type = "text/csv"
            file_ext = "csv"
        else:
            output_data = convert_df_to_excel(processed_df)
            mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            file_ext = "xlsx"
        
        st.download_button(
            label="Download Processed File",
            data=output_data,
            file_name=f"processed_file.{file_ext}",
            mime=mime_type
        )
