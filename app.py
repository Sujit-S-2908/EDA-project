import streamlit as st
import pandas as pd
import numpy as np

def upload_file():
    uploaded_file = st.file_uploader("Choose a .csv file", type=["csv"])
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload a .csv file.")
        return None

def generate_profile(df):
    if df is not None:
        st.write("## Data Preview")
        st.write(df.head())
        
        st.write("## Basic Statistics")
        st.write(df.describe())
        
        st.write("## Missing Values")
        st.write(df.isnull().sum())
        
        st.write("## Data Types")
        st.write(df.dtypes)
    else:
        st.warning("No data available for profiling.")

def feature_engineering(df):
    # Filter to include only numerical columns
    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    st.write("Select a transformation:")
    transformation = st.selectbox("Transformation", ["Log Transform", "Normalization"])
    
    columns = st.multiselect("Select columns", numerical_columns)
    
    if st.button("Apply Transformation"):
        if columns:
            transformed_df = df.copy()
            if transformation == "Log Transform":
                for col in columns:
                    transformed_df[col] = transformed_df[col].apply(lambda x: np.log(x) if x > 0 else x)
            elif transformation == "Normalization":
                for col in columns:
                    transformed_df[col] = (transformed_df[col] - transformed_df[col].mean()) / transformed_df[col].std()
            st.write("Transformed Data:")
            st.write(transformed_df)
        else:
            st.warning("Please select columns to transform.")

def main():
    st.title("Data Profiling and Feature Engineering App")
    
    # Tabs for different sections
    tabs = ["Profile Report", "Feature Engineering"]
    selected_tab = st.sidebar.selectbox("Select a tab", tabs)
    
    if selected_tab == "Profile Report":
        st.header("Profile Report")
        df = upload_file()
        if st.button("Generate Report"):
            generate_profile(df)
            if df is not None:
                st.session_state['df'] = df
            
    elif selected_tab == "Feature Engineering":
        st.header("Feature Engineering")
        if 'df' in st.session_state:
            df = st.session_state['df']
            feature_engineering(df)
        else:
            st.warning("Please use the Profile Report screen first to upload data.")

if __name__ == "__main__":
    main()
