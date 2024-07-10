import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
        
        st.write("## Interactive Plots")
        numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Display charts for all numerical features
        for column in numerical_columns:
            st.write(f"### Visualization for {column}")
            fig, ax = plt.subplots(figsize=(6, 4))  # smaller figure size
            sns.histplot(df[column], kde=True, ax=ax)
            st.pyplot(fig)
            
            fig, ax = plt.subplots(figsize=(6, 4))  # smaller figure size
            sns.boxplot(x=df[column], ax=ax)
            st.pyplot(fig)
    else:
        st.warning("No data available for profiling.")

def feature_engineering(df):
    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    st.write("Select a transformation:")
    transformation = st.selectbox("Transformation", ["Log Transform", "Normalization"])
    
    selected_column = st.selectbox("Select a column for transformation", numerical_columns)
    
    if st.button("Apply Transformation") and selected_column:
        transformed_df = df.copy()
        if transformation == "Log Transform":
            transformed_df[selected_column] = transformed_df[selected_column].apply(lambda x: np.log(x) if x > 0 else x)
        elif transformation == "Normalization":
            transformed_df[selected_column] = (transformed_df[selected_column] - transformed_df[selected_column].mean()) / transformed_df[selected_column].std()
        
        st.write("Transformed Data:")
        st.write(transformed_df)
        
        st.write("## Interactive Plots for Transformed Data")
        st.write(f"### Histogram for {selected_column}")
        fig, ax = plt.subplots(figsize=(6, 4))  # smaller figure size
        sns.histplot(transformed_df[selected_column], kde=True, ax=ax)
        st.pyplot(fig)
        
        st.write(f"### Boxplot for {selected_column}")
        fig, ax = plt.subplots(figsize=(6, 4))  # smaller figure size
        sns.boxplot(x=transformed_df[selected_column], ax=ax)
        st.pyplot(fig)

def main():
    st.title("Data Profiling and Feature Engineering App")
    
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
