import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy.stats import boxcox
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from category_encoders import BinaryEncoder, TargetEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image 

def run_feature_engineering_analysis():

    hide_menu = """
    <style>
    #MainMenu {
        visibility: hidden;
    }
    footer {
        visibility: visible;
        text-align: center;
    }
    footer:after {
        content: "Copyright © 2023 Curated with ❤️ by Surya";
        display: block;
        position: relative;
        color: DarkGrey;
        margin: auto;
    }
    <style>
    """
    @st.cache
    def load_image(image_file):
        img = Image.open(image_file)
        return img 

    info = Image.open("Images/engg.png")
    '''st.set_page_config(
            page_title="Feature Engineering Analysis",
            page_icon=info,
            #layout="wide",
        )'''

    st.markdown(hide_menu, unsafe_allow_html=True)


    def min_max_scaling_and_visualization(data):
        st.write("#### 🔢 Min-Max Scaling")
        st.info("Performs Min-Max scaling on a selected numeric column with missing values and visualizes the data before and after scaling using Seaborn.")

        numeric_columns_with_null = [col for col in data.select_dtypes(include=['number']).columns]

        column_name = st.selectbox("Select a Numeric Column for Min-Max Scaling and Visualization:", numeric_columns_with_null)

        if column_name in data.columns:
            if pd.api.types.is_numeric_dtype(data[column_name]):
                st.write(f"**Selected Column:**  {column_name}")
                st.write('')

                col1, col2 = st.columns(2)  

                with col1:
                    st.write(f"**Before Min-Max Scaling of** {column_name}")
                    st.write(data[column_name].head())
                    st.write('')
                    plt.figure(figsize=(8, 4))
                    sns.histplot(data[column_name], kde=True, color='blue')
                    plt.title(f"Before Scaling: Distribution of {column_name}")
                    plt.xlabel(column_name)
                    plt.ylabel("Frequency")
                    st.pyplot(plt)

                scaler = MinMaxScaler()
                data[column_name] = scaler.fit_transform(data[[column_name]])

                with col2:
                    st.write(f"**After Min-Max Scaling of** {column_name}")
                    st.write(data[column_name].head())
                    st.write('')
                    plt.figure(figsize=(8, 4))
                    sns.histplot(data[column_name], kde=True, color='green')
                    plt.title(f"After Min-Max Scaling: Distribution of {column_name}")
                    plt.xlabel(column_name)
                    plt.ylabel("Frequency")
                    st.pyplot(plt)

                st.success("Min-Max scaling completed successfully.")
            else:
                st.warning(f"⚠️ The selected column '{column_name}' is not numeric.")
        else:
            st.warning(f"⚠️ The selected column '{column_name}' does not exist in the dataset.")

    def z_score_scaling_and_visualization(data):
        st.write("#### 🔢 Z-Score Scaling")
        st.info("Performs Z-Score scaling on a selected numeric column with missing values and visualizes the data before and after scaling using Seaborn.")

        numeric_columns_with_null = [col for col in data.select_dtypes(include=['number']).columns]

        column_name = st.selectbox("Select a Numeric Column for Z-Score Scaling and Visualization:", numeric_columns_with_null)

        if column_name in data.columns:
            if pd.api.types.is_numeric_dtype(data[column_name]):
                st.write(f"**Selected Column:**  {column_name}")
                st.write('')

                col1, col2 = st.columns(2)  

                with col1:
                    st.write(f"**Before Z-Score Scaling of** {column_name}")
                    st.write(data[column_name].head())
                    st.write('')
                    plt.figure(figsize=(8, 4))
                    sns.histplot(data[column_name], kde=True, color='blue')
                    plt.title(f"Before Scaling: Distribution of {column_name}")
                    plt.xlabel(column_name)
                    plt.ylabel("Frequency")
                    st.pyplot(plt)

                scaler = StandardScaler()
                data[column_name] = scaler.fit_transform(data[[column_name]])

                with col2:
                    st.write(f"**After Z-Score Scaling of** {column_name}")
                    st.write(data[column_name].head())
                    st.write('')
                    plt.figure(figsize=(8, 4))
                    sns.histplot(data[column_name], kde=True, color='green')
                    plt.title(f"After Z-Score Scaling: Distribution of {column_name}")
                    plt.xlabel(column_name)
                    plt.ylabel("Frequency")
                    st.pyplot(plt)

                st.success("Z-Score scaling completed successfully.")
            else:
                st.warning(f"⚠️ The selected column '{column_name}' is not numeric.")
        else:
            st.warning(f"⚠️ The selected column '{column_name}' does not exist in the dataset.")

    def robust_scaling_and_visualization(data):
        st.write("#### 🔢 Robust Scaling")
        st.info("Performs Robust scaling on a selected numeric column with missing values and visualizes the data before and after scaling using Seaborn.")

        numeric_columns_with_null = [col for col in data.select_dtypes(include=['number']).columns]

        column_name = st.selectbox("Select a Numeric Column for Robust Scaling and Visualization:", numeric_columns_with_null)

        if column_name in data.columns:
            if pd.api.types.is_numeric_dtype(data[column_name]):
                st.write(f"**Selected Column:**  {column_name}")
                st.write('')

                col1, col2 = st.columns(2)  

                with col1:
                    st.write(f"**Before Robust Scaling of** {column_name}")
                    st.write(data[column_name].head())
                    st.write('')
                    plt.figure(figsize=(8, 4))
                    sns.histplot(data[column_name], kde=True, color='blue')
                    plt.title(f"Before Scaling: Distribution of {column_name}")
                    plt.xlabel(column_name)
                    plt.ylabel("Frequency")
                    st.pyplot(plt)

                scaler = RobustScaler()
                data[column_name] = scaler.fit_transform(data[[column_name]])

                with col2:
                    st.write(f"**After Robust Scaling: Distribution of** {column_name}")
                    st.write(data[column_name].head())
                    st.write('')
                    plt.figure(figsize=(8, 4))
                    sns.histplot(data[column_name], kde=True, color='green')
                    plt.title(f"After Robust Scaling: Distribution of {column_name}")
                    plt.xlabel(column_name)
                    plt.ylabel("Frequency")
                    st.pyplot(plt)

                st.success("Robust scaling completed successfully.")
            else:
                st.warning(f"⚠️ The selected column '{column_name}' is not numeric.")
        else:
            st.warning(f"⚠️ The selected column '{column_name}' does not exist in the dataset.")

    def perform_basic_analysis(data):
        st.header("Scaling Analysis 🔍") 

        min_max_scaling_and_visualization(data)

        st.markdown("""---""")

        z_score_scaling_and_visualization(data)

        st.markdown("""---""")

        robust_scaling_and_visualization(data)

    def one_hot_encoding(data):
        st.write("#### 0️⃣1️⃣ One-Hot Encoding")
        st.info("Performs one-hot encoding on selected categorical columns and displays the encoded columns and encoded classes before and after encoding.")

        categorical_columns = data.select_dtypes(include=['object']).columns

        column_name = st.selectbox("Select a Categorical Column for One-Hot Encoding:", categorical_columns)

        if column_name in data.columns:
            st.write(f"**Selected Column:**  {column_name}")

            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Before One-Hot Encoding of** {column_name}")
                st.write(data[[column_name]].head())

            encoder = OneHotEncoder(sparse=False)
            encoded_data = encoder.fit_transform(data[[column_name]])
            encoded_columns = encoder.get_feature_names([column_name])

            encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns)

            with col2:
                st.write(f"**After One-Hot Encoding of** {column_name}")
                st.write(encoded_df.head())

            encoded_classes = encoder.get_feature_names_out([column_name])
            st.write(f"**Encoded Classes:**")
            st.write(encoded_classes)

            st.success("One-Hot encoding completed successfully.")
            
        else:
            st.warning(f"⚠️ The selected column '{column_name}' does not exist in the dataset.")

    def label_encoding(data):
        st.write("#### 0️⃣1️⃣ Label Encoding")
        st.info("Performs label encoding on selected categorical columns and displays the encoded columns and encoded classes before and after encoding.")

        categorical_columns = data.select_dtypes(include=['object']).columns

        column_name = st.selectbox("Select a Categorical Column for Label Encoding:", categorical_columns)

        if column_name in data.columns:
            st.write(f"**Selected Column:**  {column_name}")

            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Before Label Encoding of** {column_name}")
                st.write(data[[column_name]].head())

            encoder = LabelEncoder()
            data[column_name] = encoder.fit_transform(data[column_name])

            with col2:
                st.write(f"**After Label Encoding of** {column_name}")
                st.write(data[[column_name]].head())

            encoded_classes = encoder.classes_
            st.write(f"**Encoded Classes:**")
            st.write(encoded_classes)

            st.success("Label encoding completed successfully.")

        else:
            st.warning(f"⚠️ The selected column '{column_name}' does not exist in the dataset.")

    def binary_encoding(data):
        st.write("#### 0️⃣1️⃣ Binary Encoding")
        st.info("Performs binary encoding on selected categorical columns and displays the encoded columns and encoded classes before and after encoding.")

        categorical_columns = data.select_dtypes(include=['object']).columns

        column_name = st.selectbox("Select a Categorical Column for Binary Encoding:", categorical_columns)

        if column_name in data.columns:
            st.write(f"**Selected Column:**  {column_name}")

            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Before Binary Encoding of** {column_name}")
                st.write(data[[column_name]].head())

            encoder = BinaryEncoder()
            encoded_data = encoder.fit_transform(data[[column_name]])
            encoded_columns = encoded_data.columns

            encoded_df = pd.concat([data, encoded_data], axis=1)

            with col2:
                st.write(f"**After Binary Encoding of** {column_name}")
                st.write(encoded_df[encoded_columns].head())

            st.success("Binary encoding completed successfully.")

        else:
            st.warning(f"⚠️ The selected column '{column_name}' does not exist in the dataset.")

    def ordinal_encoding(data):
        st.write("#### 0️⃣1️⃣ Ordinal Encoding")
        st.info("Performs ordinal encoding on selected categorical columns and displays the encoded columns and encoded classes before and after encoding.")

        categorical_columns = data.select_dtypes(include=['object']).columns

        column_name = st.selectbox("Select a Categorical Column for Ordinal Encoding:", categorical_columns)

        if column_name in data.columns:
            st.write(f"**Selected Column:**  {column_name}")

            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Before Ordinal Encoding of** {column_name}")
                st.write(data[[column_name]].head())

            encoder = OrdinalEncoder()
            data[column_name] = encoder.fit_transform(data[[column_name]])

            with col2:
                st.write(f"**After Ordinal Encoding of** {column_name}")
                st.write(data[[column_name]].head())

            encoded_classes = encoder.categories_
            st.write(f"**Encoded Classes:**")
            st.write(encoded_classes)

            st.success("Ordinal encoding completed successfully.")

        else:
            st.warning(f"⚠️ The selected column '{column_name}' does not exist in the dataset.")

    def target_encoding(data):
        st.write("#### 0️⃣1️⃣ Target Encoding (Mean Encoding)")
        st.info("Performs target encoding (mean encoding) on selected categorical columns and displays the encoded columns and encoded classes before and after encoding.")

        categorical_columns = data.select_dtypes(include=['object']).columns

        column_name = st.selectbox("Select a Categorical Column for Target Encoding:", categorical_columns)

        target_column = st.selectbox("Select the Target Column for Target Encoding:", data.columns)

        if column_name in data.columns:
            st.write(f"**Selected Column:**  {column_name}")

            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Before Target Encoding of** {column_name}")
                st.write(data[[column_name]].head())

            encoder = TargetEncoder()
            data[column_name] = encoder.fit_transform(data[[column_name]], data[target_column])

            with col2:
                st.write(f"**After Target Encoding of** {column_name}")
                st.write(data[[column_name]].head())

            encoded_classes = encoder.get_params()
            st.write(f"**Encoded Classes:**")
            st.write(encoded_classes)

            st.success("Target encoding (mean encoding) completed successfully.")

        else:
            st.warning(f"⚠️ The selected column '{column_name}' does not exist in the dataset.")

    def perform_intermediate_analysis(data):
        st.header("Encoding Analysis 📈")

        one_hot_encoding(data)

        st.markdown("""---""")

        label_encoding(data)

        st.markdown("""---""")

        binary_encoding(data)

        st.markdown("""---""")

        ordinal_encoding(data)
        
        st.markdown("""---""")

        target_encoding(data)


    def log_transformation_and_visualization(data):
        st.write("#### 🔄 Log Transformation")
        st.info("Applies the natural logarithm to data and visualizes the data before and after transformation using scatter plots.")

        numeric_columns = data.select_dtypes(include=['number']).columns

        column_name = st.selectbox("Select a Numeric Column for Log Transformation and Visualization:", numeric_columns)

        if column_name in data.columns:
            if pd.api.types.is_numeric_dtype(data[column_name]):
                st.write(f"**Selected Column:**  {column_name}")

                col1, col2 = st.columns(2)  

                with col1:
                    st.write(f"**Before Log Transformation of** {column_name}")
                    st.write(data[column_name].head())
                    st.write('')
                    plt.figure(figsize=(8, 4))
                    plt.scatter(range(len(data[column_name])), data[column_name], color='blue', alpha=0.7)
                    plt.title(f"Before Transformation: Scatter Plot of {column_name}")
                    plt.xlabel("Index")
                    plt.ylabel(column_name)
                    st.pyplot(plt)

                data[column_name] = np.log1p(data[column_name])

                with col2:
                    st.write(f"**After Log Transformation of** {column_name}")
                    st.write(data[column_name].head())
                    st.write('')
                    plt.figure(figsize=(8, 4))
                    plt.scatter(range(len(data[column_name])), data[column_name], color='green', alpha=0.7)
                    plt.title(f"After Log Transformation: Scatter Plot of {column_name}")
                    plt.xlabel("Index")
                    plt.ylabel(column_name)
                    st.pyplot(plt)

                st.success("Log transformation completed successfully.")
            else:
                st.warning(f"⚠️ The selected column '{column_name}' is not numeric.")
        else:
            st.warning(f"⚠️ The selected column '{column_name}' does not exist in the dataset.")

    def boxcox_transformation_and_visualization(data):
        st.write("#### 🔄 Box-Cox Transformation")
        st.info("Applies Box-Cox transformation to data and visualizes the data before and after transformation using scatter plots.")

        numeric_columns = data.select_dtypes(include=['number']).columns

        column_name = st.selectbox("Select a Numeric Column for Box-Cox Transformation and Visualization:", numeric_columns)

        if column_name in data.columns:
            if pd.api.types.is_numeric_dtype(data[column_name]):
                st.write(f"**Selected Column:**  {column_name}")

                col1, col2 = st.columns(2)  

                with col1:
                    st.write(f"**Before Box-Cox Transformation of** {column_name}")
                    st.write(data[column_name].head())
                    st.write('')
                    plt.figure(figsize=(8, 4))
                    plt.scatter(range(len(data[column_name])), data[column_name], color='blue', alpha=0.7)
                    plt.title(f"Before Transformation: Scatter Plot of {column_name}")
                    plt.xlabel("Index")
                    plt.ylabel(column_name)
                    st.pyplot(plt)

                if (data[column_name] > 0).all():
                    data[column_name], _ = boxcox(data[column_name])

                    with col2:
                        st.write(f"**After Box-Cox Transformation of** {column_name}")
                        st.write(data[column_name].head())
                        st.write('')
                        plt.figure(figsize=(8, 4))
                        plt.scatter(range(len(data[column_name])), data[column_name], color='green', alpha=0.7)
                        plt.title(f"After Box-Cox Transformation: Scatter Plot of {column_name}")
                        plt.xlabel("Index")
                        plt.ylabel(column_name)
                        st.pyplot(plt)

                    st.success("Box-Cox transformation completed successfully.")
                else:
                    st.warning("⚠️ The selected column contains non-positive values. Box-Cox transformation requires data to be positive.")
            else:
                st.warning(f"⚠️ The selected column '{column_name}' is not numeric.")
        else:
            st.warning(f"⚠️ The selected column '{column_name}' does not exist in the dataset.")

    def sqrt_transformation_and_visualization(data):
        st.write("#### 🔄 Square Root Transformation")
        st.info("Applies square root transformation to data and visualizes the data before and after transformation using scatter plots.")

        numeric_columns = data.select_dtypes(include=['number']).columns

        column_name = st.selectbox("Select a Numeric Column for Square Root Transformation and Visualization:", numeric_columns)

        if column_name in data.columns:
            if pd.api.types.is_numeric_dtype(data[column_name]):
                st.write(f"**Selected Column:**  {column_name}")

                col1, col2 = st.columns(2)  

                with col1:
                    st.write(f"**Before Square Root Transformation of** {column_name}")
                    st.write(data[column_name].head())
                    st.write('')
                    plt.figure(figsize=(8, 4))
                    plt.scatter(range(len(data[column_name])), data[column_name], color='blue', alpha=0.7)
                    plt.title(f"Before Transformation: Scatter Plot of {column_name}")
                    plt.xlabel("Index")
                    plt.ylabel(column_name)
                    st.pyplot(plt)

                data[column_name] = np.sqrt(data[column_name])

                with col2:
                    st.write(f"**After Square Root Transformation of** {column_name}")
                    st.write(data[column_name].head())
                    st.write('')
                    plt.figure(figsize=(8, 4))
                    plt.scatter(range(len(data[column_name])), data[column_name], color='green', alpha=0.7)
                    plt.title(f"After Square Root Transformation: Scatter Plot of {column_name}")
                    plt.xlabel("Index")
                    plt.ylabel(column_name)
                    st.pyplot(plt)

                st.success("Square root transformation completed successfully.")
            else:
                st.warning(f"⚠️ The selected column '{column_name}' is not numeric.")
        else:
            st.warning(f"⚠️ The selected column '{column_name}' does not exist in the dataset.")

    def exp_transformation_and_visualization(data):
        st.write("#### 🔄 Exponential Transformation")
        st.info("Applies exponential transformation to data and visualizes the data before and after transformation using scatter plots.")

        numeric_columns = data.select_dtypes(include=['number']).columns

        column_name = st.selectbox("Select a Numeric Column for Exponential Transformation and Visualization:", numeric_columns)

        if column_name in data.columns:
            if pd.api.types.is_numeric_dtype(data[column_name]):
                st.write(f"**Selected Column:**  {column_name}")

                col1, col2 = st.columns(2)  

                with col1:
                    st.write(f"**Before Exponential Transformation of** {column_name}")
                    st.write(data[column_name].head())
                    st.write('')
                    plt.figure(figsize=(8, 4))
                    plt.scatter(range(len(data[column_name])), data[column_name], color='blue', alpha=0.7)
                    plt.title(f"Before Transformation: Scatter Plot of {column_name}")
                    plt.xlabel("Index")
                    plt.ylabel(column_name)
                    st.pyplot(plt)

                data[column_name] = np.exp(data[column_name])

                with col2:
                    st.write(f"**After Exponential Transformation of** {column_name}")
                    st.write(data[column_name].head())
                    st.write('')
                    plt.figure(figsize=(8, 4))
                    plt.scatter(range(len(data[column_name])), data[column_name], color='green', alpha=0.7)
                    plt.title(f"After Exponential Transformation: Scatter Plot of {column_name}")
                    plt.xlabel("Index")
                    plt.ylabel(column_name)
                    st.pyplot(plt)

                st.success("Exponential transformation completed successfully.")
            else:
                st.warning(f"⚠️ The selected column '{column_name}' is not numeric.")
        else:
            st.warning(f"⚠️ The selected column '{column_name}' does not exist in the dataset.")

    def perform_advanced_analysis(data):
        st.header("Tranformation Analysis 🚀")

        log_transformation_and_visualization(data)

        st.markdown("""---""")

        boxcox_transformation_and_visualization(data)

        st.markdown("""---""")

        sqrt_transformation_and_visualization(data)

        st.markdown("""---""")

        exp_transformation_and_visualization(data)
        
    def main():
        st.markdown("<h1 style='text-align: center;'>🧬 Feature Engineering Analysis on Dataset</h1>", unsafe_allow_html=True)
        st.write("")
        st.write("")

        uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.dataframe(data.head())
            st.success("✅ CSV file uploaded successfully")

            st.write('')

            if data.select_dtypes(include=[np.number]).empty:
                st.warning("⚠️ The uploaded dataset does not contain numerical columns.")
            else:
                with st.expander("🔍 Scaling Analysis"):
                    perform_basic_analysis(data)
                
                with st.expander("📈 Encoding Analysis"):
                    perform_intermediate_analysis(data)
                
                with st.expander("🚀 Transformation Analysis"):
                    perform_advanced_analysis(data)


    if __name__ == "__main__":
        main()
