import streamlit as st
import pandas as pd
from PIL import Image 

def run_info_analysis():

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

    info = Image.open("Images/info.png")
    '''st.set_page_config(
            page_title="Infographics",
            page_icon=info,
            #layout="wide",
        )'''


    st.markdown(hide_menu, unsafe_allow_html=True)



    st.title("📋 Infographics on Dataset")

    st.write("")
    st.write("")

    uploaded_files = st.file_uploader("📂 Upload CSV files", type="csv", accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            expander = st.expander(f"Summary for Uploaded File: {uploaded_file.name}")
            with expander:
                df = pd.read_csv(uploaded_file)

                if uploaded_file.type != 'text/csv':
                    st.warning("⚠️ Please upload a CSV file format.")
                    continue

                st.header("📋 Basic Understanding of the Dataset")
                st.write(df.head())

                st.markdown("""---""")

                num_columns = df.shape[1]
                num_rows = df.shape[0]
                st.header('🧾 Number of Columns and Rows')
                st.write("##### 🔢 Number of Columns: ")
                st.write(num_columns)
                st.write("##### 🔢 Number of Rows: ")
                st.write(num_rows)

                st.markdown("""---""")

                st.header("🎯 Select Target Column")
                target_column = st.selectbox("📌 Choose the target column:", df.columns)
                st.write('##### The Target Column of the dataset:')
                st.write(f'<span style="background-color: #f63366; color: white; padding: 2px 6px; border-radius: 4px;">{target_column}</span>', unsafe_allow_html=True)

                st.markdown("""---""")    

                st.header("🧾 Datatypes of each column")
                data_types = df.dtypes.apply(lambda x: x.name)  
                st.write(data_types)

                st.markdown("""---""")

                st.header("❓ Missing Values Count")
                missing_values = df.isnull().sum()
                st.write(missing_values)

                st.markdown("""---""")

                st.header("🔄 Count of Unique Values")
                unique_values = df.nunique()
                st.write(unique_values)

                st.markdown("""---""")

                st.header("📈 Column Classification")
                numeric_cols = df.select_dtypes(include=['number']).columns
                categorical_cols = df.select_dtypes(include=['object']).columns
                st.write("#### Numeric Columns🔢")
                for col in numeric_cols:
                    st.write(f'<span style="background-color: #f63366; color: white; padding: 2px 6px; border-radius: 4px;">{col}</span>', unsafe_allow_html=True)
                st.write("")
                st.write("#### Categorical Columns🗂️")
                for col in categorical_cols:
                    st.write(f'<span style="background-color: #f63366; color: white; padding: 2px 6px; border-radius: 4px;">{col}</span>', unsafe_allow_html=True)
                st.write("")
                st.write("#### Target Column🎯")
                st.write(f'<span style="background-color: #f63366; color: white; padding: 2px 6px; border-radius: 4px;">{target_column}</span>', unsafe_allow_html=True)
