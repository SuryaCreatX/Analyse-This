import streamlit as st
from PIL import Image 
from Info_Analysis import run_info_analysis
from Descriptive_Analysis import run_descriptive_analysis
from Statistical_Analysis import run_statistical_analysis
from Visualization_Analysis import run_visualization_analysis
from Correlation_Analysis import run_correlation_analysis
from Missing_Outlier_Analysis import run_missing_outlier_analysis
from Feature_Engineering_Analysis import run_feature_engineering_analysis
from Feature_Selection_Analysis import run_feature_selection_analysis


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
</style>
"""
@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img 

info = Image.open("Images/model.png")
st.set_page_config(
        page_title="Analyse This!",
        page_icon=info,
        #layout="wide",
    )

st.markdown(hide_menu, unsafe_allow_html=True)

page_options = {
    "🏠 Home": "Home",
    "📋Info Analysis": "Info Analysis",  
    "📊 Descriptive Analysis": "Descriptive Analysis",
    "📈 Statistical Analysis": "Statistical Analysis",
    "📉 Visualization Analysis": "Visualization Analysis",
    "🔄 Correlation Analysis": "Correlation Analysis",
    "🔍 Missing & Outlier Analysis": "Missing & Outlier Analysis",
    "🛠️ Feature Engineering Analysis": "Feature Engineering Analysis",
    "✂️ Feature Selection Analysis": "Feature Selection Analysis"
}

st.sidebar.title("🚀 Analysis Explorer")
st.sidebar.markdown("---")

selected_page = st.sidebar.radio("Select an Analysis", list(page_options.keys()))

if selected_page == "🏠 Home":
    st.title("🚀 Analyse This!")
elif selected_page == "📋 Info Analysis":
    run_info_analysis()  
elif selected_page == "📊 Descriptive Analysis":
    run_descriptive_analysis()  
elif selected_page == "📈 Statistical Analysis":
    run_statistical_analysis()  
elif selected_page == "📉 Visualization Analysis":
    run_visualization_analysis()  
elif selected_page == "🔄 Correlation Analysis":
    run_correlation_analysis()  
elif selected_page == "🔍 Missing & Outlier Analysis":
    run_missing_outlier_analysis()  
elif selected_page == "🛠️ Feature Engineering Analysis":
    run_feature_engineering_analysis()  
elif selected_page == "✂️ Feature Selection Analysis":
    run_feature_selection_analysis()  

