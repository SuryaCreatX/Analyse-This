import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image 

def run_descriptive_analysis():

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

    info = Image.open("Images/descriptive.png")
    '''st.set_page_config(
            page_title="Descriptive Analysis",
            page_icon=info,
            #layout="wide",
        )'''


    st.markdown(hide_menu, unsafe_allow_html=True)

def run_descriptive_analysis():

    def basic_analysis(data):
        basic_expander = st.expander("🔍 Basic Analysis")
        with basic_expander:
            st.header("Basic Analysis 🔍")
            
            st.write("#### 🔢 Count")
            st.write(data.shape[0])
            
            st.markdown("""---""")

            st.write("#### 🔢 Mean", data.mean())

            st.markdown("""---""")

            st.write("#### 🔢 Standard Deviation", data.std())

            st.markdown("""---""")

            st.write("#### 🔢 Minimum", data.min())

            st.markdown("""---""")

            st.write("#### 🔢 25th Percentile (Q1)", data.quantile(0.25))

            st.markdown("""---""")

            st.write("#### 🔢 50th Percentile (Median)", data.median())

            st.markdown("""---""")

            st.write("#### 🔢 75th Percentile (Q3)", data.quantile(0.75))

            st.markdown("""---""")

            st.write("#### 🔢 Maximum", data.max())

    def kde_estimation(data):
        selected_variable = st.selectbox("Select a Variable", data.columns)
        
        fig, ax = plt.subplots()
        
        sns.kdeplot(data[selected_variable], shade=True, ax=ax)
        ax.set_xlabel(selected_variable)
        ax.set_ylabel("Density")
        
        st.pyplot(fig)

    def multivariate_analysis(data):
        st.write("Select variables for multivariate analysis:")
        selected_variables = st.multiselect("Select Variables", data.columns)

        sns.pairplot(data[selected_variables], diag_kind='kde')
        st.pyplot()

    def intermediate_analysis(data):
        intermediate_expander = st.expander("📈 Intermediate Analysis")
        with intermediate_expander:
            st.header("Intermediate Analysis 📈")
            
            st.write("#### 🧮 Range", data.max() - data.min())

            st.markdown("""---""")

            st.write("#### 🧮 Variance", data.var())

            st.markdown("""---""")

            st.write("#### 🧮 Coefficient of Variation (CV)", data.std() / data.mean())

            st.markdown("""---""")

            st.write("#### 🧮 Skewness", stats.skew(data))

            st.markdown("""---""")

            st.write("#### 🧮 Kurtosis", stats.kurtosis(data))

            st.markdown("""---""")

            st.write("#### 🧮 Interquartile Range (IQR)", data.quantile(0.75) - data.quantile(0.25))

            st.markdown("""---""")

            st.write("#### 🧮 Median Absolute Deviation (MAD)", data.mad())

            st.markdown("""---""")

            st.write("#### 🧮 Median Absolute Deviation from the Median (MAD-Median)", stats.median_absolute_deviation(data))

            st.markdown("""---""")

            st.write("#### 🧮 Z-Score", np.abs(stats.zscore(data)))

            st.markdown("""---""")

            st.write("#### 🧮 Kernerl Density Analyis (KDE)")
            kde_estimation(data)

            st.markdown("""---""")

    def regression_analysis(data):
        st.write("Select independent variable column:")
        independent_variable = st.selectbox("Independent Variable", data.columns)
        dependent_variable = st.selectbox("Dependent Variable", data.columns)

        X = data[[independent_variable]]
        y = data[dependent_variable]

        model = LinearRegression()
        model.fit(X, y)

        st.write("Regression Coefficients:")
        st.write("**Intercept** ")
        st.write(model.intercept_)
        st.write("**Slope** ")
        st.write(model.coef_[0])

    def harmonic_mean(data):
        if (data <= 0).any().any():
            st.warning("⚠️ Harmonic mean is not defined for data with negative or zero values.")
        else:
            harmonic_mean_value = stats.hmean(data)
            st.write("Harmonic Mean:", harmonic_mean_value)

    def cluster_analysis(data):
        st.write("Select the number of clusters:")
        num_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=3)

        st.write("Select features for clustering:")
        default_features = [data.columns[0]]  
        feature_columns = st.multiselect("Select Features", data.columns, default=default_features)

        X = data[feature_columns]

        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        data['Cluster'] = kmeans.fit_predict(X)

        st.write("**Cluster Centers**")
        st.write(kmeans.cluster_centers_)

        st.write("**Clustered Data**")
        st.write(data)

    def advanced_analysis(data):
        advanced_expander = st.expander("🚀 Advanced Analysis")
        with advanced_expander:
            st.header("Advanced Analysis 🚀")
            
            st.write("#### 🌟 Coefficient of Quartile Deviation (CQD)", (data.quantile(0.75) - data.quantile(0.25)) / (data.quantile(0.75) + data.quantile(0.25)))

            st.markdown("""---""")

            st.write("#### 🌟 Range Interquartile Ratio (RIR)", (data.max() - data.min()) / (data.quantile(0.75) - data.quantile(0.25)))

            st.markdown("""---""")

            st.write("#### 🌟 Relative Range", (data.max() - data.min()) / (data.max() + data.min()))
            
            st.markdown("""---""")

            st.write("#### 🌟 Variability Ratio (VR)", data.std() / data.mean())

            st.markdown("""---""")

            st.write("#### 🌟 Geometric Mean", stats.gmean(data))

            st.markdown("""---""")

            st.write("#### 🌟 Harmonic Mean")
            harmonic_mean(data)

            st.markdown("""---""")

            st.write("#### 🌟 Regression Analysis")
            regression_analysis(data.select_dtypes(include=[np.number]))

            st.markdown("""---""")

            st.write("#### 🌟 Cluster Analysis")
            cluster_analysis(data)

    def main():
        st.title("📊 Descriptive Analysis on Dataset")

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
                basic_analysis(data.select_dtypes(include=[np.number]))
                intermediate_analysis(data.select_dtypes(include=[np.number]))
                advanced_analysis(data.select_dtypes(include=[np.number]))

    if __name__ == "__main__":
        main()
