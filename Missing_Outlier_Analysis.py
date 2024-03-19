import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial import distance
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore
from sklearn.cluster import KMeans
from scipy.spatial import distance
from sklearn.covariance import LedoitWolf
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pandas.plotting import parallel_coordinates
import pingouin as pg
import missingno as msno
#from pingouin import mcar, mar
import missingno as msno
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.datasets import make_classification
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image 

def run_missing_outlier_analysis():

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

    info = Image.open("Images/miss.png")
    '''st.set_page_config(
            page_title="Missing & Outlier Analysis",
            page_icon=info,
            #layout="wide",
        )'''


    st.markdown(hide_menu, unsafe_allow_html=True)


    def count_missing_values(data):
        st.write("#### 🤷‍♂️ Count Missing Values")
        st.info("Counts the missing values in each column of the dataset.")
        
        missing_values = data.isnull().sum()
        
        st.write("Missing Value Count for Each Column:")
        st.write(missing_values)

    def percentage_missing_values(data):
        st.write("#### 🤷‍♂️ Percentage of Missing Values")
        st.info("Calculates the percentage of missing values in each column of the dataset.")
        missing_percentage = (data.isnull().sum() / len(data)) * 100
        
        st.write("Percentage of Missing Values for Each Column:")
        st.write(missing_percentage)

    def missing_data_heatmap(data):
        st.write("#### 🤷‍♂️ Missing Data Heatmap")
        st.info("Generates a heatmap to visualize the missing data in the dataset.")
        plt.figure(figsize=(10, 6))
        sns.heatmap(data.isnull(), cmap='viridis', cbar=False, yticklabels=False)
        st.pyplot(plt)

    def detect_outliers_z_score(data):
        st.write("#### 👽 Outlier Detection using Z-Score")
        st.info("Detects outliers in a numeric column using the Z-Score method.")
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()

        if len(numeric_cols) >= 1:
            selected_col = st.selectbox("Select a Numeric Column for Outlier Detection:", numeric_cols, key="outlier_col")

            if selected_col:
                st.write("**Selected Column:**  ", selected_col)

                column_data = data[selected_col]

                z_scores = np.abs(stats.zscore(column_data))

                threshold = st.slider("Select Z-Score Threshold:", min_value=1, max_value=5, value=3, key="z_score_threshold")

                outliers = np.where(z_scores > threshold)

                outlier_values = column_data.iloc[outliers[0]]

                st.write("**Threshold for Z-Score**  ", threshold)
                st.write("**Number of Outliers Detected**  ", len(outlier_values))

                if len(outlier_values) > 0:
                    st.write("**Outlier Values**")
                    st.write(outlier_values)
                else:
                    st.success("No outliers detected.")
            else:
                st.warning("⚠️ Please select a numeric column for outlier detection.")
        else:
            st.warning("⚠️ There are no numeric columns in the dataset for outlier detection.")

    def detect_outliers_percentile(data):
        st.write("#### 👽 Outlier Detection using Percentile")
        st.info("Detects outliers in a numeric column using the Percentile (IQR) method.")
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()

        if len(numeric_cols) >= 1:
            selected_col = st.selectbox("Select a Numeric Column for Outlier Detection:", numeric_cols, key="percentile_col")

            if selected_col:
                st.write("**Selected Column:**  ", selected_col)

                column_data = data[selected_col]

                Q1 = np.percentile(column_data, 25)
                Q3 = np.percentile(column_data, 75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = column_data[(column_data < lower_bound) | (column_data > upper_bound)]

                st.write("**IQR (Interquartile Range)**  ", IQR)
                st.write("**Lower Bound for Outliers**  ", lower_bound)
                st.write("**Upper Bound for Outliers**  ", upper_bound)
                st.write("**Number of Outliers Detected**  ", len(outliers))

                if len(outliers) > 0:
                    st.write("**Outlier Values**")
                    st.write(outliers)
                else:
                    st.success("No outliers detected.")
            else:
                st.warning("⚠️ Please select a numeric column for outlier detection.")
        else:
            st.warning("⚠️ There are no numeric columns in the dataset for outlier detection.")

    def detect_outliers_euclidean_distance(data):
        st.write("#### 👽 Outlier Detection using Euclidean Distance")
        st.info("Detects outliers using the Euclidean Distance method on selected numeric columns.")
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()

        if len(numeric_cols) >= 2:
            selected_cols = st.multiselect("Select Numeric Columns for Distance-Based Outlier Detection:", numeric_cols, key="distance_cols")

            if len(selected_cols) >= 2:
                st.write("**Selected Columns:**  ", selected_cols)

                column_data = data[selected_cols]

                euclidean_distances = distance.cdist(column_data, column_data, 'euclidean')
                
                threshold = st.slider("Select Euclidean Distance Threshold:", min_value=0, max_value=10, value=2, key="euclidean_threshold")

                outliers = np.argwhere(euclidean_distances > threshold)

                st.write("**Threshold for Euclidean Distance**  ", threshold)
                st.write("**Number of Outliers Detected**  ", len(outliers))

                if len(outliers) > 0:
                    st.write("**Outlier Indices (Row, Column)**")
                    st.write(outliers)
                else:
                    st.success("No outliers detected.")
            else:
                st.warning("⚠️ Please select at least two numeric columns for distance-based outlier detection.")
        else:
            st.warning("⚠️ There are not enough numeric columns (at least two) in the dataset for distance-based outlier detection.")

    def detect_outliers_scatter_plot(data):
        st.write("#### 👽 Outlier Detection using Scatter Plot")
        st.info("Detects outliers using a scatter plot and distance threshold on selected numeric columns.")
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()

        if len(numeric_cols) >= 2:
            selected_cols = st.multiselect("Select Two Numeric Columns for Scatter Plot:", numeric_cols, key="scatter_cols")

            if len(selected_cols) == 2:
                st.write("**Selected Columns:**  ", selected_cols)

                column_data = data[selected_cols]

                x_col, y_col = selected_cols

                plt.scatter(column_data[x_col], column_data[y_col], c='blue', alpha=0.7)
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.title("Scatter Plot")

                st.pyplot(plt)
                
                threshold = st.slider("Select Distance Threshold:", min_value=0, max_value=10, value=2, key="scatter_threshold")

                mean_x = column_data[x_col].mean()
                mean_y = column_data[y_col].mean()
                distances = ((column_data[x_col] - mean_x) ** 2 + (column_data[y_col] - mean_y) ** 2) ** 0.5

                outliers = column_data[distances > threshold]

                st.write("**Threshold for Distance**  ", threshold)
                st.write("**Number of Outliers Detected**  ", len(outliers))

                if len(outliers) > 0:
                    st.write("**Outlier Values**")
                    st.write(outliers)
                else:
                    st.success("No outliers detected.")
            else:
                st.warning("⚠️ Please select exactly two numeric columns for scatter plot-based outlier detection.")
        else:
            st.warning("⚠️ There are not enough numeric columns (at least two) in the dataset for scatter plot-based outlier detection.")

    def perform_basic_analysis(data):
        st.header("Basic Visual Analysis 🔍")

        count_missing_values(data)

        st.markdown("""---""")

        percentage_missing_values(data)

        st.markdown("""---""")

        missing_data_heatmap(data)

        st.markdown("""---""")

        detect_outliers_z_score(data)

        st.markdown("""---""")

        detect_outliers_percentile(data)

        st.markdown("""---""")

        detect_outliers_euclidean_distance(data)

        st.markdown("""---""")

        detect_outliers_scatter_plot(data)

    def detect_outliers_dbscan(data):
        st.write("#### 👽 Outlier Detection using DBSCAN")
        st.info("Detects outliers using the DBSCAN algorithm on selected numeric columns.")
        
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()

        if len(numeric_cols) >= 2:
            selected_cols = st.multiselect("Select Numeric Columns for DBSCAN Outlier Detection:", numeric_cols, key="dbscan_cols")

            if len(selected_cols) >= 2:
                st.write("**Selected Columns:**  ", selected_cols)

                column_data = data[selected_cols]

                eps = st.slider("Select DBSCAN Epsilon:", min_value=0.1, max_value=10.0, value=1.0, key="dbscan_eps")
                min_samples = st.slider("Select Minimum Samples:", min_value=1, max_value=10, value=5, key="dbscan_min_samples")

                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                clusters = dbscan.fit_predict(column_data)

                st.write("**Number of Outliers Detected**  ", len(clusters[clusters == -1]))

                if len(selected_cols) == 2:
                    plt.scatter(column_data[selected_cols[0]], column_data[selected_cols[1]], c=clusters, cmap='viridis')
                    plt.xlabel(selected_cols[0])
                    plt.ylabel(selected_cols[1])
                    plt.title("DBSCAN Clustering")
                    st.pyplot(plt)
                else:
                    st.warning("⚠️ To visualize clusters, select exactly two numeric columns.")
            else:
                st.warning("⚠️ Please select at least two numeric columns for DBSCAN outlier detection.")
        else:
            st.warning("⚠️ There are not enough numeric columns (at least two) in the dataset for DBSCAN outlier detection.")

    def detect_outliers_lof(data):
        st.write("#### 👽 Outlier Detection using LOF (Local Outlier Factor)")
        st.info("Detects outliers using the Local Outlier Factor (LOF) algorithm on selected numeric columns.")
        
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()

        if len(numeric_cols) >= 2:
            selected_cols = st.multiselect("Select Numeric Columns for LOF Outlier Detection:", numeric_cols, key="lof_cols")

            if len(selected_cols) >= 2:
                st.write("**Selected Columns:**  ", selected_cols)

                column_data = data[selected_cols]

                n_neighbors = st.slider("Select Number of Neighbors:", min_value=1, max_value=20, value=5, key="lof_neighbors")
                contamination = st.slider("Select Contamination (Percentage of Outliers):", min_value=0.0, max_value=0.5, value=0.05, key="lof_contamination")

                lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
                outliers = lof.fit_predict(column_data)

                st.write("**Number of Outliers Detected**  ", len(outliers[outliers == -1]))
            else:
                st.warning("⚠️ Please select at least two numeric columns for LOF outlier detection.")
        else:
            st.warning("⚠️ There are not enough numeric columns (at least two) in the dataset for LOF outlier detection.")

    def detect_outliers_isolation_forest(data):
        st.write("#### 👽 Outlier Detection using Isolation Forest")
        st.info("Detects outliers using the Isolation Forest algorithm on selected numeric columns.")
        
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()

        if len(numeric_cols) >= 2:
            selected_cols = st.multiselect("Select Numeric Columns for Isolation Forest Outlier Detection:", numeric_cols, key="isolation_forest_cols")

            if len(selected_cols) >= 2:
                st.write("**Selected Columns:**  ", selected_cols)

                column_data = data[selected_cols]

                contamination = st.slider("Select Contamination (Percentage of Outliers):", min_value=0.0, max_value=0.5, value=0.05, key="isolation_forest_contamination")

                isolation_forest = IsolationForest(contamination=contamination)
                outliers = isolation_forest.fit_predict(column_data)

                st.write("**Number of Outliers Detected**  ", len(outliers[outliers == -1]))
            else:
                st.warning("⚠️ Please select at least two numeric columns for Isolation Forest outlier detection.")
        else:
            st.warning("⚠️ There are not enough numeric columns (at least two) in the dataset for Isolation Forest outlier detection.")

    def detect_outliers_grubbs_test(data):
        st.write("#### 👽 Outlier Detection using Grubbs' Test")
        st.info("Detects outliers using Grubbs' Test on a selected numeric column.")
        
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()

        if len(numeric_cols) >= 1:
            selected_col = st.selectbox("Select a Numeric Column for Grubbs' Test Outlier Detection:", numeric_cols, key="grubbs_test_col")

            if selected_col:
                st.write("**Selected Column:**  ", selected_col)

                column_data = data[selected_col]

                z_scores = zscore(column_data)
                outliers = (np.abs(z_scores) > 3)

                st.write("**Number of Outliers Detected**  ", len(column_data[outliers]))
            else:
                st.warning("⚠️ Please select a numeric column for Grubbs' Test outlier detection.")
        else:
            st.warning("⚠️ There are no numeric columns in the dataset for Grubbs' Test outlier detection.")

    def detect_outliers_kmeans(data):
        st.write("#### 👽 Outlier Detection using K-Means Clustering")
        st.info("Detects outliers using K-Means Clustering on selected numeric columns.")
        
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()

        if len(numeric_cols) >= 2:
            selected_cols = st.multiselect("Select Numeric Columns for K-Means Outlier Detection:", numeric_cols, key="kmeans_cols")

            if len(selected_cols) >= 2:
                st.write("**Selected Columns:**  ", selected_cols)
                
                if data[selected_cols].isnull().any().any():
                    st.warning("⚠️ The selected columns contain missing values (NaN). Please clean your data before calculating the correlation.")
                    return
                
                numeric_data = data.select_dtypes(include=[np.number])
                if np.isinf(numeric_data).any().any():
                    st.warning("⚠️ The selected columns contain infinite values (infs). Please clean your data before feature selection.")
                    return

                column_data = data[selected_cols]

                n_clusters = st.slider("Select Number of Clusters (K):", min_value=2, max_value=10, value=3, key="kmeans_clusters")

                kmeans = KMeans(n_clusters=n_clusters)
                kmeans.fit(column_data)

                distances = kmeans.transform(column_data)
                min_distances = distances.min(axis=1)

                threshold = st.slider("Select Distance Threshold:", min_value=0.1, max_value=10.0, value=2.0, key="kmeans_threshold")

                outliers = min_distances > threshold

                st.write("**Number of Outliers Detected**  ", len(column_data[outliers]))
            else:
                st.warning("⚠️ Please select at least two numeric columns for K-Means outlier detection.")
        else:
            st.warning("⚠️ There are not enough numeric columns (at least two) in the dataset for K-Means outlier detection.")

    def mean_imputation_and_visualization(data):
        st.write("#### 🤷‍♂️ Mean Imputation")
        st.info("Performs mean imputation on a selected numeric column with missing values and visualizes the data before and after imputation using Seaborn.")
        
        numeric_columns_with_null = [col for col in data.select_dtypes(include=['number']).columns if data[col].isnull().any()]

        if numeric_columns_with_null:
            column_name = st.selectbox("Select a Numeric Column with Missing Values for Mean Imputation and Visualization:", numeric_columns_with_null)

            if column_name in data.columns:
                if pd.api.types.is_numeric_dtype(data[column_name]):
                    st.write("**Selected Column:**  ", column_name)

                    col1, col2 = st.columns(2)  
                    with col1:
                        plt.figure(figsize=(8, 4))
                        sns.histplot(data[column_name].dropna(), kde=True, color='blue')
                        plt.title(f"Before Imputation: Distribution of {column_name}")
                        plt.xlabel(column_name)
                        plt.ylabel("Frequency")
                        st.pyplot(plt)

                    imputer = SimpleImputer(strategy='mean')
                    data[column_name] = imputer.fit_transform(data[[column_name]])

                    with col2:
                        plt.figure(figsize=(8, 4))
                        sns.histplot(data[column_name], kde=True, color='green')
                        plt.title(f"After Mean Imputation: Distribution of {column_name}")
                        plt.xlabel(column_name)
                        plt.ylabel("Frequency")
                        st.pyplot(plt)

                    st.success("Mean imputation completed successfully.")
                else:
                    st.warning(f"⚠️ The selected column '{column_name}' is not numeric.")
            else:
                st.warning(f"⚠️ The selected column '{column_name}' does not exist in the dataset.")
        else:
            st.warning("⚠️ There are no numeric columns with missing values in the dataset.")

    def median_imputation(data):
        st.write("#### 🤷‍♂️ Median Imputation")
        st.info("Performs median imputation on selected numeric columns with missing values.")
        
        numeric_cols_with_null = [col for col in data.select_dtypes(include='number').columns if data[col].isnull().any()]

        if numeric_cols_with_null:
            column_name = st.selectbox("Select a Numeric Column for Median Imputation:", numeric_cols_with_null)

            if column_name in data.columns:
                if pd.api.types.is_numeric_dtype(data[column_name]):
                    st.write("**Selected Column:**  ", column_name)

                    col1, col2 = st.columns(2)  
                    with col1:
                        plt.figure(figsize=(8, 4))
                        sns.histplot(data[column_name].dropna(), kde=True, color='blue')
                        plt.title(f"Before Imputation: Distribution of {column_name}")
                        plt.xlabel(column_name)
                        plt.ylabel("Frequency")
                        st.pyplot(plt)

                    imputer = SimpleImputer(strategy='median')
                    data[column_name] = imputer.fit_transform(data[[column_name]])

                    with col2:
                        plt.figure(figsize=(8, 4))
                        sns.histplot(data[column_name], kde=True, color='green')
                        plt.title(f"After Median Imputation: Distribution of {column_name}")
                        plt.xlabel(column_name)
                        plt.ylabel("Frequency")
                        st.pyplot(plt)

                    st.success("Median imputation completed successfully.")
                else:
                    st.warning(f"⚠️ The selected column '{column_name}' is not numeric.")
            else:
                st.warning(f"⚠️ The selected column '{column_name}' does not exist in the dataset.")
        else:
            st.warning("⚠️ There are no numeric columns with missing values in the dataset.")

    def mode_imputation(data):
        st.write("#### 🤷‍♂️ Mode Imputation")
        st.info("Performs mode imputation on selected categorical columns with missing values.")
        
        categorical_cols_with_null = [col for col in data.select_dtypes(include=['object','bool']).columns if data[col].isnull().any()]

        if categorical_cols_with_null:
            column_name = st.selectbox("Select a Categorical Column for Mode Imputation:", categorical_cols_with_null)

            if column_name in data.columns:
                if pd.api.types.is_categorical_dtype(data[column_name]):
                    st.write("**Selected Column:**  ", column_name)

                    col1, col2 = st.columns(2)  
                    with col1:
                        plt.figure(figsize=(8, 4))
                        sns.countplot(data[column_name].dropna(), color='blue')
                        plt.title(f"Before Imputation: Count of {column_name}")
                        plt.xlabel(column_name)
                        plt.ylabel("Count")
                        st.pyplot(plt)

                    imputer = SimpleImputer(strategy='most_frequent')
                    data[column_name] = imputer.fit_transform(data[[column_name]])

                    with col2:
                        plt.figure(figsize=(8, 4))
                        sns.countplot(data[column_name], color='green')
                        plt.title(f"After Mode Imputation: Count of {column_name}")
                        plt.xlabel(column_name)
                        plt.ylabel("Count")
                        st.pyplot(plt)

                    st.success("Mode imputation completed successfully.")
                else:
                    st.warning(f"⚠️ The selected column '{column_name}' is not categorical.")
            else:
                st.warning(f"⚠️ The selected column '{column_name}' does not exist in the dataset.")
        else:
            st.warning("⚠️ There are no categorical columns with missing values in the dataset.")

    def little_mcar_test(data):
        st.write("#### 🤷‍♂️ Little's MCAR Test")
        st.info("Performs Little's Missing Completely at Random (MCAR) Test on selected columns with missing values.")

        numeric_cols_with_null = [col for col in data.select_dtypes(include=['number']).columns if data[col].isnull().any()]

        if numeric_cols_with_null:
            column_name = st.selectbox("Select a Numeric Column for Little's MCAR Test:", numeric_cols_with_null)

            if column_name in data.columns:
                if pd.api.types.is_numeric_dtype(data[column_name]):
                    st.write("**Selected Column:**  ", column_name)

                    test_data = data[[column_name]].copy()

                    imputer = IterativeImputer(max_iter=10, random_state=0)
                    test_data[column_name] = imputer.fit_transform(test_data[[column_name]])

                    observed_null = data[column_name].isnull().sum()
                    expected_null = test_data[column_name].isnull().sum()
                    test_statistic = (observed_null - expected_null) ** 2 / expected_null

                    from scipy.stats import chi2
                    p_value = 1 - chi2.cdf(test_statistic, df=1)

                    st.write("**Chi-Squared Test Statistic:**  ", test_statistic)
                    st.write("**p-value:**  ", p_value)

                    if p_value < 0.05:
                        st.warning("⚠️ The p-value is less than 0.05, indicating that the column may not be MCAR.")
                    else:
                        st.success("✅ The p-value is greater than or equal to 0.05, suggesting that the column may be MCAR.")
                else:
                    st.warning(f"⚠️ The selected column '{column_name}' is not numeric.")
            else:
                st.warning(f"⚠️ The selected column '{column_name}' does not exist in the dataset.")
        else:
            st.warning("⚠️ There are no numeric columns with missing values in the dataset.")


    def perform_intermediate_analysis(data):
        st.header("Intermediate Visual Analysis 📈")

        mean_imputation_and_visualization(data)

        st.markdown("""---""")

        median_imputation(data)

        st.markdown("""---""")

        mode_imputation(data)

        st.markdown("""---""")

        little_mcar_test(data)

        st.markdown("""---""")

        detect_outliers_dbscan(data) 

        st.markdown("""---""")

        detect_outliers_lof(data) 

        st.markdown("""---""")

        detect_outliers_isolation_forest(data) 

        st.markdown("""---""")

        detect_outliers_grubbs_test(data) 
        
        st.markdown("""---""")

        detect_outliers_kmeans(data) 

    def perform_pca(data):
        st.write("#### 🤷‍♂️ Principal Component Analysis (PCA)")
        st.info("Performs PCA on numeric variables to visualize data structure.")

        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()

        if len(numeric_cols) >= 2:
            selected_cols = st.multiselect("Select Numeric Columns for PCA:", numeric_cols, key="pca_cols")

            if len(selected_cols) >= 2:
                st.write("**Selected Columns:**  ", selected_cols)

                X = data[selected_cols]

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                pca = PCA(n_components=2)
                principal_components = pca.fit_transform(X_scaled)

                pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

                plt.figure(figsize=(8, 6))
                plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.5)
                plt.title("PCA")
                plt.xlabel("Principal Component 1 (PC1)")
                plt.ylabel("Principal Component 2 (PC2)")
                st.pyplot(plt)

            else:
                st.warning("⚠️ Please select at least two numeric columns for PCA.")
        else:
            st.warning("⚠️ There are not enough numeric columns (at least two) in the dataset for PCA.")


    def perform_tsne(data):
        st.write("#### 🤷‍♂️ t-Distributed Stochastic Neighbor Embedding (t-SNE)")
        st.info("Performs t-SNE on numeric and categorical variables to visualize high-dimensional data.")

        all_cols = data.columns.tolist()
        selected_cols = st.multiselect("Select Columns for t-SNE (Numeric and Categorical):", all_cols, key="tsne_cols")

        if selected_cols:
            st.write("**Selected Columns:**  ", selected_cols)

            X = data[selected_cols]

            if len(X.columns) >= 2:
                tsne = TSNE(n_components=2, random_state=0)
                tsne_result = tsne.fit_transform(X)

                tsne_df = pd.DataFrame(data=tsne_result, columns=['Dimension 1', 'Dimension 2'])

                plt.figure(figsize=(8, 6))
                plt.scatter(tsne_df['Dimension 1'], tsne_df['Dimension 2'], alpha=0.5)
                plt.title("t-SNE")
                plt.xlabel("Dimension 1")
                plt.ylabel("Dimension 2")
                st.pyplot(plt)

            else:
                st.warning("⚠️ Please select at least two columns for t-SNE.")
        else:
            st.warning("⚠️ Please select columns for t-SNE.")

    def plot_advanced_heatmap(data):
        st.write("#### 🤷‍♂️ Advanced Heatmaps")
        st.info("Plots advanced heatmaps for visualizing complex missing data patterns.")

        plt.figure(figsize=(10, 6))
        msno.matrix(data)
        plt.title("Advanced Heatmap")
        st.pyplot(plt)

    def propensity_score_matching(data):
        st.write("#### 🤷‍♂️ Propensity Score Matching")
        st.info("Performs propensity score matching on both numeric and categorical variables when dealing with observational data to reduce bias in treatment effect estimation.")
        
        st.write("Performing propensity score matching in process: ")

        X, y = make_classification(n_samples=1000, n_features=10, random_state=0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(X_train)

        treated_indices = np.where(y_train == 1)[0]
        control_indices = np.where(y_train == 0)[0]

        matched_indices = []

        for treated_index in treated_indices:
            matched_index = knn.kneighbors([X_train[treated_index]], return_distance=False)[0][0]
            matched_indices.append(matched_index)

        matched_X = X_train[matched_indices]
        matched_y = y_train[matched_indices]

        model = RandomForestClassifier(random_state=0)
        model.fit(matched_X, matched_y)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        st.success("Propensity score matching completed.")
        st.write("**Accuracy after matching:**  ", accuracy)


    def detect_outliers_mahalanobis(data):
        st.write("#### 👽 Outlier Detection using Mahalanobis Distance")
        st.info("Detects outliers using Mahalanobis Distance on selected numeric columns.")
        
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()

        if len(numeric_cols) >= 2:
            selected_cols = st.multiselect("Select Numeric Columns for Mahalanobis Distance Outlier Detection:", numeric_cols, key="mahalanobis_cols")

            if len(selected_cols) >= 2:
                st.write("**Selected Columns:**  ", selected_cols)

                if data[selected_cols].isnull().any().any():
                    st.warning("⚠️ The selected columns contain missing values (NaN). Please clean your data before outlier detection.")
                    return

                numeric_data = data.select_dtypes(include=[np.number])
                if np.isinf(numeric_data).any().any():
                    st.warning("⚠️ The selected columns contain infinite values (infs). Please clean your data before feature selection.")
                    return

                column_data = data[selected_cols]

                covariance = LedoitWolf().fit(column_data).covariance_
                mahalanobis_distances = distance.cdist(column_data, [column_data.mean().values], 'mahalanobis', VI=covariance)

                threshold = st.slider("Select Mahalanobis Distance Threshold:", min_value=0, max_value=10, value=2, key="mahalanobis_threshold")

                outliers = mahalanobis_distances > threshold

                st.write("**Threshold for Mahalanobis Distance**  ", threshold)
                st.write("**Number of Outliers Detected**  ", len(outliers[outliers]))
            else:
                st.warning("⚠️ Please select at least two numeric columns for Mahalanobis Distance outlier detection.")
        else:
            st.warning("⚠️ There are not enough numeric columns (at least two) in the dataset for Mahalanobis Distance outlier detection.")

    def detect_outliers_cosine_similarity(data):
        st.write("#### 👽 Outlier Detection using Cosine Similarity")
        st.info("Detects outliers using Cosine Similarity on selected numeric columns.")
        
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()

        if len(numeric_cols) >= 2:
            selected_cols = st.multiselect("Select Numeric Columns for Cosine Similarity Outlier Detection:", numeric_cols, key="cosine_similarity_cols")

            if len(selected_cols) >= 2:
                st.write("**Selected Columns:**  ", selected_cols)

                if data[selected_cols].isnull().any().any():
                    st.warning("⚠️ The selected columns contain missing values (NaN). Please clean your data before outlier detection.")
                    return

                numeric_data = data.select_dtypes(include=[np.number])
                if np.isinf(numeric_data).any().any():
                    st.warning("⚠️ The selected columns contain infinite values (infs). Please clean your data before feature selection.")
                    return

                column_data = data[selected_cols]

                cosine_similarities = cosine_similarity(column_data)

                threshold = st.slider("Select Cosine Similarity Threshold:", min_value=0.1, max_value=1.0, value=0.9, key="cosine_similarity_threshold")

                outliers = cosine_similarities < threshold

                st.write("**Threshold for Cosine Similarity**  ", threshold)
                st.write("**Number of Outliers Detected**  ", len(outliers[outliers]))
            else:
                st.warning("⚠️ Please select at least two numeric columns for Cosine Similarity outlier detection.")
        else:
            st.warning("⚠️ There are not enough numeric columns (at least two) in the dataset for Cosine Similarity outlier detection.")

    def detect_outliers_oneclass_svm(data):
        st.write("#### 👽 Outlier Detection using One-Class SVM")
        st.info("Detects outliers using One-Class SVM on selected numeric columns.")
        
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()

        if len(numeric_cols) >= 2:
            selected_cols = st.multiselect("Select Numeric Columns for One-Class SVM Outlier Detection:", numeric_cols, key="oneclass_svm_cols")

            if len(selected_cols) >= 2:
                st.write("**Selected Columns:**  ", selected_cols)

                if data[selected_cols].isnull().any().any():
                    st.warning("⚠️ The selected columns contain missing values (NaN). Please clean your data before outlier detection.")
                    return

                numeric_data = data.select_dtypes(include=[np.number])
                if np.isinf(numeric_data).any().any():
                    st.warning("⚠️ The selected columns contain infinite values (infs). Please clean your data before feature selection.")
                    return

                column_data = data[selected_cols]

                nu = st.slider("Select Nu Parameter (Anomaly Proportion):", min_value=0.01, max_value=0.5, value=0.1, key="oneclass_svm_nu")

                one_class_svm = OneClassSVM(nu=nu)
                one_class_svm.fit(column_data)

                outliers = one_class_svm.predict(column_data) == -1

                st.write("**Anomaly Proportion (Nu)**  ", nu)
                st.write("**Number of Outliers Detected**  ", len(outliers[outliers]))
            else:
                st.warning("⚠️ Please select at least two numeric columns for One-Class SVM outlier detection.")
        else:
            st.warning("⚠️ There are not enough numeric columns (at least two) in the dataset for One-Class SVM outlier detection.")

    def detect_outliers_autoencoder(data):
        st.write("#### 👽 Outlier Detection using Autoencoders")
        st.info("Detects outliers using Autoencoders on selected numeric columns.")
        
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()

        if len(numeric_cols) >= 2:
            selected_cols = st.multiselect("Select Numeric Columns for Autoencoder Outlier Detection:", numeric_cols, key="autoencoder_cols")

            if len(selected_cols) >= 2:
                st.write("**Selected Columns:**  ", selected_cols)

                if data[selected_cols].isnull().any().any():
                    st.warning("⚠️ The selected columns contain missing values (NaN). Please clean your data before outlier detection.")
                    return

                numeric_data = data.select_dtypes(include=[np.number])
                if np.isinf(numeric_data).any().any():
                    st.warning("⚠️ The selected columns contain infinite values (infs). Please clean your data before feature selection.")
                    return

                column_data = data[selected_cols]

                scaler = StandardScaler()
                column_data_standardized = scaler.fit_transform(column_data)

                input_dim = len(selected_cols)
                encoding_dim = 5

                input_layer = keras.Input(shape=(input_dim,))
                encoder_layer = layers.Dense(encoding_dim, activation='relu')(input_layer)
                decoder_layer = layers.Dense(input_dim, activation='sigmoid')(encoder_layer)

                autoencoder = keras.Model(inputs=input_layer, outputs=decoder_layer)

                autoencoder.compile(optimizer='adam', loss='mean_squared_error')
                autoencoder.fit(column_data_standardized, column_data_standardized, epochs=100, batch_size=32, shuffle=True, validation_split=0.2)

                reconstructions = autoencoder.predict(column_data_standardized)
                mse = np.mean(np.square(column_data_standardized - reconstructions), axis=1)

                threshold = st.slider("Select Reconstruction Error Threshold:", min_value=0.1, max_value=10.0, value=2.0, key="autoencoder_threshold")

                outliers = mse > threshold

                st.write("**Threshold for Reconstruction Error**  ", threshold)
                st.write("**Number of Outliers Detected**  ", len(outliers[outliers]))
            else:
                st.warning("⚠️ Please select at least two numeric columns for Autoencoder outlier detection.")
        else:
            st.warning("⚠️ There are not enough numeric columns (at least two) in the dataset for Autoencoder outlier detection.")

    def perform_advanced_analysis(data):
        st.header("Advanced Visual Analysis 🚀")

        perform_pca(data)

        st.markdown("""---""")

        perform_tsne(data)

        st.markdown("""---""")

        plot_advanced_heatmap(data)

        st.markdown("""---""")

        propensity_score_matching(data)

        st.markdown("""---""")

        detect_outliers_mahalanobis(data)

        st.markdown("""---""")

        detect_outliers_cosine_similarity(data)

        st.markdown("""---""")

        detect_outliers_oneclass_svm(data)

        st.markdown("""---""")

        detect_outliers_autoencoder(data)
    

    def main():
        st.markdown("<h1 style='text-align: center;'>❓ Missing & Outlier Value Analysis on Dataset</h1>", unsafe_allow_html=True)

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
                    with st.expander("🔍 Basic Analysis"):
                        perform_basic_analysis(data)
                    
                    with st.expander("📈 Intermediate Analysis"):
                        perform_intermediate_analysis(data)
                    
                    with st.expander("🚀 Advanced Analysis"):
                        perform_advanced_analysis(data)


    if __name__ == "__main__":
        main()