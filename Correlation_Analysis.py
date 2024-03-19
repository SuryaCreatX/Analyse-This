import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from scipy.signal import correlate
from scipy.stats import chi2_contingency
import networkx as nx
from scipy.stats import (
    pearsonr,
    spearmanr,
    pointbiserialr,
    chi2_contingency,
    f_oneway,
)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import scipy.stats as stats
from PIL import Image 

def run_correlation_analysis():

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

    info = Image.open("Images/corr.png")
    '''st.set_page_config(
            page_title="Correlation Analysis",
            page_icon=info,
            #layout="wide",
        )'''

    st.markdown(hide_menu, unsafe_allow_html=True)

    def calculate_pearson_correlation(data):
        st.write("Calculate Pearson Correlation Coefficient")
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) < 2:
            st.warning("⚠️ There are not enough numeric columns (at least two) for correlation calculation.")
            return

        col1, col2 = st.columns(2)
        selected_col1 = col1.selectbox("Select the first numeric column:", numeric_cols, key = 'p_c1')
        selected_col2 = col2.selectbox("Select the second numeric column:", numeric_cols, key = 'p_c2')

        if data[selected_col1].isnull().any() or data[selected_col2].isnull().any():
            st.warning("⚠️ The selected columns contain missing values (NaN). Please clean your data before calculating the correlation.")
            return
        
        if np.isinf(data[selected_col1]).any() or np.isinf(data[selected_col2]).any():
            st.warning("⚠️ The selected columns contain infinite values (infs). Please clean your data before calculating the correlation.")
            return

        pearson_corr = np.corrcoef(data[selected_col1], data[selected_col2])[0, 1]

        st.write("**Pearson Correlation Coefficient**  ", pearson_corr)

    def calculate_spearman_rank_correlation(data):
        st.write("Calculate Spearman Rank Correlation:")

        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) < 2:
            st.warning("⚠️ There are not enough numeric columns (at least two) for correlation calculation.")
            return

        col1, col2 = st.columns(2)
        selected_col1 = col1.selectbox("Select the first numeric column:", numeric_cols, key = 'p_c3')
        selected_col2 = col2.selectbox("Select the second numeric column:", numeric_cols, key = 'p_c4')

        if data[selected_col1].isnull().any() or data[selected_col2].isnull().any():
            st.warning("⚠️ The selected columns contain missing values (NaN). Please clean your data before calculating the correlation.")
            return
        
        if np.isinf(data[selected_col1]).any() or np.isinf(data[selected_col2]).any():
            st.warning("⚠️ The selected columns contain infinite values (infs). Please clean your data before calculating the correlation.")
            return

        spearman_corr, _ = spearmanr(data[selected_col1], data[selected_col2])

        st.write("**Spearman Rank Correlation Coefficient**  ", spearman_corr)

    def calculate_kendall_tau_rank_correlation(data):
        st.write("Calculate Kendall Tau Rank Correlation:")

        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) < 2:
            st.warning("⚠️ There are not enough numeric columns (at least two) for correlation calculation.")
            return

        col1, col2 = st.columns(2)
        selected_col1 = col1.selectbox("Select the first numeric column:", numeric_cols, key = 'p_c5')
        selected_col2 = col2.selectbox("Select the second numeric column:", numeric_cols, key = 'p_c6')

        if data[selected_col1].isnull().any() or data[selected_col2].isnull().any():
            st.warning("⚠️ The selected columns contain missing values (NaN). Please clean your data before calculating the correlation.")
            return
        
        if np.isinf(data[selected_col1]).any() or np.isinf(data[selected_col2]).any():
            st.warning("⚠️ The selected columns contain infinite values (infs). Please clean your data before calculating the correlation.")
            return

        kendall_tau_corr, _ = kendalltau(data[selected_col1], data[selected_col2])

        st.write("**Kendall Tau Rank Correlation Coefficient**  ", kendall_tau_corr)

    def scatter_plot(data):
        st.info("Select two numeric columns to create a scatter plot:")

        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()

        if len(numeric_cols) < 2:
            st.warning("⚠️ There are not enough numeric columns (at least two) to create a scatter plot.")
            return

        col1, col2 = st.columns(2)
        x_col = col1.selectbox("X-axis (Independent Variable):", numeric_cols, key = 'p_c7')
        y_col = col2.selectbox("Y-axis (Dependent Variable):", numeric_cols, key = 'p_c8')

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=x_col, y=y_col, data=data)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"Scatter Plot between {x_col} and {y_col}")
        st.pyplot(plt)

    def heatmap(data):
        st.info("Visualize correlations between numeric columns using a heatmap:")

        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()

        if len(numeric_cols) < 2:
            st.warning("⚠️ There are not enough numeric columns (at least two) to create a heatmap.")
            return

        corr_matrix = data[numeric_cols].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Heatmap")
        st.pyplot(plt)

    def perform_basic_analysis(data):
        st.header("Basic Analysis 🔍")

        st.write('#### 💯 Pearson Correlation  Anaysis')
        calculate_pearson_correlation(data)

        st.markdown("""---""")

        st.write('#### 💯 Spearman Rank Correlation Anaysis')
        calculate_spearman_rank_correlation(data)
        
        st.markdown("""---""")

        st.write('#### 💯 Kendall Tau Rank Anaysis')
        calculate_kendall_tau_rank_correlation(data)

        st.markdown("""---""")

        st.write('#### 📊 Scatter Plot Analysis')
        scatter_plot(data)

        st.markdown("""---""")

        st.write('#### 📊 Heatmap Analysis')
        heatmap(data)


    def partial_correlation(data):
        st.write("Calculate Partial Correlation:")
        st.write("Select three numeric columns to calculate the partial correlation:")

        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()

        if len(numeric_cols) < 3:
            st.warning("⚠️ There are not enough numeric columns (at least three) to calculate partial correlation.")
            return

        col1, col2, col3 = st.columns(3)
        var1 = col1.selectbox("Variable 1:", numeric_cols, key = 'col1')
        var2 = col2.selectbox("Variable 2:", numeric_cols, key = 'col12')
        control_var = col3.selectbox("Control Variable:", numeric_cols, key = 'col3')

        if var1 == var2 or var1 == control_var or var2 == control_var:
            st.warning("⚠️ Selected variables must be independent. Please choose different variables.")
            return

        partial_corr = pg.partial_corr(data, x=var1, y=var2, covar=control_var, method="pearson")['r']

        st.write("**Partial Correlation Coefficient**  ", partial_corr)

    def point_biserial_correlation(data):
        st.write("Calculate Point-Biserial Correlation:")
        st.write("Select a binary (dichotomous) column and a continuous column:")

        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

        if len(numeric_cols) == 0 or len(categorical_cols) == 0:
            st.warning("⚠️ There are not enough numeric or categorical columns for point-biserial correlation.")
            return

        col1, col2 = st.columns(2)
        binary_var = col1.selectbox("Binary (Dichotomous) Variable:", categorical_cols, key='p_c11')
        continuous_var = col2.selectbox("Continuous Variable:", numeric_cols, key='p_c12')

        data[binary_var] = data[binary_var].map({'M': 0, 'F': 1})

        if np.any(np.isnan(data[binary_var])) or np.any(np.isnan(data[continuous_var])) or \
        np.any(np.isinf(data[binary_var])) or np.any(np.isinf(data[continuous_var])):
            st.warning("⚠️ The selected columns contain NaNs or infs. Please clean your data before calculating the correlation.")
            return

        group1 = data[data[binary_var] == data[binary_var].unique()[0]][continuous_var]
        group2 = data[data[binary_var] == data[binary_var].unique()[1]][continuous_var]

        point_biserial_corr, _ = stats.pointbiserialr(data[binary_var], data[continuous_var])

        st.write("**Point-Biserial Correlation Coefficient:**", point_biserial_corr)


    def cramers_v(data):
        st.write("Calculate Cramer's V:")
        st.write("Select two categorical columns to calculate Cramer's V:")

        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

        if len(categorical_cols) < 2:
            st.warning("⚠️ There are not enough categorical columns (at least two) to calculate Cramer's V.")
            return

        col1, col2 = st.columns(2)
        cat_var1 = col1.selectbox("Categorical Variable 1:", categorical_cols, key = 'p_c13')
        cat_var2 = col2.selectbox("Categorical Variable 2:", categorical_cols, key = 'p_c14')

        confusion_matrix = pd.crosstab(data[cat_var1], data[cat_var2])
        chi2, _, _, _ = stats.chi2_contingency(confusion_matrix)

        n = confusion_matrix.sum().sum()
        min_dim = min(confusion_matrix.shape) - 1

        cramers_v = np.sqrt(chi2 / (n * min_dim))

        st.write("**Cramer's V**  ", cramers_v)


    def perform_intermediate_analysis(data):
        st.header("Intermediate Analysis 📈")

        st.write('#### 💯 Partial Correlation  Anaysis')
        partial_correlation(data)

        st.markdown("""---""")

        st.write('#### 💯 Point-Biserial Correlation Anaysis')
        point_biserial_correlation(data)

        st.markdown("""---""")

        st.write("#### 💯 Cramer's V Anaysis")
        cramers_v(data)    

    def canonical_correlation_analysis(data):
        st.write("Perform Canonical Correlation Analysis (CCA):")
        st.write("Select two sets of numeric variables to assess associations between them")

        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()

        if len(numeric_cols) < 4:
            st.warning("⚠️ There are not enough numeric columns (at least four) to perform CCA.")
            return

        col1, col2 = st.columns(2)
        set1_vars = col1.multiselect("Select variables for Set 1:", numeric_cols, [], key = 'b1')
        set2_vars = col2.multiselect("Select variables for Set 2:", numeric_cols, [], key = 'b2')

        if not set1_vars or not set2_vars:
            st.warning("⚠️ Please select variables for both sets.")
            return

        cca = pd.concat([data[set1_vars], data[set2_vars]], axis=1)
        cca_corr = cca.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(cca_corr, annot=True, cmap="coolwarm", linewidths=0.5)
        plt.title("Canonical Correlation Analysis (CCA) Heatmap")
        st.pyplot(plt)

    def distance_correlation(data):
        st.write("Calculate Distance Correlation:")
        st.write("Select two sets of numeric variables to measure dependence between them")

        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()

        if len(numeric_cols) < 2:
            st.warning("⚠️ There are not enough numeric columns (at least two) to calculate distance correlation.")
            return

        col1, col2 = st.columns(2)
        set1_vars = col1.multiselect("Select variables for Set 1:", numeric_cols, [])
        set2_vars = col2.multiselect("Select variables for Set 2:", numeric_cols, [])

        if not set1_vars or not set2_vars:
            st.warning("⚠️ Please select variables for both sets.")
            return

        set1_data = data[set1_vars]
        set2_data = data[set2_vars]

        def dist_corr(X, Y):
            A = pdist(X)
            B = pdist(Y)

            if not np.allclose(A, A.T) or not np.allclose(B, B.T):
                st.warning("⚠️ The distance matrices 'X' are not symmetric. Please check your data.")
                return None

            return squareform(np.outer(A, B)) / np.sqrt(np.outer(A, A) * np.outer(B, B))

        distance_corr = dist_corr(set1_data, set2_data)

        if distance_corr is not None:
            distance_corr = distance_corr.mean()
            st.write("**Distance Correlation**  ", distance_corr)

    def time_series_cross_correlation(data):
        st.write("Calculate Time Series Cross-Correlation:")
        st.write("Select two time series variables for cross-correlation analysis")

        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()

        if len(numeric_cols) < 2:
            st.warning("⚠️ There are not enough numeric columns (at least two) to calculate time series cross-correlation.")
            return

        col1, col2 = st.columns(2)
        time_series1 = col1.selectbox("Time Series Variable 1:", numeric_cols, key = 'a7')
        time_series2 = col2.selectbox("Time Series Variable 2:", numeric_cols, key = 'a8')

        if time_series1 == time_series2:
            st.warning("⚠️ Please select two different time series variables.")
            return

        max_lag = min(len(data[time_series1]), len(data[time_series2])) - 1
        lags = list(range(-max_lag, max_lag + 1))

        cross_corr_values = [np.correlate(data[time_series1], data[time_series2], mode='valid') for lag in lags]

        max_corr_index = np.argmax(cross_corr_values)
        max_corr_lag = lags[max_corr_index]
        max_corr_value = cross_corr_values[max_corr_index]

        st.write(f"**Maximum Cross-Correlation (Lag {max_corr_lag}):**")
        st.write(max_corr_value)

    def network_analysis(data):
        st.write("**Perform Network Analysis**")
        st.info("Represent variables as nodes and correlations as edges in a network")

        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()

        if len(numeric_cols) < 2:
            st.warning("⚠️ There are not enough numeric columns (at least two) to perform network analysis.")
            return

        corr_matrix = data[numeric_cols].corr()

        G = nx.Graph()

        for i, var1 in enumerate(numeric_cols):
            for j, var2 in enumerate(numeric_cols):
                if i < j:
                    correlation = corr_matrix.loc[var1, var2]
                    G.add_edge(var1, var2, weight=correlation)

        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G)
        labels = {node: node for node in G.nodes()}
        nx.draw(G, pos, with_labels=True, labels=labels, font_size=10, node_size=1000, node_color='skyblue', font_color='black')
        st.pyplot(plt)


    def perform_advanced_analysis(data):
        st.header("Advanced Analysis 🚀")

        st.write('#### 💯 Canonical Correlation Anaysis')
        canonical_correlation_analysis(data)

        st.markdown("""---""")

        st.write('#### 💯 Distance Correlation Anaysis')
        distance_correlation(data)

        st.markdown("""---""")

        st.write('#### 💯 Time Series Cross Correlation Anaysis')
        time_series_cross_correlation(data)

        st.markdown("""---""")

        st.write('#### 📊 Network Correlation Anaysis')
        network_analysis(data)



    def correlation_analysis(data, target_column):
        st.info("Calculate and visualize correlations with the target column.")

        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        predictor_cols = [col for col in numeric_cols if col != target_column]

        if len(numeric_cols) == 0:
            st.warning("⚠️ There are no numeric columns to perform correlation analysis.")
            return

        if len(predictor_cols) == 0:
            st.warning("⚠️ There are no predictor columns for correlation analysis.")
            return

        predictor_col = st.selectbox("Select Predictor Column:", predictor_cols)
        correlation_method = st.selectbox("Select Correlation Method:", ['Pearson', 'Spearman', 'Point-Biserial', 'Cramer\'s V', 'ANOVA', 'Partial Correlation'], key='s1')
        visualization_type = st.selectbox("Select Visualization Type:", ['Scatter Plot', 'Box Plot', 'Histogram', 'Heatmap'], key='ss1')

        st.subheader(f"🤖 {correlation_method} Correlation - {target_column} vs {predictor_col}")

        if correlation_method == 'Pearson':
            if target_column in numeric_cols:
                corr_coefficient, _ = pearsonr(data[target_column], data[predictor_col])
            else:
                st.warning("⚠️ The target column must be numeric for Pearson correlation.")
                return
        elif correlation_method == 'Spearman':
            if target_column in numeric_cols:
                corr_coefficient, _ = spearmanr(data[target_column], data[predictor_col])
            else:
                st.warning("⚠️ The target column must be numeric for Spearman correlation.")
                return
        elif correlation_method == 'Point-Biserial':
            if target_column in numeric_cols:

                if data[target_column].isnull().any() or data[predictor_col].isnull().any():
                    st.warning("⚠️ The selected columns contain missing values (NaN). Please clean your data before calculating the correlation.")
                    return
        
                if np.isinf(data[target_column]).any() or np.isinf(data[predictor_col]).any():
                    st.warning("⚠️ The selected columns contain infinite values (infs). Please clean your data before calculating the correlation.")
                    return
                
                corr_coefficient, _ = pointbiserialr(data[target_column], data[predictor_col])
            else:
                st.warning("⚠️ The target column must be numeric for Point-Biserial correlation.")
                return
        elif correlation_method == "Cramer's V":
            if target_column in categorical_cols and predictor_col in categorical_cols:
                confusion_matrix = pd.crosstab(data[target_column], data[predictor_col])
                corr_coefficient = cramers_v(confusion_matrix)
            else:
                st.warning("⚠️ Both target and predictor columns must be categorical for Cramer's V.")
                return
        elif correlation_method == 'ANOVA':
            if target_column in numeric_cols and predictor_col in categorical_cols:
                groups = [data[data[predictor_col] == group][target_column] for group in data[predictor_col].unique()]
                _, p_value = f_oneway(*groups)
                st.write(f"**ANOVA P-Value:** {p_value:.4f}")
            else:
                st.warning("⚠️ To use ANOVA, the target column must be numeric and the predictor column must be categorical.")
                return
        elif correlation_method == 'Partial Correlation':
            covariate_cols = [col for col in numeric_cols if col != target_column and col != predictor_col]
            if len(covariate_cols) == 0:
                st.warning("⚠️ There are no covariate columns for partial correlation.")
                return
            partial_corr_coeff, _ = pg.partial_corr(data[[target_column, predictor_col] + covariate_cols], target_column, predictor_col, method='pearson')
            st.write(f"**Partial Correlation Coefficient:** {partial_corr_coeff:.4f}")
        

        if visualization_type == 'Scatter Plot':
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=predictor_col, y=target_column, data=data)
            plt.title(f"{correlation_method} Correlation Scatter Plot")
            st.pyplot(plt)
        elif visualization_type == 'Box Plot':
            plt.figure(figsize=(8, 6))
            sns.boxplot(x=predictor_col, y=target_column, data=data)
            plt.title(f"{correlation_method} Correlation Box Plot")
            st.pyplot(plt)
        elif visualization_type == 'Histogram':
            plt.figure(figsize=(8, 6))
            sns.histplot(data, x=predictor_col, hue=target_column, kde=True)
            plt.title(f"{correlation_method} Correlation Histogram")
            st.pyplot(plt)
        elif visualization_type == 'Heatmap' and correlation_method == "Cramer's V":
            plt.figure(figsize=(8, 6))
            sns.heatmap(confusion_matrix, annot=True, cmap="YlGnBu")
            plt.title("Cramer's V Heatmap")
            st.pyplot(plt)
        else:
            st.warning('⚠️ Plot is not valid. Choose a different correlation or plotting technique.')


    def perform_correlation_target (data):
        st.header("Correlation v/s Target 🎯")  

        target_column = st.selectbox('Select the target column:', data.columns, key = 'ttarget')

        if target_column not in data.columns:
            st.warning(f"⚠️ The selected target column '{target_column}' does not exist in the DataFrame.")
        else:
            st.write('#### 💻 Correlation Matrix') 
            if st.checkbox('Show Correlation Matrix'):
                correlation_matrix = data.corr()
                st.write(correlation_matrix[target_column])

        st.markdown("""---""")

        st.write('#### 🔄 Multi Correlation Analysis')
        correlation_analysis(data, target_column)

        
    def main():
        st.title("🔗 Correlation Analysis on Dataset")

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

                with st.expander("🎯 Correlation v/s Target"):
                    perform_correlation_target (data)

    if __name__ == "__main__":
        main()
