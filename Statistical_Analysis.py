import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, poisson
from scipy.stats import kruskal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cosine
import warnings
from PIL import Image 

def run_statistical_analysis():

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

    info = Image.open("Images/stats.png")
    '''st.set_page_config(
            page_title="Statistical Analysis",
            page_icon=info,
            #layout="wide",
        )'''

    st.markdown(hide_menu, unsafe_allow_html=True)


    def detect_outliers_with_zscore(data):
        st.write("#### 🌌 Outlier Detection using Z-Score")
        
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) >= 1:
            selected_col = st.selectbox("Select a Numeric Column for Outlier Detection:", numeric_cols, key="outlier_col")
            
            if selected_col:
                st.write("**Selected Column:**  ",selected_col)
                
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

    def calculate_and_plot_distributions(data):
        st.write("#### 🎲 Probability Distributions")
        
        st.info("Probability distributions are models that describe the likelihood of different outcomes.")
        
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        selected_col = st.selectbox("Select a Numeric Column from the Dataset:", numeric_cols, key="selected_col")
        
        if selected_col:
            st.write(f"**Selected Column:** {selected_col}")
            
            mean_normal = np.mean(data[selected_col])
            std_dev_normal = np.std(data[selected_col], ddof=1)  
            
            st.write("**Mean (μ)**  ", mean_normal)
            st.write("**Standard Deviation (σ)**  ", std_dev_normal)
            
            x_normal = np.linspace(mean_normal - 3*std_dev_normal, mean_normal + 3*std_dev_normal, 1000)
            y_normal = norm.pdf(x_normal, mean_normal, std_dev_normal)
            
            plt.figure(figsize=(8, 5))
            plt.plot(x_normal, y_normal, color='blue')
            plt.xlabel("X")
            plt.ylabel("Probability Density")
            plt.title("Normal Distribution")
            plt.grid(True)
            
            st.pyplot(plt)
            
            lam_poisson = np.mean(data[selected_col])
            
            st.write("**Mean (λ) for Poisson Distribution**  ", lam_poisson)
            
            x_poisson = np.arange(0, int(round(lam_poisson)) + 1)
            y_poisson = poisson.pmf(x_poisson, lam_poisson)
            
            plt.figure(figsize=(8, 5))
            plt.bar(x_poisson, y_poisson, color='green', alpha=0.7)
            plt.xlabel("X")
            plt.ylabel("Probability Mass Function")
            plt.title("Poisson Distribution")
            plt.grid(True)
            
            st.pyplot(plt)
            
        else:
            st.warning("⚠️ Please select a numeric column from the dataset.")

    def perform_basic_analysis(data):
        st.header("Basic Analysis 🔍")

        st.write("#### 🗒️ Descriptive Analysis")
        st.write("")
        numeric_cols = data.select_dtypes(include=['number'])

        if not numeric_cols.empty:
            st.write("###### 📚 Mean ", numeric_cols.mean())
            st.markdown("""---""")
            st.write("###### 📚 Median ", numeric_cols.median())
            st.markdown("""---""")
            st.write("###### 📚 Mode ", numeric_cols.mode().iloc[0])  
            st.markdown("""---""")
            st.write("###### 📚 Range ", numeric_cols.max() - numeric_cols.min())
            st.markdown("""---""")
            st.write("###### 📚 Standard Deviation ", numeric_cols.std())
        else:
            st.warning("⚠️  No numeric columns found in the dataset.")
        
        st.markdown("""---""")

        st.write("#### 📡 Frequency Distribution")
        col = st.selectbox("Select a column for frequency distribution:", data.columns)
        freq_table = data[col].value_counts().reset_index()
        freq_table.columns = [col, "Frequency"]
        st.write(freq_table)

        st.markdown("""---""")

        detect_outliers_with_zscore(data)

        st.markdown("""---""")

        calculate_and_plot_distributions(data)


    def perform_chi_square_test(data):    
        st.write("Select two categorical columns for the test:")
        
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        
        if len(categorical_cols) >= 2:
            col1, col2 = st.columns(2)
            selected_col1 = col1.selectbox("Column 1:", categorical_cols, key="selectbox_col1")
            selected_col2 = col2.selectbox("Column 2:", categorical_cols, key="selectbox_col2")
            
            contingency_table = pd.crosstab(data[selected_col1], data[selected_col2])
            
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            
            st.write("**Chi-Square Statistic**  ", chi2)
            st.write("**P-Value**  ", p_value)
            st.write("**Degrees of Freedom**  ", dof)
            
            if p_value < 0.05:
                st.success("Conclusion: There is a significant association between the selected columns.")
            else:
                st.success("Conclusion: There is no significant association between the selected columns.")
        else:
            st.warning("⚠️ There are not enough categorical columns (at least two) to perform a Chi-Square Test.")

    def perform_ttest(data):
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()

        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            selected_col1 = col1.selectbox("Column 1:", numeric_cols)
            selected_col2 = col2.selectbox("Column 2:", numeric_cols)

            t_statistic, p_value = stats.ttest_ind(data[selected_col1], data[selected_col2])

            st.write("**T-Statistic**  ", t_statistic)
            st.write("**P-Value**  ", p_value)
        else:
            st.warning("⚠️ There are not enough numeric columns (at least two) to perform a t-test.")

    def perform_anova(data):
        st.write("Select a column for one-way ANOVA:")

        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()

        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            selected_col = col1.selectbox("Column:", numeric_cols, key="selectbox_col3")
            group_col = col2.selectbox("Grouping Column:", data.columns, key="selectbox_col4")

            groups = [data[data[group_col] == group][selected_col] for group in data[group_col].unique()]
            f_statistic, p_value = stats.f_oneway(*groups)

            st.write("**F-Statistic**  ", f_statistic)
            st.write("**P-Value**  ", p_value)
        else:
            st.warning("⚠️ There are not enough numeric columns (at least two) to perform ANOVA.")

    def perform_wilcoxon_rank_sum(data):
        st.write("Select two columns for Wilcoxon rank-sum test (Mann-Whitney U test):")

        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()

        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            selected_col1 = col1.selectbox("Column 1:", numeric_cols, key="selectbox_col5")
            selected_col2 = col2.selectbox("Column 2:", numeric_cols, key="selectbox_col6")

            u_statistic, p_value = stats.mannwhitneyu(data[selected_col1], data[selected_col2], alternative='two-sided')

            st.write("**U-Statistic**  ", u_statistic)
            st.write("**P-Value**  ", p_value)
        else:
            st.warning("⚠️ There are not enough numeric columns (at least two) to perform Wilcoxon rank-sum test.")

    def perform_logistic_regression(data):
        st.write("##### 📈 Logistic Regression Analysis")
        st.write("Select a binary target variable and one or more independent numeric/categorical columns:")
        
        suitable_cols = data.select_dtypes(include=['number']).columns.tolist()
        
        if len(suitable_cols) >= 2:
            col1, col2 = st.columns(2)
            
            target_col = col1.selectbox("Target Variable:", suitable_cols, key="logistic_target")
            independent_cols = col2.multiselect("Independent Variables:", suitable_cols, key="logistic_ind")
            
            if independent_cols:
                X = data[independent_cols]
                y = data[target_col]
                
                if len(y.unique()) != 2:
                    st.warning("⚠️ Please select a binary target variable (0 or 1).")
                    return
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                model = LogisticRegression()
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                confusion = confusion_matrix(y_test, y_pred)
                #classification_rep = classification_report(y_test, y_pred)
                
                st.write("**Accuracy**  ", accuracy)
                st.write("**Confusion Matrix**  ")
                st.write(confusion)
                
            else:
                st.warning("⚠️ Please select at least one independent variable.")
        
        else:
            st.warning("⚠️ There are not enough suitable columns (at least two) to perform logistic regression.")

    def perform_linear_regression(data):
        st.write("##### 📈 Linear Regression Analysis")
        st.write("Select a dependent and one or more independent numeric columns:")
        
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            
            dependent_col = col1.selectbox("Dependent Variable:", numeric_cols, key="linear_dep")
            independent_cols = col2.multiselect("Independent Variables:", numeric_cols, key="linear_ind")
            
            if independent_cols:
                X = data[independent_cols]
                y = data[dependent_col]
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                model = LinearRegression()
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                
                mse = mean_squared_error(y_test, y_pred)
                r_squared = r2_score(y_test, y_pred)
                
                st.write("**Mean Squared Error**  ", mse)
                st.write("**R-squared**  ", r_squared)
                
            else:
                st.warning("⚠️ Please select at least one independent variable.")
        
        else:
            st.warning("⚠️ There are not enough numeric columns (at least two) to perform linear regression.")

    def perform_discriminant_analysis(data):
        st.write("#### 🚫 Linear Discriminant Analysis (LDA)")
        st.write("Select a binary target variable and one or more independent numeric/categorical columns:")
        
        suitable_cols = data.select_dtypes(include=['number']).columns.tolist()
        
        if len(suitable_cols) >= 2:
            col1, col2 = st.columns(2)
            
            target_col = col1.selectbox("Target Variable:", suitable_cols, key="lda_target")
            independent_cols = col2.multiselect("Independent Variables:", suitable_cols, key="lda_ind")
            
            if independent_cols:
                X = data[independent_cols]
                y = data[target_col]
                
                if len(y.unique()) != 2:
                    st.warning("⚠️ Please select a binary target variable (0 or 1).")
                    return
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                lda = LinearDiscriminantAnalysis()
                lda.fit(X_train, y_train)
                
                y_pred = lda.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                confusion = confusion_matrix(y_test, y_pred)
                #classification_rep = classification_report(y_test, y_pred)
                
                st.write("**Accuracy**  ", accuracy)
                st.write("**Confusion Matrix**  ")
                st.write(confusion)

                
            else:
                st.warning("⚠️ Please select at least one independent variable.")
        
        else:
            st.warning("⚠️ There are not enough suitable columns (at least two) to perform Linear Discriminant Analysis (LDA).")

    def perform_factor_analysis(data):
        st.write("#### 🧩 Factor Analysis")
        st.write("Select numeric columns for factor analysis:")
        
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) >= 2:
            selected_cols = st.multiselect("Select Columns for Factor Analysis:", numeric_cols, key="factor_cols")
            
            if selected_cols:
                X = data[selected_cols]
                
                fa = FactorAnalysis(n_components=len(selected_cols))
                X_fa = fa.fit_transform(X)
                
                st.write("**Factor Loadings** ")
                st.write(pd.DataFrame(fa.components_, columns=selected_cols, index=[f"Factor {i+1}" for i in range(len(selected_cols))]))
                
            else:
                st.warning("⚠️ Please select at least two numeric columns for factor analysis.")
        
        else:
            st.warning("⚠️ There are not enough numeric columns (at least two) to perform Factor Analysis.")

    def perform_intermediate_analysis(data):
        st.header("Intermediate Analysis 📈")
        
        st.write("#### 🤔 Hypothesis Testing")
        st.write('')
        st.write('##### ✏️ T-Test')
        perform_ttest(data)
        st.write('')
        st.write('')
        st.write('##### ➡️ One-Way ANOVA')
        perform_anova(data)
        st.write('')
        st.write('')
        st.write("##### 🕊️ Chi-Square Test of Independence")
        perform_chi_square_test(data)
        st.write('')
        st.write('')
        st.write('##### 🔝 Wilcoxon rank-sum test (Mann-Whitney U test)')
        perform_wilcoxon_rank_sum(data)

        st.markdown("""---""")

        perform_linear_regression(data)
        st.write('')
        st.write('')
        perform_logistic_regression(data)

        st.markdown("""---""")

        perform_discriminant_analysis(data)

        st.markdown("""---""")

        perform_factor_analysis(data)

    def perform_pca_analysis(data):
        st.write("#### 💡 Principal Component Analysis (PCA)")
        st.write("Select numeric columns for PCA:")
        
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) >= 2:
            selected_cols = st.multiselect("Select Columns for PCA:", numeric_cols, key="pca_cols")
            
            if selected_cols:
                X = data[selected_cols]
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                pca = PCA()
                X_pca = pca.fit_transform(X_scaled)
                
                explained_variance_ratio = pca.explained_variance_ratio_
                
                st.write("**Principal Components**")
                st.write(pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(len(selected_cols))]))
                
                st.write("**Explained Variance Ratio**")
                st.write(explained_variance_ratio)
                
            else:
                st.warning("⚠️ Please select at least two numeric columns for PCA.")
        
        else:
            st.warning("⚠️ There are not enough numeric columns (at least two) to perform PCA.")

    def perform_kruskal_wallis_test(data):
        st.write("#### 🧪 Kruskal-Wallis Test")
        st.write("Select numeric columns for the Kruskal-Wallis Test:")
        
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) >= 2:
            selected_cols = st.multiselect("Select Columns for Kruskal-Wallis Test:", numeric_cols, key="kruskal_cols")
            
            if len(selected_cols) >= 2:
                st.write(f"**Selected Columns:** {', '.join(selected_cols)}")
                
                groups = [data[col] for col in selected_cols]
                
                try:
                    stat, p_value = kruskal(*groups)
                    
                    st.write("**Kruskal-Wallis Test Statistic**  ", stat)
                    st.write("**p-value**  " ,p_value)
                    
                    if p_value < 0.05:
                        st.success("The p-value is less than 0.05, indicating statistically significant differences between groups.")
                    else:
                        st.success("The p-value is greater than or equal to 0.05, indicating no statistically significant differences between groups.")
                    
                except Exception as e:
                    st.warning(f"⚠️ An error occurred while performing the Kruskal-Wallis Test: {str(e)}")
            else:
                st.warning("⚠️ Please select at least two numeric columns for the Kruskal-Wallis Test.")
        
        else:
            st.warning("⚠️ There are not enough numeric columns (at least two) to perform the Kruskal-Wallis Test.")

    def calculate_confidence_interval(data, confidence_level=0.95):
        st.write("#### 💪 Confidence Intervals")
        st.write("Confidence Level  ", confidence_level)
        
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) >= 1:
            selected_col = st.selectbox("Select a Numeric Column:", numeric_cols, key="confidence_col")
            
            if selected_col:
                st.write(f"**Selected Column:** {selected_col}")
                
                sample_data = data[selected_col]
                
                try:
                    mean = np.mean(sample_data)
                    std = np.std(sample_data, ddof=1)
                    n = len(sample_data)
                    margin_of_error = stats.t.ppf((1 + confidence_level) / 2, n - 1) * (std / np.sqrt(n))
                    lower_bound = mean - margin_of_error
                    upper_bound = mean + margin_of_error
                    
                    st.write("**Sample Mean**  ", mean)
                    st.write("**Margin of Error**  ", margin_of_error)
                    st.write("**Confidence Interval**  ", [lower_bound, upper_bound])
                    
                except Exception as e:
                    st.warning(f"⚠️ An error occurred while calculating the confidence interval: {str(e)}")
            else:
                st.warning("⚠️ Please select a numeric column for calculating the confidence interval.")
        
        else:
            st.warning("⚠️ There are no numeric columns in the dataset for calculating the confidence interval.")

    def random_sampling_and_clt(data):
        st.write("#### 🎰 Random Sampling and Central Limit Theorem")
        
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) >= 1:
            selected_col = st.selectbox("Select a Numeric Column for Sampling:", numeric_cols, key="sampling_col")
            
            if selected_col:
                st.write(f"**Selected Column:** {selected_col}")
                
                population_data = data[selected_col]
                
                sample_size = st.number_input("Enter Sample Size:", min_value=1, max_value=len(population_data), value=30, key="sample_size")
                num_samples = st.number_input("Enter Number of Samples:", min_value=1, max_value=100, value=30, key="num_samples")
                
                sample_means = []
                
                
                for _ in range(int(num_samples)):
                    sample = np.random.choice(population_data, size=sample_size, replace=False)
                    
                    sample_mean = np.mean(sample)
                    sample_means.append(sample_mean)
                    
                plt.figure(figsize=(8, 5))
                sns.histplot(sample_means, bins=20, color='skyblue', kde=True)
                plt.xlabel("Sample Means")
                plt.ylabel("Frequency")
                plt.title("Sampling Distribution of Sample Means")
                plt.grid(True)
                
                st.pyplot(plt)
                
                st.info("As per the Central Limit Theorem, the sampling distribution of sample means approximates a normal distribution.")
                    
            else:
                st.warning("⚠️ Please select a numeric column for random sampling.")
        
        else:
            st.warning("⚠️ There are no numeric columns in the dataset for random sampling.")

    def perform_advanced_analysis(data):
        st.header("Advanced Analysis 🚀")
        
        perform_pca_analysis(data)

        st.markdown("""---""")

        perform_kruskal_wallis_test(data)

        st.markdown("""---""")

        calculate_confidence_interval(data, confidence_level=0.95)

        st.markdown("""---""")

        random_sampling_and_clt(data)

    def main():
        st.title("📉 Statistical Analysis on Dataset")
        
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
