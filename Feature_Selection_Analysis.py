import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import VarianceThreshold
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE, SelectFromModel, SelectPercentile, RFECV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from boruta import BorutaPy
from geneticalgorithm import geneticalgorithm as ga
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from PIL import Image 

def run_feature_selection_analysis():

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

    info = Image.open("Images/select.png")
    '''st.set_page_config(
            page_title="Best Feature Analysis",
            page_icon=info,
            #layout="wide",
        )'''

    st.markdown(hide_menu, unsafe_allow_html=True)


    def correlation_analysis(data):
        st.write("#### 🔗 Correlation Analysis")
        st.info("Identify and visualize correlations between numeric features.")

        if data.select_dtypes(include=['number']).shape[1] >= 2:
            corr_matrix = data.select_dtypes(include=['number']).corr()
            st.write("###### 📈 Correlation Matrix")
            st.write(corr_matrix)

            st.write("###### 📊 Correlation Heatmap")
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
            plt.title("Heatmap for Feature Selction")
            st.pyplot(plt)
        else:
            st.warning("⚠️ There are not enough numeric columns (at least two) in the dataset for correlation analysis.")

    def univariate_feature_selection(data):
        st.write("#### 🎯 Univariate Feature Selection")
        st.info("Assess the relevance of individual features using univariate feature selection.")

        if data.isnull().any().any():
            st.warning("⚠️ The selected columns contain missing values (NaN). Please clean your data before feature selection.")
            return

        numeric_data = data.select_dtypes(include=[np.number])
        if np.isinf(numeric_data).any().any():
            st.warning("⚠️ The selected columns contain infinite values (infs). Please clean your data before feature selection.")
            return

        target_column = st.selectbox("Select the target column:", data.columns)

        numeric_features = data.select_dtypes(include=['number']).drop(columns=[target_column])
        target = data[target_column]
        num_available_features = numeric_features.shape[1]

        if num_available_features >= 2:
            max_num_features = min(15, num_available_features)
            num_features_to_select = st.slider("Select the number of top features to keep:", min_value=1, max_value=max_num_features, value=15, key="num_features_slider1")

            if num_features_to_select <= num_available_features:
                selector = SelectKBest(score_func=f_regression, k=num_features_to_select)
                selector.fit(numeric_features, target)

                selected_features = numeric_features.columns[selector.get_support()]
                st.write("**Selected Features:**", selected_features.tolist())
            else:
                st.warning("⚠️ You've selected more features than available (max {} features). Please adjust your selection.".format(num_available_features))
        else:
            st.warning("⚠️ There are not enough numeric columns (at least two) in the dataset for univariate feature selection.")

    def variance_threshold(data):
        st.write("#### 📉 Variance Threshold")
        st.info("Filter out low-variance features using a variance threshold.")
        
        if 'data' not in locals():
            st.warning("⚠️ Please load your dataset before using the Variance Threshold function.")
            return

        numeric_features = data.select_dtypes(include=['number'])

        if numeric_features.shape[1] >= 2:
            threshold = st.slider("Select the variance threshold:", min_value=0.0, max_value=1.0, value=0.0, key="variance_threshold_slider")

            selector = VarianceThreshold(threshold=threshold)
            selector.fit(numeric_features)

            selected_features = numeric_features.columns[selector.get_support()]
            st.write("**Selected Features:**", selected_features.tolist())
        else:
            st.warning("⚠️ There are not enough numeric columns (at least two) in the dataset for variance thresholding.")

    def perform_basic_analysis(data):
        st.header("Basic Analysis 🔍")

        correlation_analysis(data)

        st.markdown("""---""")

        univariate_feature_selection(data)

        st.markdown("""---""")

        variance_threshold(data)

    def rfe_feature_selection(data):
        st.write("#### 🔄 Recursive Feature Elimination (RFE)")
        st.info("Select a subset of the most relevant features using Recursive Feature Elimination (RFE).")

        if data.isnull().any().any():
            st.warning("⚠️ The selected columns contain missing values (NaN). Please clean your data before feature selection.")
            return

        numeric_data = data.select_dtypes(include=[np.number])
        if np.isinf(numeric_data).any().any():
            st.warning("⚠️ The selected columns contain infinite values (infs). Please clean your data before feature selection.")
            return

        target_column = st.selectbox("Select the target column:", data.columns, key="t1")
        numeric_features = data.select_dtypes(include=['number']).drop(columns=[target_column])
        target = data[target_column]
        num_available_features = numeric_features.shape[1]

        if num_available_features >= 2:
            max_num_features = min(15, num_available_features)
            num_features_to_select = st.slider("Select the number of top features to keep:", min_value=1, max_value=max_num_features, value=15, key="num_features_slider")

            if num_features_to_select <= num_available_features:
                if st.button("Start Feature Selection", key = 's5'):
                    with st.spinner(text='In progress'):
                        model = LinearRegression()
                        selector = RFE(estimator=model, n_features_to_select=num_features_to_select)
                        selector.fit(numeric_features, target)

                        selected_features = numeric_features.columns[selector.support_]
                        st.write("**Selected Features:**", selected_features.tolist())
            else:
                st.warning("⚠️ You've selected more features than available (max {} features). Please adjust your selection.".format(num_available_features))
        else:
            st.warning("⚠️ There are not enough numeric columns for RFE feature selection.")

    def tree_feature_importance(data):
        st.write("#### 🌲 Feature Importance from Trees")
        st.info("Select features based on their importance scores from tree-based models.")

        if data.isnull().any().any():
            st.warning("⚠️ The selected columns contain missing values (NaN). Please clean your data before feature selection.")
            return

        numeric_data = data.select_dtypes(include=[np.number])
        if np.isinf(numeric_data).any().any():
            st.warning("⚠️ The selected columns contain infinite values (infs). Please clean your data before feature selection.")
            return

        target_column = st.selectbox("Select the target column:", data.columns, key="t2")
        model = RandomForestClassifier()

        numeric_features = data.select_dtypes(include=['number']).drop(columns=[target_column])
        target = data[target_column]

        if numeric_features.shape[1] >= 2:
            if st.button("Start Feature Selection", key = 's6'):
                with st.spinner(text='In progress'):
                    model.fit(numeric_features, target)
                    importance_scores = model.feature_importances_

                    selected_features = numeric_features.columns[importance_scores.argsort()[::-1]]
                    st.write("**Selected Features:**", selected_features.tolist())
        else:
            st.warning("⚠️ There are not enough numeric columns (at least two) for feature importance from trees.")

    def l1_regularization(data):
        st.write("#### 🏢 L1 Regularization (Lasso)")
        st.info("Use L1 regularization for feature selection, effective for linear models.")

        if data.isnull().any().any():
            st.warning("⚠️ The selected columns contain missing values (NaN). Please clean your data before feature selection.")
            return

        numeric_data = data.select_dtypes(include=[np.number])
        if np.isinf(numeric_data).any().any():
            st.warning("⚠️ The selected columns contain infinite values (infs). Please clean your data before feature selection.")
            return

        target_column = st.selectbox("Select the target column:", data.columns, key="t3")
        model = Lasso(alpha=0.01)

        numeric_features = data.select_dtypes(include=['number']).drop(columns=[target_column])
        target = data[target_column]

        if numeric_features.shape[1] >= 2:
            if st.button("Start Feature Selection", key = 's4'):
                with st.spinner(text='In progress'):
                    model.fit(numeric_features, target)

                    selected_features = numeric_features.columns[model.coef_ != 0]
                    st.write("**Selected Features:**", selected_features.tolist())
        else:
            st.warning("⚠️ There are not enough numeric columns (at least two) for L1 regularization (Lasso) feature selection.")

    def select_percentile(data):
        st.write("#### 📊 SelectPercentile")
        st.info("Select a percentage of features based on their scores.")

        if data.isnull().any().any():
            st.warning("⚠️ The selected columns contain missing values (NaN). Please clean your data before feature selection.")
            return

        numeric_data = data.select_dtypes(include=[np.number])
        if np.isinf(numeric_data).any().any():
            st.warning("⚠️ The selected columns contain infinite values (infs). Please clean your data before feature selection.")
            return

        target_column = st.selectbox("Select the target column:", data.columns, key="t4")
        numeric_features = data.select_dtypes(include=['number']).drop(columns=[target_column])
        target = data[target_column]

        if numeric_features.shape[1] >= 2:
            percentile = st.slider("Select the percentile of features to keep:", min_value=1, max_value=100, value=10, key="percentile_slider")

            if st.button("Start Feature Selection", key = 's3'):
                with st.spinner(text='In progress'):
                    selector = SelectPercentile(score_func=f_regression, percentile=percentile)
                    selector.fit(numeric_features, target)

                    selected_features = numeric_features.columns[selector.get_support()]
                    st.write("**Selected Features:**", selected_features.tolist())
        else:
            st.warning("⚠️ There are not enough numeric columns (at least two) for SelectPercentile feature selection.")

    def select_from_model(data):
        st.write("#### 🎯 SelectFromModel")
        st.info("Select features based on importance scores from a model.")

        if data.isnull().any().any():
            st.warning("⚠️ The selected columns contain missing values (NaN). Please clean your data before feature selection.")
            return

        numeric_data = data.select_dtypes(include=[np.number])
        if np.isinf(numeric_data).any().any():
            st.warning("⚠️ The selected columns contain infinite values (infs). Please clean your data before feature selection.")
            return

        target_column = st.selectbox("Select the target column:", data.columns, key="t5")
        model = RandomForestClassifier()

        numeric_features = data.select_dtypes(include=['number']).drop(columns=[target_column])
        target = data[target_column]

        if numeric_features.shape[1] >= 2:
            if st.button("Start Feature Selection", key = 's2'):
                with st.spinner(text='In progress'):
                    selector = SelectFromModel(model)
                    selector.fit(numeric_features, target)

                    selected_features = numeric_features.columns[selector.get_support()]
                    st.write("**Selected Features:**", selected_features.tolist())
        else:
            st.warning("⚠️ There are not enough numeric columns (at least two) for SelectFromModel feature selection.")

    def rfecv_feature_selection(data):
        st.write("#### 🔄 Recursive Feature Elimination with Cross-Validation (RFECV)")
        st.info("Select the optimal number of features using RFECV.")

        if data.isnull().any().any():
            st.warning("⚠️ The selected columns contain missing values (NaN). Please clean your data before feature selection.")
            return

        numeric_data = data.select_dtypes(include=[np.number])
        if np.isinf(numeric_data).any().any():
            st.warning("⚠️ The selected columns contain infinite values (infs). Please clean your data before feature selection.")
            return

        target_column = st.selectbox("Select the target column:", data.columns, key="t6")
        model = DecisionTreeClassifier()

        numeric_features = data.select_dtypes(include=['number']).drop(columns=[target_column])
        target = data[target_column]

        if numeric_features.shape[1] >= 2:
            if st.button("Start Feature Selection", key = 's1'):
                with st.spinner(text='In progress'):
                    selector = RFECV(model, cv=5)
                    selector.fit(numeric_features, target)

                    selected_features = numeric_features.columns[selector.support_]
                    st.write("**Selected Features:**", selected_features.tolist())
        else:
            st.warning("⚠️ There are not enough numeric columns (at least two) for RFECV feature selection.")

    def perform_intermediate_analysis(data):
        st.header("Intermediate Analysis 📈")

        rfe_feature_selection(data)

        st.markdown("""---""")

        tree_feature_importance(data)

        st.markdown("""---""")

        l1_regularization(data)

        st.markdown("""---""")

        select_percentile(data)

        st.markdown("""---""")

        select_from_model(data)

        st.markdown("""---""")

        rfecv_feature_selection(data)

    def sequential_feature_selection(data):
        st.write("#### ⏭️ Sequential Feature Selection (SFS/SBS)")
        st.info("Systematically explore feature combinations using Sequential Feature Selection (SFS/SBS).")

        direction = st.radio("Select the direction:", ("Forward", "Backward"), key="sfs_direction")
        num_features_to_select = st.slider("Select the number of top features to keep:", min_value=1, max_value=15, value=5, key="num_features_slider2")
        target_column = st.selectbox("Select the target column:", data.columns, key="sfs_target_column")

        numeric_features = data.select_dtypes(include=['number']).drop(columns=[target_column])
        target = data[target_column]

        num_splits = min(5, len(target))
        
        if st.button("Select Features", key = 'ke1'):
            with st.spinner("Selecting features..."):
                if direction == 'Forward':
                    sfs = SequentialFeatureSelector(estimator=RandomForestClassifier(max_features=num_features_to_select), cv=num_splits)
                else:
                    sfs = SequentialFeatureSelector(estimator=RandomForestClassifier(max_features=num_features_to_select), cv=num_splits)

                sfs.fit(numeric_features, target)

                selected_features = numeric_features.columns[list(sfs.k_feature_idx_)]
                st.write("**Selected Features:**", selected_features.tolist())

    def sffs_feature_selection(data):
        st.write("#### ⏭️ Sequential Floating Forward Selection (SFFS)")
        st.info("Systematically explore feature combinations using Sequential Floating Forward Selection (SFFS).")

        target_column = st.selectbox("Select the target column:", data.columns, key="sffs_target_column")
        num_features_to_select = st.slider("Select the number of top features to keep:", min_value=1, max_value=15, value=5, key="num_features_slider4")

        numeric_features = data.select_dtypes(include=['number']).drop(columns=[target_column])
        target = data[target_column]

        if st.button("Select Features", key = 'ke2'):
            with st.spinner("Selecting features..."):
                sffs = SequentialFeatureSelector(estimator=RandomForestClassifier(), k_features=num_features_to_select, forward=True, floating=True)
                sffs.fit(numeric_features, target)

                selected_features = numeric_features.columns[list(sffs.k_feature_idx_)]
                st.write("**Selected Features:**", selected_features.tolist())

    def sbfs_feature_selection(data):
        st.write("#### ⏭️ Sequential Floating Backward Selection (SBFS)")
        st.info("Systematically explore feature combinations using Sequential Floating Backward Selection (SBFS).")

        target_column = st.selectbox("Select the target column:", data.columns, key="sbfs_target_column")
        num_features_to_select = st.slider("Select the number of top features to keep:", min_value=1, max_value=15, value=5, key="num_features_slider5")

        numeric_features = data.select_dtypes(include=['number']).drop(columns=[target_column])
        target = data[target_column]

        if st.button("Select Features", key = 'ke3'):
            with st.spinner("Selecting features..."):
                sffs = SequentialFeatureSelector(estimator=RandomForestClassifier(), k_features=num_features_to_select, forward=True, floating=True)
                sffs.fit(numeric_features, target)

                selected_features_forward = numeric_features.columns[list(sffs.k_feature_idx_)]
                selected_features_backward = selected_features_forward[:num_features_to_select]

                st.write("**Selected Features:**", selected_features_backward.tolist())


    def feature_selection_with_deep_learning(data):
        st.write("#### 🌐 Feature Selection with Deep Learning")
        st.info("Leverage deep learning techniques for feature selection within the model.")

        target_column = st.selectbox("Select the target column:", data.columns, key="deep_learning_target_column")
        num_features_to_select = st.slider("Select the number of top features to keep:", min_value=1, max_value=15, value=5, key="num_features_slider6")

        numeric_features = data.select_dtypes(include=['number']).drop(columns=[target_column])
        target = data[target_column]

        feature_names = numeric_features.columns.tolist()

        scaler = StandardScaler()
        numeric_features = scaler.fit_transform(numeric_features)

        if st.button("Select Features", key='ke5'):
            with st.spinner("Selecting features..."):
                model = Sequential()
                model.add(Dense(64, input_dim=numeric_features.shape[1], activation='relu'))
                model.add(Dropout(0.5))
                model.add(Dense(1, activation='sigmoid'))

                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                model.fit(numeric_features, target, epochs=50, batch_size=32)

                importance_scores = np.abs(model.layers[0].get_weights()[0])
                feature_importance = importance_scores.sum(axis=1)  
                selected_indices = np.argsort(feature_importance)[::-1][:num_features_to_select]
                selected_features = [feature_names[i] for i in selected_indices]

                st.write("**Selected Features:**", selected_features)

    def dimensionality_reduction(data):
        st.write("#### 🌌 Dimensionality Reduction (PCA/t-SNE)")
        st.info("Reduce feature dimensionality while preserving information using dimensionality reduction techniques.")

        method = st.selectbox("Select the dimensionality reduction method:", ["PCA", "t-SNE"], key="dim_reduction_method")
        num_components = 3

        numeric_features = data.select_dtypes(include=['number'])

        if st.button("Reduce Dimensionality", key = 'ke4'):
            with st.spinner(text='Reducing dimensionality...'):
                if method == 'PCA':
                    pca = PCA(n_components=num_components)
                    reduced_features = pca.fit_transform(numeric_features)
                elif method == 't-SNE':
                    tsne = TSNE(n_components=num_components)
                    reduced_features = tsne.fit_transform(numeric_features)

                reduced_df = pd.DataFrame(reduced_features, columns=[f"Component {i+1}" for i in range(num_components)])
                st.write("Reduced Features:", reduced_df)

    def perform_advanced_analysis(data):
        st.header("Advanced Analysis 🚀")

        sequential_feature_selection(data)

        st.markdown("""---""")

        sffs_feature_selection(data)

        st.markdown("""---""")

        sbfs_feature_selection(data)

        st.markdown("""---""")

        feature_selection_with_deep_learning(data)

        st.markdown("""---""")

        dimensionality_reduction(data)


    def main():
        st.title("🏅 Best Feature Analysis on Dataset")

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
