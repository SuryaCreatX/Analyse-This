import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from wordcloud import WordCloud
from mpl_toolkits.mplot3d import Axes3D
from pandas.plotting import parallel_coordinates
import streamlit.components.v1 as components
from PIL import Image 

def run_visualization_analysis():

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

    info = Image.open("Images/visual.png")
    '''st.set_page_config(
            page_title="Visualization Analysis",
            page_icon=info,
            #layout="wide",
        )'''


    st.markdown(hide_menu, unsafe_allow_html=True)

    def plot_histogram(data):
        st.write(" #### 🪄  Histogram")
        st.info('Display a histogram of the selected numeric column')
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.warning("⚠️ There are no numeric columns for Histogram analysis.")
            return
        
        predictor_col = st.selectbox("Select Predictor Column:", numeric_cols, key='histogram_column_select')

        plt.figure(figsize=(8, 6))
        sns.histplot(data, x=predictor_col, kde=True)
        plt.xlabel(predictor_col)
        plt.ylabel("Frequency")
        plt.title(f"Histogram of {predictor_col}")
        st.pyplot(plt)

    def plot_bar_chart(data):
        st.write("#### 🪄 Bar Chart")
        st.info("Visualize data using a bar chart to compare categories or discrete data.")
        
        categorical_cols = data.select_dtypes(include='object').columns.tolist()
        
        if not categorical_cols:
            st.warning("⚠️ There are no categorical columns for Bar Chart analysis.")
            return
        
        category_col = st.selectbox("Select Category Column:", categorical_cols, key='bar_chart_category_select')
        
        plt.figure(figsize=(8, 6))
        data[category_col].value_counts().plot(kind='bar')
        plt.xlabel(category_col)
        plt.ylabel("Frequency")
        plt.title(f"Bar Chart of {category_col}")
        st.pyplot(plt)

    def plot_line_chart(data):
        st.write("#### 🪄 Line Chart")
        st.info("Visualize trends over time or continuous data using a line chart.")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.warning("⚠️ There are no numeric columns for Line Chart analysis.")
            return
        
        x_col = st.selectbox("Select X-Axis Column:", numeric_cols, key='line_chart_x_axis_select')
        y_col = st.selectbox("Select Y-Axis Column:", numeric_cols, key='line_chart_y_axis_select')
        
        plt.figure(figsize=(8, 6))
        plt.plot(data[x_col], data[y_col])
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"Line Chart of {x_col} vs {y_col}")
        st.pyplot(plt)

    def plot_scatter_plot(data):
        st.write("#### 🪄 Scatter Plot")
        st.info("Visualize relationships between two numerical variables using a scatter plot.")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.warning("⚠️ There are not enough numeric columns (at least 2) for Scatter Plot analysis.")
            return
        
        x_col = st.selectbox("Select X-Axis Column:", numeric_cols, key='scatter_plot_x_axis_select')
        y_col = st.selectbox("Select Y-Axis Column:", numeric_cols, key='scatter_plot_y_axis_select')
        
        plt.figure(figsize=(8, 6))
        plt.scatter(data[x_col], data[y_col])
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"Scatter Plot of {x_col} vs {y_col}")
        st.pyplot(plt)

    def plot_pie_chart(data):
        st.write("#### 🪄 Pie Chart")
        st.info("Visualize the composition of a whole using a pie chart.")
        
        categorical_cols = data.select_dtypes(include='object').columns.tolist()
        
        if not categorical_cols:
            st.warning("⚠️ There are no categorical columns for Pie Chart analysis.")
            return
        
        category_col = st.selectbox("Select Category Column:", categorical_cols, key='pie_chart_category_select')
        
        plt.figure(figsize=(8, 6))
        data[category_col].value_counts().plot(kind='pie', autopct='%1.1f%%')
        plt.title(f"Pie Chart of {category_col}")
        st.pyplot(plt)

    def plot_box_plot(data):
        st.write("#### 🪄 Box Plot (Box-and-Whisker Plot)")
        st.info("Visualize the distribution of a numerical variable using a box plot.")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.warning("⚠️ There are no numeric columns for Box Plot analysis.")
            return
        
        numeric_col = st.selectbox("Select Numeric Column:", numeric_cols, key='box_plot_numeric_column_select')
        
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=data, x=numeric_col)
        plt.xlabel(numeric_col)
        plt.title(f"Box Plot of {numeric_col}")
        st.pyplot(plt)

    def plot_heatmap(data):
        st.write("#### 🪄 Heatmap")
        st.info("Visualize relationships between variables in a matrix format using a heatmap.")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.warning("⚠️ There are not enough numeric columns (at least 2) for Heatmap analysis.")
            return
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
        plt.title("Heatmap of Variable Correlations")
        st.pyplot(plt)

    def plot_area_chart(data):
        st.write("#### 🪄 Area Chart")
        st.info("Visualize stacked data and compare the contribution of categories over time using an area chart.")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.warning("⚠️ There are no numeric columns for Area Chart analysis.")
            return
        
        x_col = st.selectbox("Select X-Axis Column:", numeric_cols, key='area_chart_x_axis_select')
        y_col = st.selectbox("Select Y-Axis Column:", numeric_cols, key='area_chart_y_axis_select')
        
        try:
            plt.figure(figsize=(8, 6))
            data.plot(x=x_col, y=y_col, kind='area')  
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(f"Area Chart of {x_col} vs {y_col}")
            st.pyplot(plt)
        except KeyError:
            st.warning(f"⚠️ One or both of the selected columns, {x_col} and {y_col}, do not exist in the DataFrame.")


    def perform_basic_analysis(data):
        st.header("Basic Visual Analysis 🔍")

        plot_histogram(data)

        st.markdown("""---""")

        plot_bar_chart(data)

        st.markdown("""---""")

        plot_line_chart(data)

        st.markdown("""---""")

        plot_scatter_plot(data)

        st.markdown("""---""")

        plot_pie_chart(data)

        st.markdown("""---""")

        plot_box_plot(data)

        st.markdown("""---""")

        plot_heatmap(data)

        st.markdown("""---""")

        plot_area_chart(data)

    def plot_violin_plot(data):
        st.write("#### 🌱 Violin Plot")
        st.info("Visualize the distribution of a variable using a violin plot.")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.warning("⚠️ There are no numeric columns for Violin Plot analysis.")
            return
        
        numeric_col = st.selectbox("Select Numeric Column:", numeric_cols, key='violin_plot_numeric_column_select')
        
        plt.figure(figsize=(8, 6))
        sns.violinplot(x=data[numeric_col])
        plt.xlabel(numeric_col)
        plt.title(f"Violin Plot of {numeric_col}")
        st.pyplot(plt)

    def plot_grouped_bar_chart(data):
        st.write("#### 🌱 Grouped Bar Chart")
        st.info("Compare multiple categories across subgroups within each category using a grouped bar chart.")
        
        categorical_cols = data.select_dtypes(include='object').columns.tolist()
        
        if len(categorical_cols) < 2:
            st.warning("⚠️ There are not enough categorical columns (at least 2) for Grouped Bar Chart analysis.")
            return
        
        group_col = st.selectbox("Select Grouping Column:", categorical_cols, key='grouped_bar_chart_group_column_select')
        subcategory_col = st.selectbox("Select Subcategory Column:", categorical_cols, key='grouped_bar_chart_subcategory_column_select')
        
        grouped_data = data.groupby([group_col, subcategory_col]).size().unstack().fillna(0)
        grouped_data.plot(kind='bar', stacked=True, figsize=(8, 6))
        plt.xlabel(group_col)
        plt.ylabel("Count")
        plt.title(f"Grouped Bar Chart of {group_col} with {subcategory_col} Subcategories")
        st.pyplot(plt)

    def plot_stacked_bar_chart(data):
        st.write("#### 🌱 Stacked Bar Chart")
        st.info("Show the composition of a whole by stacking subcategories on top of each other using a stacked bar chart.")
        
        categorical_cols = data.select_dtypes(include='object').columns.tolist()
        
        if len(categorical_cols) < 2:
            st.warning("⚠️ There are not enough categorical columns (at least 2) for Stacked Bar Chart analysis.")
            return
        
        category_col = st.selectbox("Select Category Column:", categorical_cols, key='stacked_bar_chart_category_select')
        subcategory_col = st.selectbox("Select Subcategory Column:", categorical_cols, key='stacked_bar_chart_subcategory_column_select')
        
        stacked_data = data.groupby([category_col, subcategory_col]).size().unstack().fillna(0)
        stacked_data.plot(kind='bar', stacked=True, figsize=(8, 6))
        plt.xlabel(category_col)
        plt.ylabel("Count")
        plt.title(f"Stacked Bar Chart of {category_col} with {subcategory_col} Subcategories")
        st.pyplot(plt)

    def plot_bubble_chart(data):
        st.write("#### 🌱 Bubble Chart")
        st.info("Explore relationships among three numerical variables using a bubble chart.")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 3:
            st.warning("⚠️ There are not enough numeric columns (at least 3) for Bubble Chart analysis.")
            return
        
        x_col = st.selectbox("Select X-Axis Column:", numeric_cols, key='bubble_chart_x_axis_select')
        y_col = st.selectbox("Select Y-Axis Column:", numeric_cols, key='bubble_chart_y_axis_select')
        size_col = st.selectbox("Select Size Column:", numeric_cols, key='bubble_chart_size_column_select')
        
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=x_col, y=y_col, size=size_col, data=data)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"Bubble Chart of {x_col} vs {y_col} (Sized by {size_col})")
        st.pyplot(plt)

    def plot_dendrogram(data):
        st.write("#### 🌱 Dendrogram")
        st.info("Visualize hierarchical clustering relationships in data using a dendrogram.")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.warning("⚠️ There are no numeric columns for Dendrogram analysis.")
            return
        
        plt.figure(figsize=(8, 6))
        sns.clustermap(data[numeric_cols].corr(), cmap='coolwarm', figsize=(8, 6))
        plt.title("Dendrogram of Hierarchical Clustering")
        st.pyplot(plt)


    def plot_radar_chart(data):
        st.write("#### 🌱 Radar Chart (Spider Chart)")
        st.info("Compare multiple quantitative variables on a common scale using a radar chart.")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.warning("⚠️ There are not enough numeric columns (at least 2) for Radar Chart analysis.")
            return
        
        category_col = st.selectbox("Select Category Column:", numeric_cols, key='radar_chart_category_select')
        variable_cols = st.multiselect("Select Quantitative Variables:", numeric_cols, key='radar_chart_variable_select')
        
        if not variable_cols:
            st.warning("⚠️ Please select at least two quantitative variables for Radar Chart analysis.")
            return

        radar_data = data[[category_col] + variable_cols]

        fig = px.line_polar(
            radar_data, r=variable_cols, theta=variable_cols, line_close=True,
            color_discrete_sequence=['rgb(255, 127, 14)'],  
        )

        fig.update_polars(
            radialaxis=dict(
                visible=True,
                range=[0, radar_data[variable_cols].max().max()],  
            ),
        )

        fig.update_layout(
            polar=dict(
                bgcolor='rgba(0, 0, 0, 0.7)',  
            ),
        )

        fig.update_traces(fill='toself', fillcolor='rgba(255, 127, 14, 0.5)')  

        st.plotly_chart(fig)

    def perform_intermediate_analysis(data):
        st.header("Intermediate Visual Analysis 📈")

        plot_violin_plot(data)

        st.markdown("""---""")

        plot_grouped_bar_chart(data)

        st.markdown("""---""")

        plot_stacked_bar_chart(data)

        st.markdown("""---""")

        plot_bubble_chart(data)

        st.markdown("""---""")

        plot_dendrogram(data)

        st.markdown("""---""")

        plot_radar_chart(data)

    def plot_3d(data):
        st.write("#### 🧩 3D Plot")
        st.info("Visualize three-dimensional relationships between variables in a 3D space.")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 3:
            st.warning("⚠️ There are not enough numeric columns (at least 3) for 3D Plot analysis.")
            return
        
        x_col = st.selectbox("Select X-Axis Column:", numeric_cols, key='3d_plot_x_axis_select')
        y_col = st.selectbox("Select Y-Axis Column:", numeric_cols, key='3d_plot_y_axis_select')
        z_col = st.selectbox("Select Z-Axis Column:", numeric_cols, key='3d_plot_z_axis_select')
        
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[x_col], data[y_col], data[z_col], c='b', marker='o')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_zlabel(z_col)
        ax.set_title(f"3D Plot of {x_col}, {y_col}, and {z_col}")
        st.pyplot(fig)

    def plot_parallel_coordinates(data):
        st.write("#### 🧩 Parallel Coordinates Plot")
        st.info("Display multivariate data by plotting each variable on a separate axis.")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.warning("⚠️ There are no numeric columns for Parallel Coordinates Plot analysis.")
            return
        
        st.write("Select numeric columns for the Parallel Coordinates Plot:")
        selected_cols = st.multiselect("Select Columns:", numeric_cols, key='parallel_coordinates_select')
        
        if len(selected_cols) < 2:
            st.warning("⚠️ Select at least 2 numeric columns for Parallel Coordinates Plot.")
            return

        target_col = st.selectbox("Select Target Column:", numeric_cols, key='parallel_coordinates_target_select')
        
        fig = plt.figure(figsize=(8, 6))
        parallel_coordinates(data[selected_cols + [target_col]], target_col, colormap=plt.get_cmap("Set2"))
        plt.title(f"Parallel Coordinates Plot (Target: {target_col})")
        st.pyplot(fig)

    def plot_streamgraph(data):
        st.write("#### 🧩 Streamgraph")
        st.info("Visualize changes in data over time, showing trends and patterns in a stacked area chart format.")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.warning("⚠️ There are not enough numeric columns (at least 2) for Streamgraph analysis.")
            return
        
        time_col = st.selectbox("Select Time Column:", numeric_cols, key='streamgraph_time_select')
        value_cols = st.multiselect("Select Value Columns:", numeric_cols, key='streamgraph_value_select')
        
        if len(value_cols) < 1:
            st.warning("⚠️ Select at least one value column for Streamgraph analysis.")
            return
        
        traces = []
        for value_col in value_cols:
            trace = go.Scatter(
                x=data[time_col],
                y=data[value_col],
                fill='tonexty',
                mode='none',
                name=value_col
            )
            traces.append(trace)
        
        layout = go.Layout(
            title=f"Streamgraph of {', '.join(value_cols)} Over Time",
            xaxis=dict(title=time_col),
            yaxis=dict(title="Value"),
            hovermode='closest',
            showlegend=True
        )
        
        fig = go.Figure(data=traces, layout=layout)
        
        st.plotly_chart(fig)

    def plot_polar_plot(data):
        st.write("#### 🧩 Polar Plot")
        st.info("Display data on a circular axis, ideal for visualizing periodic data.")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.warning("⚠️ There are not enough numeric columns (at least 2) for Polar Plot analysis.")
            return
        
        angle_col = st.selectbox("Select Angle Column:", numeric_cols, key='polar_plot_angle_select')
        value_col = st.selectbox("Select Value Column:", numeric_cols, key='polar_plot_value_select')
        
        fig = go.Figure(data=go.Scatterpolar(
            r=data[value_col],
            theta=data[angle_col],
            mode='markers'
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(data[value_col])]
                )
            ),
            title=f"Polar Plot of {value_col} vs {angle_col}"
        )
        st.plotly_chart(fig)

    def perform_advanced_analysis(data):
        st.header("Advanced Visual Analysis 🚀")

        plot_3d(data)

        st.markdown("""---""")

        plot_parallel_coordinates(data)

        st.markdown("""---""")

        plot_streamgraph(data)

        st.markdown("""---""")

        plot_polar_plot(data)

    def scatter_plot(data):
        st.write("#### Scatter Plot")
        target_column = st.selectbox("Select Target Column:", data.select_dtypes(['int64', 'float64']).columns.tolist(), key = 'plt1')
        predictor_col = st.selectbox("Select Predictor Column:", data.select_dtypes(['int64', 'float64']).columns.tolist(), key = 'plt2')

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=predictor_col, y=target_column, data=data)
        plt.title(f"Scatter Plot: {target_column} vs {predictor_col}")
        st.pyplot(plt)

    def line_plot(data):
        st.write("#### 🧰 Line Plot")
        target_column = st.selectbox("Select Target Column:", data.select_dtypes(['int64', 'float64']).columns.tolist(), key = 'plt3')
        predictor_col = st.selectbox("Select Predictor Column:", data.select_dtypes(['int64', 'float64']).columns.tolist(), key = 'plt4')

        plt.figure(figsize=(8, 6))
        sns.lineplot(x=predictor_col, y=target_column, data=data)
        plt.title(f"Line Plot: {target_column} vs {predictor_col}")
        st.pyplot(plt)

    def bar_chart(data):
        st.write("#### 🧰 Bar Chart")
        target_column = st.selectbox("Select Target Column:", data.select_dtypes(['int64', 'float64']).columns.tolist(), key = 'plt5')
        predictor_col = st.selectbox("Select Predictor Column:", data.select_dtypes(['object']).columns.tolist(), key = 'plt6')

        plt.figure(figsize=(8, 6))
        sns.barplot(x=predictor_col, y=target_column, data=data)
        plt.title(f"Bar Chart: {target_column} vs {predictor_col}")
        st.pyplot(plt)

    def box_plot(data):
        st.write("#### 🧰 Box Plot")
        target_column = st.selectbox("Select Target Column:", data.select_dtypes(['int64', 'float64']).columns.tolist(), key = 'plt7')
        predictor_col = st.selectbox("Select Predictor Column:", data.select_dtypes(['int64', 'float64']).columns.tolist(), key = 'plt8')

        plt.figure(figsize=(8, 6))
        sns.boxplot(x=predictor_col, y=target_column, data=data)
        plt.title(f"Box Plot: {target_column} vs {predictor_col}")
        st.pyplot(plt)

    def line_chart_multiple_lines(data):
        st.write("#### 🧰 Line Chart with Multiple Lines")
        target_column = st.selectbox("Select Target Column:", data.select_dtypes(['object']).columns.tolist(), key = 'plt9')
        predictor_col = st.selectbox("Select Predictor Column:", data.select_dtypes(['int64', 'float64']).columns.tolist(), key = 'plt10')

        plt.figure(figsize=(8, 6))
        sns.lineplot(x=predictor_col, y=target_column, hue=target_column, data=data)
        plt.title(f"Line Chart with Multiple Lines: {target_column} vs {predictor_col}")
        st.pyplot(plt)

    def stacked_bar_chart(data):
        st.write("#### 🧰 Stacked Bar Chart")
        target_column = st.selectbox("Select Target Column:", data.select_dtypes(['object']).columns.tolist(), key = 'plt11')
        predictor_col = st.selectbox("Select Predictor Column:", data.select_dtypes(['object']).columns.tolist(), key = 'plt12')

        plt.figure(figsize=(8, 6))
        data_pivot = data.pivot_table(index=predictor_col, columns=target_column, aggfunc='size', fill_value=0)
        data_pivot.plot(kind='bar', stacked=True)
        plt.title(f"Stacked Bar Chart: {target_column} vs {predictor_col}")
        st.pyplot(plt)

    def grouped_bar_chart(data):
        st.write("#### 🧰 Grouped Bar Chart")
        target_column = st.selectbox("Select Target Column:", data.select_dtypes(['object']).columns.tolist(), key = 'plt13')
        predictor_col = st.selectbox("Select Predictor Column:", data.select_dtypes(['int64', 'float64']).columns.tolist(), key = 'plt14')

        plt.figure(figsize=(8, 6))
        sns.barplot(x=predictor_col, y=target_column, hue=target_column, data=data)
        plt.title(f"Grouped Bar Chart: {target_column} vs {predictor_col}")
        st.pyplot(plt)


    def countplot(data):
        st.write("#### 🧰 Countplot")
        target_column = st.selectbox("Select Target Column:", data.select_dtypes(['object']).columns.tolist(), key = 'plt19')
        predictor_col = st.selectbox("Select Predictor Column:", data.select_dtypes(['object']).columns.tolist(), key = 'plt20')

        plt.figure(figsize=(8, 6))
        sns.countplot(x=predictor_col, hue=target_column, data=data)
        plt.title(f"Countplot: {target_column} vs {predictor_col}")
        st.pyplot(plt)

    def residual_plot(data):
        st.write("#### 🧰 Residual Plot")
        target_column = st.selectbox("Select Target Column:", data.select_dtypes(['int64', 'float64']).columns.tolist())
        predictor_col = st.selectbox("Select Predictor Column:", data.select_dtypes(['int64', 'float64']).columns.tolist())

        plt.figure(figsize=(8, 6))
        sns.residplot(x=predictor_col, y=target_column, data=data, lowess=True)
        plt.title(f"Residual Plot: {target_column} vs {predictor_col}")
        st.pyplot(plt)

    def generate_plot(data):
        st.info("Visualize custom columns with the target column.")

        plot_type = st.selectbox("Select Visualization Type:", ['Scatter Plot', 'Line Plot', 'Bar Chart', 'Box Plot',
                                                                'Line Chart with Multiple Lines', 'Stacked Bar Chart',
                                                                'Grouped Bar Chart', 'Countplot', 'Residual Plot'])

        if plot_type == 'Scatter Plot':
            scatter_plot(data)
        elif plot_type == 'Line Plot':
            line_plot(data)
        elif plot_type == 'Bar Chart':
            bar_chart(data)
        elif plot_type == 'Box Plot':
            box_plot(data)
        elif plot_type == 'Line Chart with Multiple Lines':
            line_chart_multiple_lines(data)
        elif plot_type == 'Stacked Bar Chart':
            stacked_bar_chart(data)
        elif plot_type == 'Grouped Bar Chart':
            grouped_bar_chart(data)
        elif plot_type == 'Countplot':
            countplot(data)
        elif plot_type == 'Residual Plot':
            residual_plot(data)
        else:
            st.warning("⚠️ Invalid plot type selected.")

    def perform_super_advanced_analysis(data):
        st.header("Visual Analysis v/s Target 🎯")

        target_column = st.selectbox('Select the target column:', data.columns, key = 'target')

        st.write('#### 💻 Correlation Matrix')
        if st.checkbox('Show Correlation Matrix'):
            correlation_matrix = data.corr()
            st.write(correlation_matrix[target_column])

        st.markdown("""---""")

        st.write('#### 🔄 Multi - Target Visualization ')
        generate_plot(data)
        
    def main():
        st.title("🌈 Visualization Analysis on Dataset")

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

                    with st.expander("🎯 Visual Analysis v/s Target"):
                        perform_super_advanced_analysis(data)


    if __name__ == "__main__":
        main()