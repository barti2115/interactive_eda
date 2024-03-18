import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Turn off warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
target_names = iris.target_names

# Streamlit App
st.title('Iris Dataset Exploratory Data Analysis')

# Sidebar
st.sidebar.title('Customize Visualization')
selected_feature = st.sidebar.selectbox('Select feature for analysis', iris.feature_names)

# Main content
st.write('## Overview of Iris Dataset')
st.write(iris_df.head())

st.write('## Summary Statistics')
st.write(iris_df.describe())

# Histogram
st.write('## Histogram of Selected Feature')
plt.figure(figsize=(8, 6))
sns.histplot(data=iris_df, x=selected_feature, kde=True)
st.pyplot()

# Pairplot
st.write('## Pairplot of Features')
plt.figure(figsize=(10, 8))
sns.pairplot(iris_df, hue='target', palette='viridis')
st.pyplot()

# Boxplot
st.write('## Boxplot of Selected Feature by Target')
plt.figure(figsize=(8, 6))
sns.boxplot(x='target', y=selected_feature, data=iris_df)
plt.xlabel('Target')
plt.ylabel(selected_feature)
st.pyplot()
