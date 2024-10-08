# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Set up the title
st.title("Iris Flower Classification with SVM")

# Define the columns globally
columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class_labels']

# Load the data
@st.cache_data  # Updated caching method
def load_data():
    df = pd.read_csv('iris.data', names=columns)
    return df

# Load the dataset
df = load_data()
st.write("### Iris Dataset")
st.dataframe(df.head())

# Basic statistical analysis
st.write("### Statistical Summary")
st.write(df.describe())

# Separate features and target
data = df.values
X = data[:, 0:4]
Y = data[:, 4]

# Calculate average of each feature for all classes
Y_Data = np.array([np.average(X[:, i][Y == j].astype('float32')) for i in range(X.shape[1]) for j in np.unique(Y)])
Y_Data_reshaped = Y_Data.reshape(4, 3)
Y_Data_reshaped = np.swapaxes(Y_Data_reshaped, 0, 1)

# Use the global columns variable here
X_axis = np.arange(len(columns) - 1)
width = 0.25

# Plot the average
st.write("### Average Feature Values for Each Class")
fig, ax = plt.subplots()
ax.bar(X_axis, Y_Data_reshaped[0], width, label='Setosa')
ax.bar(X_axis + width, Y_Data_reshaped[1], width, label='Versicolor')
ax.bar(X_axis + width * 2, Y_Data_reshaped[2], width, label='Virginica')
ax.set_xticks(X_axis)
ax.set_xticklabels(columns[:4])
ax.set_xlabel("Features")
ax.set_ylabel("Value in cm.")
ax.legend(bbox_to_anchor=(1.3, 1))
st.pyplot(fig)

# Split the data into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Support Vector Machine algorithm
svn = SVC()
svn.fit(X_train, y_train)

# Predict from the test dataset
predictions = svn.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, predictions)
st.write(f"### Model Accuracy: {accuracy:.2f}")

# Detailed classification report
st.write("### Classification Report")
report = classification_report(y_test, predictions, output_dict=True)
st.text(classification_report(y_test, predictions))

# Input for new predictions
st.write("### Predict the Species")
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.5)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)

if st.button("Predict"):
    X_new = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = svn.predict(X_new)
    st.write(f"**Prediction of Species:** {prediction[0]}")


