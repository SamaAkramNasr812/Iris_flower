Here's the updated README for your Iris Flower Classification Project incorporating the second code snippet for the Streamlit application:

```markdown
# Iris Flower Classification Project

## Description
The Iris Flower Classification Project is a machine learning application that classifies iris flowers into three species based on their sepal and petal dimensions. This project utilizes a Support Vector Machine (SVM) model to predict the species of iris flowers using the well-known Iris dataset. The application features an interactive Streamlit interface that allows users to input flower dimensions and receive predictions.

## Live Demo
You can access the live demo of the app here: [Iris Flower Classification App](https://vuydnwft28t8xte9rhdewf.streamlit.app/).

## GitHub Repository
The source code for this project is available on GitHub: [Iris Flower Classification GitHub](https://github.com/SamaAkramNasr812/Iris_flower).

## Installation Instructions
To run the Iris Flower Classification Project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SamaAkramNasr812/Iris_flower.git
   cd Iris_flower
   ```

2. **Install dependencies**:
   Ensure you have Python installed, then create a virtual environment and install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Data Analysis and Visualization
The project starts by loading the Iris dataset and performing basic statistical analysis. It visualizes the dataset using bar charts to show the average dimensions of each species.

### Streamlit Application
The Streamlit app allows users to:
1. View the Iris dataset and its statistical summary.
2. Visualize the average feature values for each iris species.
3. Input flower dimensions to predict the species.

### Code Overview
Here is the main code for the Streamlit application:

```python
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

# Plot the average
st.write("### Average Feature Values for Each Class")
fig, ax = plt.subplots()
X_axis = np.arange(len(columns) - 1)
width = 0.25
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
```

### Features
- Data loading and preprocessing.
- Data visualization using Matplotlib and Seaborn.
- SVM model training and evaluation.
- Interactive prediction of iris species based on user input.

## Technologies Used
- **Programming Language**: Python
- **Framework**: Streamlit
- **Libraries**: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, Pickle

## Contributing
Contributions are welcome! If you would like to contribute to this project, please follow these guidelines:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push your branch and submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements
- Special thanks to the contributors of the libraries used in this project.
- Inspiration from various machine learning resources and tutorials.

## Contact Information
For any inquiries or feedback, please reach out via GitHub: [SamaAkramNasr812](https://github.com/SamaAkramNasr812).

## Additional Notes
Feel free to explore the code and modify the project as per your requirements. Feedback and suggestions for improvement are always welcome!
```

### Notes:
- Ensure that the `requirements.txt` file is updated to include all necessary dependencies for your Streamlit app.
- Adjust any specific details or sections as needed based on your project requirements.
