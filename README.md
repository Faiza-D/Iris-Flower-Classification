Iris Species Prediction with KNN and Tkinter GUI

Project Overview

This project implements a K-Nearest Neighbors (KNN) classifier to predict the species of an Iris flower based on its sepal and petal dimensions. It also includes a user-friendly Tkinter GUI to allow interactive predictions.

Requirements

Python 3.x
pandas
seaborn
matplotlib
scikit-learn
tkinter
Installation

You can install the required libraries using pip:

Bash
pip install pandas seaborn matplotlib scikit-learn tkinter
Use code with caution.

Data

The project assumes you have a CSV file named iris.csv containing the Iris flower dataset with the following columns:

sepal_length (cm)
sepal_width (cm)
petal_length (cm)
petal_width (cm)
species (Iris-setosa, Iris-versicolor, Iris-virginica)
Replace the path in the code ('C:/PythonProject/iris flower/iris.csv') with the actual location of your CSV file.

Running the Project

Save the code as iris_prediction.py.
Open a terminal or command prompt and navigate to the directory containing the script and your CSV file.
Run the script using the following command:
Bash
python iris_prediction.py
Use code with caution.

This will perform the following:

Load the Iris dataset from the CSV file.
Explore the data (shape, class distribution, feature statistics).
Visualize the data using Seaborn pairplots and scatterplots.
Split the data into training and testing sets.
Train a KNN classifier with hyperparameter tuning (default settings used in this example).
Evaluate the model performance using accuracy, classification report, and confusion matrix.
Run the Tkinter GUI for interactive prediction.
Using the GUI

The GUI will appear titled "Iris Species Prediction".
Enter values for sepal length, sepal width, petal length, and petal width.
Click the "Predict" button.
The predicted species will be displayed in the result label.
Note: The current code uses default hyperparameter values for the KNN classifier. You can explore hyperparameter tuning to potentially improve the model's performance.

Further Exploration

Explore different machine learning algorithms for Iris species prediction (e.g., Decision Trees, Support Vector Machines).
Implement cross-validation to obtain more robust model evaluation.
Enhance the GUI with additional functionalities or visualizations.
License

This project is distributed under the (insert your preferred license, e.g., MIT License). See the LICENSE file for details.
