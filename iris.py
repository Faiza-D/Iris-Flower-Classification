import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tkinter as tk

# Load the Iris dataset from the CSV file
df = pd.read_csv('C:/PythonProject/iris flower/iris.csv')

# Explore the dataset
print("dataset: ", df.head())
print("Dataset Shape:", df.shape)
print("Class Distribution:")
print(df['species'].value_counts())
print("Feature Statistics:")
print(df.describe())
print(df.info())
# Visualize the data with Seaborn
sns.set(style="whitegrid")

# Pairplot to visualize relationships between features
sns.pairplot(df, hue='species', diag_kind='hist')
plt.title('Pairplot of Iris Features')
plt.show()

# Scatterplot of petal length vs. petal width
sns.scatterplot(x='petal_length', y='petal_width', data=df, hue='species')
plt.title('Petal Length vs. Petal Width')
plt.show()

# Split the data into training and testing sets
X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a KNN classifier with hyperparameter tuning
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

classification_rep = classification_report(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

print('Accuracy:', accuracy)
print('Classification Report:\n', classification_rep)
print('Confusion Matrix:\n', confusion_mat)

# Create a prediction function
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    new_sample = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=X.columns)
    predicted_species = knn.predict(new_sample)[0]
    return predicted_species



# Create the Tkinter GUI
def create_gui():
    window = tk.Tk()
    window.title("Iris Species Prediction")

    # Labels and entry fields
    sepal_length_label = tk.Label(window, text="Sepal Length (cm):")
    sepal_length_entry = tk.Entry(window)
    sepal_length_label.grid(row=0, column=0)
    sepal_length_entry.grid(row=0, column=1)

    sepal_width_label = tk.Label(window, text="Sepal Width (cm):")
    sepal_width_entry = tk.Entry(window)
    sepal_width_label.grid(row=1, column=0)
    sepal_width_entry.grid(row=1, column=1)

    petal_length_label = tk.Label(window, text="Petal Length (cm):")
    petal_length_entry = tk.Entry(window)
    petal_length_label.grid(row=2, column=0)
    petal_length_entry.grid(row=2, column=1)

    petal_width_label = tk.Label(window, text="Petal Width (cm):")
    petal_width_entry = tk.Entry(window)
    petal_width_label.grid(row=3, column=0)
    petal_width_entry.grid(row=3, column=1)

    def predict():
        sepal_length = float(sepal_length_entry.get())
        sepal_width = float(sepal_width_entry.get())
        petal_length = float(petal_length_entry.get())
        petal_width = float(petal_width_entry.get())
        predicted_species = predict_species(sepal_length, sepal_width, petal_length, petal_width)
        result_label.config(text="Predicted Species: " + predicted_species)

    # Prediction button
    predict_button = tk.Button(window, text="Predict", command=predict)
    predict_button.grid(row=4, columnspan=2)

    # Result label
    result_label = tk.Label(window, text="")
    result_label.grid(row=5, columnspan=2)


    window.mainloop()


# Start the GUI
create_gui()