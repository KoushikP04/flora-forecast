from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#Loading the Dataset
iris = load_iris()

#Preparing the Data (Input Features & Target Labels)
X = iris.data
y = iris.target

#Splitting the data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#Training the ML Model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Evaluate the model
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

# Make predictions
X_new = [[5.0, 3.5, 1.3, 0.2], [6.7, 3.1, 4.4, 1.4], [6.0, 3.0, 4.8, 1.8]]
y_pred = clf.predict(X_new)
print(f"Predictions: {y_pred}")


