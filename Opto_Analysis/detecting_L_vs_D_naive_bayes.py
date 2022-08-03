from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

##################### GUIDE #####################

"""
Dataset:
    - Feature matrix
        - contains all the vectors, rows of the dataset in which each vector consists of the value of dependent features
        - So in this case rows will be pixels 
    - Response vector
"""

##################### GUIDE #####################


iris = load_iris()

# store the feature matrix (X) and response vector (y)
X = iris.data
y = iris.target

# Splitting X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

# Training the model on training set
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# making predictions on the testing set
y_pred = gnb.predict(X_test)

print("Gaussian Naive Bayes model accuracy (in %): ", metrics.accuracy_score(y_test, y_pred)*100)


