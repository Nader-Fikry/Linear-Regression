import pandas as pd  # to take the data from the spreadsheet.
import numpy as np   # to work with the matrices operations.
from sklearn.preprocessing import LabelEncoder  # to encode the data.
from sklearn.model_selection import train_test_split  # method to split the data into train and test.


def hypothesis(theta, features):
    hyp = theta[:, 0] + theta[:, 1] * features[:, 0] + theta[:, 2] * features[:, 1] + theta[:, 3] * features[:, 2] + theta[:, 4] * features[:, 3] + theta[:, 5] * features[:, 4] + theta[:, 6] * features[:, 5]
    return hyp

def meanSquaredError(theta, features, y_actual):
    n = len(features)
    y_predicted = hypothesis(theta, features)  # Make prediction.
    cost = (1 / n) * sum([val**2 for val in (y_actual - y_predicted)])
    print('The accuracy is:', cost)

def gradientDescent(theta, feature):
    learning_rate = 0.0001  # Learning rate.
    length = len(feature)
    y_predicted = hypothesis(theta, feature)  # Initial value for the hypothesis.

    # Rub the gradient descent for two times.
    for i in range(2):
        theta[:, 0] = theta[:, 0] - learning_rate * (-(2 / length) * sum(y_train - y_predicted))
        theta[:, 1] = theta[:, 1] - learning_rate * (-(2 / length) * sum(x_train[:, 0] * (y_train - y_predicted)))
        theta[:, 2] = theta[:, 2] - learning_rate * (-(2 / length) * sum(x_train[:, 1] * (y_train - y_predicted)))
        theta[:, 3] = theta[:, 3] - learning_rate * (-(2 / length) * sum(x_train[:, 2] * (y_train - y_predicted)))
        theta[:, 4] = theta[:, 4] - learning_rate * (-(2 / length) * sum(x_train[:, 3] * (y_train - y_predicted)))
        theta[:, 5] = theta[:, 5] - learning_rate * (-(2 / length) * sum(x_train[:, 4] * (y_train - y_predicted)))
        theta[:, 6] = theta[:, 6] - learning_rate * (-(2 / length) * sum(x_train[:, 5] * (y_train - y_predicted)))
        meanSquaredError(theta, feature, y_train)  # Call the loss method to see the accuracy increase with each interation.
    return theta


# Load the dataset.
dataset = pd.read_csv("insurance.csv")
input_data = np.array(dataset.iloc[:, : -1].values)
output_data = np.array(dataset.iloc[:, 6].values)

# Encode the data, and transfer it from string to numbers.
encoder = LabelEncoder()
input_data[:, 1] = encoder.fit_transform(input_data[:, 1])
input_data[:, 4] = encoder.fit_transform(input_data[:, 4])
input_data[:, 5] = encoder.fit_transform(input_data[:, 5])

# Split the data into train and test.
x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2)

#  Initialization for theta.
ini_parameters = np.array([[0, 0, 0, 0, 0, 0, 0]])

# Get the best parameters.
parameters = gradientDescent(ini_parameters, x_train)

# predict and get the accuracy.
meanSquaredError(parameters, x_test, y_test)



