import numpy as np
import pandas as pd

# The fake and real profiles are previously entered into an Excel table
# Data scraping can be used so our previously chosen profile features/parameters are inputted into our database/table
# We import the database using the pandas library, and the data is stored in a dataframe named training set
trainingset = pd.read_csv('name_of_excel_file.csv')

# We tidy up our data set by dropping the missing values
trainingset = trainingset.dropna()

# We choose the data from the table and prepare the profile features that we will use to train our model
# Prepare the data set by converting values to numerical form and keeping them between a reasonable interval
trainingset = trainingset[['Account age',
                           'Gender',
                           'User age',
                           'Link in description',
                           'Number of messages sent out',
                           'Number of friend requests sent out',
                           'Entered location',
                           'Location by IP',
                           'Fake or Not']]
# Account age is at first in months and later converted to weeks
trainingset['Account age'] = trainingset['Account age'] * 4
# Entered location and location IP at account creation is compared and saved as boolean value
if trainingset['Entered location'] == trainingset['Location by IP']: trainingset['Location match'] = 1
else: trainingset['Location match'] = 0
# Age is divided by 100
trainingset['User age'] = trainingset['User age'] / 100
# Gender converted into a boolean value
if trainingset['Gender'] == "Female" or "female":
    trainingset['Gender'] = 1
else:
    trainingset['Gender'] = 0
# If there is a hyperlink in description, boolean value set to 1
# Regex used to find http
url_found = trainingset['Link in description'].match('\S', "http", trainingset['Link in description'].I)
if url_found:
    trainingset['Link in description'] = 1
else:
    trainingset['Link in description'] = 0
# Number of messages sent out based on age of the account in weeks
trainingset['Number of messages sent out'] = trainingset['Number of messages sent out'] / trainingset['Account age']
# Number of friend requests sent out based on age of the account in weeks
trainingset['Number of friend requests sent out'] = \
    trainingset['Number of friend requests sent out'] / trainingset['Account age']

# Below values could be modified depending on how the columns look like in our data set
# Stores input features, all columns except last one
inputset = trainingset.iloc[:, -2]
# Stores output, a boolean value which is the last column
outputset = trainingset.iloc[:, -1]

# We slice the the input and output variables
# Training data set is to train the model
# Test data set is used to determine accuracy
split = int(len(trainingset)*0.8)
input_train, input_test, output_train, output_test = inputset[:split], inputset[split:], outputset[:split], outputset[split:]

# We reduce possible bias that could happen later on by feature scaling again
# This process makes the mean of all the input features equal to zero and also converts their variance to 1
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
input_train = sc.fit_transform(input_train)
input_test = sc.transform(input_test)

from keras.models import Sequential
from keras.layers import Dense, Activation
model = Sequential()
# First layer of our sequential model
# There will be 128 neurons in the hidden layer
# Model will take inputs to the hidden layer as the number of columns in our input dataframe
# The starting values for the weights of the different neurons is initialized in a uniform distribution
# Activation method is set to Sigmoid function
model.add(Dense(units=128, input_dim=trainingset.shape[1],
          kernel_initializer='uniform',
          activation='sigmoid'))
# We build the output layer
model.add(Dense(units=1,
                kernel_initializer='uniform',
                activation='sigmoid'))

# Using Stochastic Gradient Descent as optimizer
# We are using the mean squared error as loss/cost function (output_true - output_prediction)^2
model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])

# Fitting our Neural Network to our training data set
# batch_size = Number of data points the model uses to compute error before backpropagating the errors
# epochs = Number of times the training of the model will be performed
model.fit(input_train, output_train, batch_size=25, epochs=100)

# Testing accuracy on the part of our dataset that we dedicated for verification
output_pred = model.predict(input_test)
# Test output is converted into percentage
output_pred = (output_pred * 100)
# We store the values of output_pred into a new column in our database
trainingset['Predicted Percentage'] = np.NaN
trainingset.iloc[(len(trainingset) - len(output_pred)):,-1:] = output_pred
percentage_dataset = trainingset.dropna()

# Visualizing accuracy and loss between training and test data set
# verbose = (0 = silent) (1 = progress bar) (2 = one line per epoch)
score = model.evaluate(input_test, output_test, verbose=2)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

