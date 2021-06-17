import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

# Reading dataset
def read_data(csv_file):
    input_data = pd.read_csv(csv_file, usecols = ['Open', 'High', 'Low', 'Close'])
    out_put_data = pd.read_csv(csv_file, usecols = ['Close'])
    return input_data, out_put_data

def reading_processing_input(csv_file):
    input_data = np.array(pd.read_csv(csv_file, usecols = ['Open', 'High', 'Low', 'Close']))
    input_data = np.reshape(input_data, (input_data.shape[0], 1, input_data.shape[1]))
    return input_data

# Building 
def data_processing(x,y):
    x = x.round(decimals = 2)
    y = y.round(decimals = 2)
    x = np.array(x[:-1])
    y = np.array(y[1:])
    return x, y

# Standarizing dataset to 
def data_standarization(input):
    scaler = StandardScaler()
    scaler.fit(input)
    input = scaler.transform(input)
    return input, scaler

# Sequencing data to feed into the RNN
def sequencing_data(input, output):
    x = np.reshape(input, (input.shape[0], 1, input.shape[1]))
    y = np.reshape(output, (output.shape[0], 1, output.shape[1]))
    return x, y

# Splitting dataset linearly to preserve temporary correlation
def train_test_linear_split(x,y, train_ratio):
    train_boundary = int(train_ratio*len(x))
    train_x = x[0:train_boundary]
    test_x  = x[train_boundary:]
    train_y = y[0:train_boundary]
    test_y  = y[train_boundary:]
    return train_x, test_x, train_y, test_y

# Helper method to extract a portion of data
def data_subset(x, y, data_percentage):
    data_boundary = int(len(x)*data_percentage)
    x = x[data_boundary:]
    y = y[data_boundary:]
    return x, y 

# training and compiling model by providing train data and test data along with the timescale,
# the name of the index that is being approximate, the number of epochs our model and the learning rate
def train_model(train_x, test_x, train_y, test_y, csv_file, number_epochs, loss_function, learning_rate, opt, call_back):
    activation_f = 'relu'
    model = Sequential()
    model.add(LSTM(128, input_shape = train_x.shape[1:], activation = activation_f, return_sequences = True))
    model.add(LSTM(64, activation = activation_f, return_sequences = True))
    model.add(LSTM(32, activation = activation_f, return_sequences = True))
    model.add(LSTM(8,  activation = activation_f, return_sequences = True))
    model.add(Dense(4, activation = activation_f))
    model.add(Dense(1, activation = activation_f))
    model.compile(loss= loss_function, optimizer=opt)
    model.fit(train_x, train_y, validation_data = (test_x, test_y), epochs = number_epochs, callbacks = [call_back])
    return model

#Saving model to a .h5 file 
def save_model(model, number_epochs, lr, loss_function, trial_number):
    model.save(csv_file + '_epochs_' + str(number_epochs) + '_lr_' + str(lr) + '_loss_' + str(loss_function) + '_number_' + str(trial_number) +'.h5') 

# Plotting protype function
def plot(x_axis, y_axis, title, x_label, y_label):
    for axis in y_axis:
        plt.plot(x_axis, axis['y_axis'], color = axis['color'], linestyle = '-', marker = '.' , label = axis['label']) #Lines
    plt.title(title) #Title of Plot
    plt.xlabel(x_label) #Label for x axis
    plt.ylabel(y_label) #Label for y axis
    plt.grid(True) #Deploying grid
    plt.legend() #Deploying legends
    plt.show()

# Plotting the index approximation vs actual index over the testing set
def plotting_approximation(test_y, predicted_y):
    test_y = np.reshape(test_y, (len(test_y),))
    predicted_y = np.reshape(predicted_y, (len(predicted_y),))
    x = range(len(test_y))
    plot(x ,[{'y_axis': test_y, 'color': 'g', 'label': 'real'}, 
        {'y_axis': predicted_y, 'color': 'r', 'label': 'predicted'}],
        'Index', 'Days', 'Prediction')

# Plotting the percentage error in the model
def plotting_percentage_error(test_y, predicted_y):
    test_y = np.reshape(test_y, (len(test_y),))
    predicted_y = np.reshape(predicted_y, (len(predicted_y),))
    percentage_error = 100*np.abs((test_y - predicted_y)/test_y)
    x = range(len(test_y))
    plot(x, [{'y_axis' :percentage_error, 'color': 'b', 'label': 'percentage error'}], 'Percentage Error', 'Days', '%')
    
