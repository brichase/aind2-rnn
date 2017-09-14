import numpy as np

from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    n_pairs = len(series) - window_size
    for n_pair in range(n_pairs):
        X.append(series[n_pair: window_size + n_pair])
        y.append(series[window_size + n_pair])
        
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape = (window_size, 1)))
    model.add(Dense(1))
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    ascii_lowercase = 'abcdefghijklmnopqrstuvwxyz'
    valid_characters = ''.join(punctuation) + ascii_lowercase
    
    list_text = list(text)
    
    # Replace unwanted characters with a space
    for i, character in enumerate(text):
        if character not in valid_characters:
            list_text[i] = ' '
    
    text = ''.join(list_text)
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    # Create input output pairs
    number_of_pairs = (len(text) // step_size) - (window_size // step_size)
    for pair_index in range(number_of_pairs):
        text_index = pair_index * step_size
        inputs.append(text[text_index: text_index + window_size])
        outputs.append(text[text_index + window_size])
    
    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape = (window_size, num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))
    model.summary()
    return model
