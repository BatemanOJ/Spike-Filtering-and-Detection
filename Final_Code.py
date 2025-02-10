import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
import os

# Import filterinf and spike dectection modules
from bandpass_filter import bandpass_filter 
from add_noise import add_noise
from detect_spikes import detect_spikes     
             

# Import modules for NN and making windows
from Neural_Network import NeuralNetwork
from Create_window import create_window                     
from Create_training_windows import create_training_window
from predicted_class_output import predicted_class_output

# Import module to store final index and class data to file
from Data_storage import data_storage


# Load Dataset D1
mat1 = spio.loadmat('D1.mat', squeeze_me=True)
mat2 = spio.loadmat('D2.mat', squeeze_me=True)
mat3 = spio.loadmat('D3.mat', squeeze_me=True)
mat4 = spio.loadmat('D4.mat', squeeze_me=True)
mat5 = spio.loadmat('D5.mat', squeeze_me=True)
mat6 = spio.loadmat('D6.mat', squeeze_me=True)
d_1 = mat1['d']
Index = mat1['Index']
Class = mat1['Class']
d_2 = mat2['d']
d_3 = mat3['d']
d_4 = mat4['d']
d_5 = mat5['d']
d_6 = mat6['d']

sorted = sorted(zip(Index, Class))  # Sort Index and class lists in order based on values in index
Index, Class = zip(*sorted) # Unzip back into individual lists

# Apply bandpass filter
lowcut = 3.5  # Hz
highcut = 1500  # Hz
fs = 25000  # Sampling rate in Hz
d_1_filtered = bandpass_filter(d_1, lowcut, highcut, fs)
d_2_filtered = bandpass_filter(d_2, lowcut, highcut, fs)
d_3_filtered = bandpass_filter(d_3, lowcut, highcut, fs)
d_4_filtered = bandpass_filter(d_4, lowcut, highcut, fs)
d_5_filtered = bandpass_filter(d_5, lowcut, highcut, fs)
d_6_filtered = bandpass_filter(d_6, lowcut, highcut, fs)

# Add noise to d_1 signal for training
d_1_noise = add_noise(d_1_filtered, noise_level = 0.5)
d_1_double_noise = add_noise(d_1_filtered, noise_level = 0.75)


# Detect spikes in each data set
detected_spikes_d1, properties = detect_spikes(d_1_filtered, height=0.75, prominence=0.5, width=5, distance=50)
detected_spikes_d2, properties = detect_spikes(d_2_filtered, height=0.75, prominence=0.5, width=6, distance=50)
detected_spikes_d3, properties = detect_spikes(d_3_filtered, height=0.85, prominence=0.5, width=7, distance=30)

# detected_spikes_d4, properties = detect_spikes(d_4_filtered, height=1.15, prominence=0.5, width=7.5, distance=35) # originally submitted line
detected_spikes_d4, properties = detect_spikes(d_4_filtered, height=1.25, prominence=0.5, width=7.5, distance=55) # new line

# detected_spikes_d5, properties = detect_spikes(d_5_filtered, height=1.75, prominence=3.5, width=7, distance=30) # originally submitted line
detected_spikes_d5, properties = detect_spikes(d_5_filtered, height=1.8, prominence=3.5, width=7, distance=30) # new line

# detected_spikes_d6, properties = detect_spikes(d_6_filtered, height=1.8, prominence=4.875, width=10, distance=30) # originally submitted line
detected_spikes_d6, properties = detect_spikes(d_6_filtered, height=2, prominence=4.75, width=10, distance=30) # new line


print(len(detected_spikes_d1), len(detected_spikes_d2), len(detected_spikes_d3), len(detected_spikes_d4), len(detected_spikes_d5), len(detected_spikes_d6))


windows = []
windows_test = []
labels = []
labels_test = []

# Detect windows in test data (d_1)
for i, idx in enumerate(Index):
    if idx >= 45 and idx + 45 < len(d_1):
        # Extract window around spike
        window = d_1_filtered[idx - 20: idx + 60]
        windows_test.append(window)
        labels_test.append(Class[i])

windows_test = np.array(windows_test)   # Make windows_test into an array
labels_test = np.array(labels_test)     # Make labels_test into an array
windows_test_1=[]
windows_test_2=[]
windows_test_3=[]
windows_test_4=[]
windows_test_5=[]

# Find the windows in each class and store them
for i in range(len(Class)):
    if Class[i] == 1:
        windows_test_1.append(windows_test[i])
    elif Class[i] == 2:
        windows_test_2.append(windows_test[i])
    elif Class[i] == 3:
        windows_test_3.append(windows_test[i])
    elif Class[i] == 4:
        windows_test_4.append(windows_test[i])
    elif Class[i] == 5:
        windows_test_5.append(windows_test[i])

# Find the average of the windows in each class
# Can be used to check if the predicted spikes for each class are similiar to the known spikes
average_array_1 = np.mean(windows_test_1, axis=0)
average_array_2 = np.mean(windows_test_2, axis=0)
average_array_3 = np.mean(windows_test_3, axis=0)
average_array_4 = np.mean(windows_test_4, axis=0)
average_array_5 = np.mean(windows_test_5, axis=0)
d_1_average_array = [average_array_1, average_array_2, average_array_3, average_array_4, average_array_5]


windows_output = []
labels_output = []

# Create training data from the known classes and indexs in data set d_1
# Training data is augmented by moving the data vertically, horitontally and by adding noise to the data 

windows_output, labels_output = create_training_window(Index, Class, d_1, 20, 60, -0.5, d_1_filtered, windows_output, labels_output) # Moved vertically by -0.5
windows_output, labels_output = create_training_window(Index, Class, d_1, 20, 60, 0.5, d_1_filtered, windows_output, labels_output) # Moved vertically by +0.5
windows_output, labels_output = create_training_window(Index, Class, d_1, 20, 60, 1, d_1_filtered, windows_output, labels_output) # Moved vertically by +1
windows_output, labels_output = create_training_window(Index, Class, d_1, 20, 60, -1, d_1_filtered, windows_output, labels_output) # Moved vertically by -1
windows_output, labels_output = create_training_window(Index, Class, d_1, 10, 70, 0, d_1_filtered, windows_output, labels_output) # Moved horizontally +10
windows_output, labels_output = create_training_window(Index, Class, d_1, 15, 65, 0, d_1_filtered, windows_output, labels_output) # Moved horizontally +5
windows_output, labels_output = create_training_window(Index, Class, d_1, 25, 55, 0, d_1_filtered, windows_output, labels_output) # Moved horizontally -5
windows_output, labels_output = create_training_window(Index, Class, d_1, 20, 60, 0, d_1_noise, windows_output, labels_output) # Windows with added noise
windows_output, labels_output = create_training_window(Index, Class, d_1, 20, 60, 0, d_1_double_noise, windows_output, labels_output) # Windows with added noise


# Convert to numpy arrays
labels = np.array(labels_output)
windows = np.array(windows_output)
labels = np.array(labels.flatten())

# Create the windows for the test data sets using the detected spike locations
windows_d_2 = create_window(detected_spikes_d2, d_2, d_2_filtered)
windows_d_3 = create_window(detected_spikes_d3, d_3, d_3_filtered)
windows_d_4 = create_window(detected_spikes_d4, d_4, d_4_filtered)
windows_d_5 = create_window(detected_spikes_d5, d_5, d_5_filtered)
windows_d_6 = create_window(detected_spikes_d6, d_6, d_6_filtered)

output_nodes = 5
instance = NeuralNetwork(80, 300, output_nodes, 0.05)

# One hot encode the labels for the classes
def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels-1] 

labels_one_hot = one_hot_encode(labels, output_nodes)

maxValueD1 = max(d_1)
maxValueD2 = max(d_2)
maxValueD3 = max(d_3)
maxValueD4 = max(d_4)
maxValueD5 = max(d_5)
maxValueD6 = max(d_6)


# Train the network
for _ in range(25):
    for window, label in zip(windows, labels):
        inputs = (np.asarray(window, dtype=np.float64) / maxValueD1  * 0.99) + 0.01

        targets = np.zeros(output_nodes) + 0.01

        targets[int(label-1)] = 0.99

        instance.train(inputs, targets)


predicted_class_output_d_1 = []
scorecard = []
counter = 0
#Query the network for the test data (d_1)
for windows, label in zip(windows_test, labels_test):
    correct_label = label
    inputs = (np.asarray(windows, dtype=np.float64) / maxValueD1  * 0.99) + 0.01
    outputs = instance.query(inputs)
    label = np.argmax(outputs) + 1
    
    # Count wether the NN correctly classified the spike
    if (label == correct_label):
        scorecard.append(1)
        counter += 1
    else:
        scorecard.append(0)
        counter += 1
        pass
    pass
    predicted_class_d_1 = np.argmax(outputs) + 1
    predicted_class_output_d_1.append(predicted_class_d_1)
counter = len(windows_test)
print(f"Counter accuracy d1 = {(np.sum(scorecard)/counter)*100}, {np.sum(scorecard)}/{counter}")

# Query the network for d_2
predicted_class_output_d_2 = []
for windows, label in zip(windows_d_2, windows_d_2):
    inputs = (np.asarray(windows, dtype=np.float64) / maxValueD1  * 0.99) + 0.01
    outputs = instance.query(inputs)
    predicted_class_d_2 = np.argmax(outputs) + 1
    predicted_class_output_d_2.append(predicted_class_d_2)


# Query the network for d_3
predicted_class_output_d_3 = []
for windows, label in zip(windows_d_3, windows_d_3):
    inputs = (np.asarray(windows, dtype=np.float64) / maxValueD1  * 0.99) + 0.01
    outputs = instance.query(inputs)
    predicted_class_d_3 = np.argmax(outputs) + 1
    predicted_class_output_d_3.append(predicted_class_d_3)


# Query the network for d_4
predicted_class_output_d_4 =[]
for windows, label in zip(windows_d_4, windows_d_4):
    inputs = (np.asarray(windows, dtype=np.float64) / maxValueD1  * 0.99) + 0.01
    outputs = instance.query(inputs)
    predicted_class_d_4 = np.argmax(outputs) + 1
    predicted_class_output_d_4.append(predicted_class_d_4)


# Query the network for d_5
predicted_class_output_d_5 =[]
for windows, label in zip(windows_d_5, windows_d_5):
    inputs = (np.asarray(windows, dtype=np.float64) / maxValueD1  * 0.99) + 0.01
    outputs = instance.query(inputs)
    predicted_class_d_5 = np.argmax(outputs) + 1
    predicted_class_output_d_5.append(predicted_class_d_5)


# Query the network for d_6
predicted_class_output_d_6 =[]
for windows, label in zip(windows_d_6, windows_d_6):
    inputs = (np.asarray(windows, dtype=np.float64) / maxValueD1  * 0.99) + 0.01
    outputs = instance.query(inputs)
    predicted_class_d_6 = np.argmax(outputs) + 1
    predicted_class_output_d_6.append(predicted_class_d_6)


data_storage(detected_spikes_d2, detected_spikes_d3, detected_spikes_d4, detected_spikes_d5, detected_spikes_d6, 
                 predicted_class_output_d_2, predicted_class_output_d_3, predicted_class_output_d_4, predicted_class_output_d_5, predicted_class_output_d_6)
