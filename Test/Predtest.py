# Data Preprocessing

# from google.colab import drive
# drive.mount('/content/drive')

import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)

df = pd.read_csv("LSTMandTabular_with_Synthetic.csv")

import ast

df['Flipped_Mouse_Data'] = df['Flipped_Mouse_Data'].apply(lambda x: ast.literal_eval(x))
df = df[df['Flipped_Mouse_Data'].apply(lambda x: any('t' in point for point in x))]

import matplotlib.pyplot as plt
import seaborn as sns

# Set the figure size
plt.figure(figsize=(10, 3))
sns.countplot(x='Initial_Emotion', data=df)

from sklearn.preprocessing import  StandardScaler

numerical_features = ['Response_Time', 'Speed', 'Velocity', 'maximum_positive_deviation',
                      'maximum_negative_deviation', 'DTW','Direction_Change_Freq_10', 'Direction_Change_Freq_30',
                      'Direction_Change_Freq_45', 'Direction_Change_Freq_90','centroid_mp_dist',
                      'Img_Radius', 'Slope', 'Narrowness', 'point_dissimilarity', 'vector_dissimilarity'] # 'Age',
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

from sklearn.model_selection import train_test_split

# Separate features and target variable
X = df.drop('Initial_Emotion', axis=1)
y = df['Initial_Emotion']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)

import pandas as pd

# Assuming y_test is your target variable
# Convert y_test to a pandas Series if it's not already
y_test_series = pd.Series(y_test)

# Count the occurrences of each class (0 and 1)
class_counts = y_test_series.value_counts()

# Print the counts
print("Number of 0s in y_test:", class_counts[0])
print("Number of 1s in y_test:", class_counts[1])

X_train_lstm, X_test_lstm = pd.DataFrame(X_train['Flipped_Mouse_Data']), pd.DataFrame(X_test['Flipped_Mouse_Data'])
X_train_tabular, X_test_tabular = X_train.drop(['Flipped_Mouse_Data'], axis=1), X_test.drop(['Flipped_Mouse_Data'], axis=1)


X_train_lstm.head()


X_train_tabular.columns

X_train_tabular = X_train_tabular[['Response_Time', 'maximum_positive_deviation',
       'maximum_negative_deviation', 'DTW', 'Speed', 'Velocity',
       'Index_of_difficulty', 'Throughput', 'Direction_Change_Freq_10',
       'Direction_Change_Freq_30', 'Direction_Change_Freq_45',
       'Direction_Change_Freq_90', 'centroid_mp_dist', 'Img_Radius', 'Slope',
       'Narrowness', 'point_dissimilarity', 'vector_dissimilarity']]

X_test_tabular = X_test_tabular[['Response_Time', 'maximum_positive_deviation',
       'maximum_negative_deviation', 'DTW', 'Speed', 'Velocity',
       'Index_of_difficulty', 'Throughput', 'Direction_Change_Freq_10',
       'Direction_Change_Freq_30', 'Direction_Change_Freq_45',
       'Direction_Change_Freq_90', 'centroid_mp_dist', 'Img_Radius', 'Slope',
       'Narrowness', 'point_dissimilarity', 'vector_dissimilarity']]

X_train_tabular.head()

## LSTM Data prepro

# Extract features
def extract_features(data):
    features = []
    for entry in data:
        feature_entry = []
        for point in entry:
            if 't' in point:
                x = point.get('x', 0)
                y = point.get('y', 0)
                t = point['t']
                feature_entry.append([x, y, t])
                # feature_entry.append([x, y])
        features.append(feature_entry)
    return features

X_train_lstm = extract_features(X_train_lstm['Flipped_Mouse_Data'])
X_test_lstm = extract_features(X_test_lstm['Flipped_Mouse_Data'])

print(X_train_lstm[0])

from tensorflow.keras.preprocessing.sequence import pad_sequences

# Find the maximum length of sequences in X_train_lstm and X_test_lstm
max_len_train = max(len(seq) for seq in X_train_lstm)
max_len_test = max(len(seq) for seq in X_test_lstm)

# Determine the maximum padding length
max_padding_length = max(max_len_train, max_len_test)

# Pad sequences using the determined maximum padding length
X_train_lstm = pad_sequences(X_train_lstm, maxlen=max_padding_length, dtype='float32', padding='post')
X_test_lstm = pad_sequences(X_test_lstm, maxlen=max_padding_length, dtype='float32', padding='post')


print(X_train_lstm[0].shape)
X_test_lstm[0].shape

X_train.shape








# ########################################### start LTM + Tabular
# LTM + Tabular

#custom model 
#Register the Custom Layer 
#final model which can be saved and used for prediction

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, concatenate, Masking
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import GlobalAveragePooling1D, Reshape, LeakyReLU, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import RNN
import numpy as np

@tf.keras.utils.register_keras_serializable()
class CustomLSTM(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(CustomLSTM, self).__init__(**kwargs)
        self.units = units
        self.state_size = (units, units)

    def build(self, input_shape):
        self.forget_gate_w = self.add_weight(shape=(input_shape[-1], self.units),
                                             initializer='glorot_uniform',
                                             trainable=True)
        self.forget_gate_b = self.add_weight(shape=(self.units,),
                                             initializer='zeros',
                                             trainable=True)

        self.input_gate_w = self.add_weight(shape=(input_shape[-1], self.units),
                                            initializer='glorot_uniform',
                                            trainable=True)
        self.input_gate_b = self.add_weight(shape=(self.units,),
                                            initializer='zeros',
                                            trainable=True)

        self.output_gate_w = self.add_weight(shape=(input_shape[-1], self.units),
                                             initializer='glorot_uniform',
                                             trainable=True)
        self.output_gate_b = self.add_weight(shape=(self.units,),
                                             initializer='zeros',
                                             trainable=True)

        self.cell_state_update_w = self.add_weight(shape=(input_shape[-1], self.units),
                                                   initializer='glorot_uniform',
                                                   trainable=True)
        self.cell_state_update_b = self.add_weight(shape=(self.units,),
                                                   initializer='zeros',
                                                   trainable=True)

        self.reg = self.add_weight(shape=(self.units,),
                                   initializer='zeros',
                                   trainable=True)

    def call(self, inputs, states):
        h_prev, c_prev = states

        forget_gate = tf.sigmoid(tf.matmul(inputs, self.forget_gate_w) + self.forget_gate_b)
        input_gate = tf.sigmoid(tf.matmul(inputs, self.input_gate_w) + self.input_gate_b)
        output_gate = tf.sigmoid(tf.matmul(inputs, self.output_gate_w) + self.output_gate_b)
        cell_gate = tf.tanh(tf.matmul(inputs, self.cell_state_update_w) + self.cell_state_update_b)

        Ct = (forget_gate * c_prev) + (input_gate * cell_gate) + self.reg
        ht = output_gate * tf.tanh(Ct)

        return ht, [ht, Ct]

    def get_config(self):
        config = super(CustomLSTM, self).get_config()
        config.update({'units': self.units})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Assuming X_train_lstm, X_train_tabular, y_train are defined

# Define LSTM model
lstm_ip = Input(shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]))
x = Masking()(lstm_ip)
x = RNN(CustomLSTM(8), return_sequences=False)(x)
lstm_output = x

# Define tabular model
tabular_input = Input(shape=(X_train_tabular.shape[1],))
tabular_output = Dense(8, activation='relu')(tabular_input)

# Concatenate LSTM and tabular outputs
concatenated = concatenate([lstm_output, tabular_output])

# Additional Dense layers
x = Dense(8, activation='relu')(concatenated)

# Output layer
output = Dense(1, activation='sigmoid')(x)

# Create model
model = Model(inputs=[lstm_ip, tabular_input], outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history3 = model.fit([X_train_lstm, X_train_tabular], y_train, epochs=100, batch_size=100, validation_split=0.2)






# Serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Serialize weights to HDF5
model.save_weights("model.weights.h5")
print("Saved model weights to disk")


from tensorflow.keras.models import model_from_json

# Load JSON and create model
with open("model.json", "r") as json_file:
    loaded_model_json = json_file.read()

# Custom objects need to be passed here
loaded_model = model_from_json(loaded_model_json, custom_objects={'CustomLSTM': CustomLSTM})

# Load weights into the new model
loaded_model.load_weights("model.weights.h5")
print("Loaded model from disk")


from sklearn.metrics import confusion_matrix, accuracy_score
# 1. Calculate accuracy
y_pred = model.predict([X_test_lstm, X_test_tabular])
y_pred_binary = (y_pred > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred_binary)

print("Model Accuracy:", accuracy)

# 2. Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_binary)
print("Confusion Matrix:")
print(conf_matrix)



# ########################################### end LTM + Tabular


################################ Prediction  #################################



import pandas as pd
import numpy as np

# Prediction

df = pd.read_csv("LSTMandTabular_with_Synthetic.csv")

df = df[['Flipped_Mouse_Data','Response_Time']]
df.head()

df = df.head(1)

# EXECUTE THIS CELL IF Flipped_Mouse_Data IS IN STRING FORMAT

import ast

df['Flipped_Mouse_Data'] = df['Flipped_Mouse_Data'].apply(lambda x: ast.literal_eval(x))
df = df[df['Flipped_Mouse_Data'].apply(lambda x: any('t' in point for point in x))]

## Feature Engg

### Perpendicular deviation

import math

# Function to find the line given two points
def lineFromPoints(P, Q):
    a = Q[1] - P[1]
    b = P[0] - Q[0]
    c = - a*(P[0]) - b*(P[1])
    return a,b,c

# Function to find distance
def shortest_distance(x1, y1, a, b, c):
    d = abs((a * x1 + b * y1 + c)) / (math.sqrt(a * a + b * b))
    return d

def deviation_from_ideal_path(a,b,c,data):
  global dists
  dists = []

  sum_dist = 0

  for point in data:
    d = shortest_distance(point['x'],point['y'],a,b,c)
    if((a*point['x']+b*point['y']+c) < 0):
      dists.append(-d)
    else:
      dists.append(d)

  #   sum_dist += d ** 2

  # deviation = math.sqrt(sum_dist/len(data))
  # return deviation
  return dists

def calculate_deviation(cords):
  cord_array = np.array([(point['x'],point['y']) for point in cords])
  start_point = cord_array[0]
  end_point = cord_array[-1]

  a,b,c = lineFromPoints(start_point,end_point)

  deviation_dists = deviation_from_ideal_path(a,b,c,cords)
  return deviation_dists

df['deviation_dists'] = df['Flipped_Mouse_Data'].apply(calculate_deviation)
df.head()

df['maximum_positive_deviation'] = df['deviation_dists'].apply(lambda x: max(x) if any(dev > 0 for dev in x) else 0)
df['maximum_negative_deviation'] = df['deviation_dists'].apply(lambda x: min(x) if any(dev < 0 for dev in x) else 0)

df.head()

### Dynamic Time Warping

import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

def Find_dtw_dist(cords):
    # Convert the string representation of the list to an actual list
    # mouse_data_list = ast.literal_eval(cords)
    mouse_data_list = cords

    # Extracting the first and last points from the mouse data
    first_point = mouse_data_list[0]
    last_point = mouse_data_list[-1]

    # Extracting x, y, and t values from the first and last points
    # x1, y1, t1 = first_point['x'], first_point['y'], first_point['t']
    # x2, y2, t2 = last_point['x'], last_point['y'], last_point['t']

    x1, y1 = first_point['x'], first_point['y']
    x2, y2 = last_point['x'], last_point['y']

    # Calculating the slope of the ideal path
    slope = (y2 - y1) / (x2 - x1)

    # Generating the ideal path
    ideal_path = []
    for point in mouse_data_list:
        # For each y value in the mouse data, calculate the corresponding x value
        x = x1 + (point['y'] - y1) / slope
        # Keep the t value unchanged
        # t = point['t']
        # Append the x, y, t values to the ideal path
        ideal_path.append({'x': x, 'y': point['y']})  # , 't': t  ADD THIS FOR 't' dataset

    #extracting the x and y coordinates for dtw dist
    first_row_coords = [(point['x'], point['y']) for point in  mouse_data_list]
    ideal_path_coords = [(point['x'], point['y']) for point in ideal_path]

    # Extract x and y coordinates from the input data
    x1, y1 = zip(*first_row_coords)
    x2, y2 = zip(*ideal_path_coords)

    # Calculate the DTW distance using only the spatial coordinates
    distance, _ = fastdtw(np.column_stack((x1, y1)), np.column_stack((x2, y2)), dist=euclidean)
    return distance/len(cords)

df['DTW'] = df['Flipped_Mouse_Data'].apply(Find_dtw_dist)


df.head()

### Speed and velocity

import pandas as pd
from ast import literal_eval
import math

# # Assuming your data is in a CSV file named 'trajectory_data.csv'
# df = pd.read_csv('/content/drive/MyDrive/FYP/mouse_tracking_new.csv')

# # Convert the 'cord' column from string to list of dictionaries
# df['cord'] = df['cord'].apply(literal_eval)

# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return math.sqrt((point2['x'] - point1['x'])**2 + (point2['y'] - point1['y'])**2)

# Calculate Euclidean distance between consecutive coordinates and sum to get total distance
df['distance'] = df['Flipped_Mouse_Data'].apply(lambda x: sum(euclidean_distance(x[i], x[i+1]) for i in range(len(x)-1)))

# Calculate speed (distance / time) in pixels per millisecond
df['Speed'] = df['distance']*1000 / df['Response_Time']

# Calculate velocity (Euclidean distance between initial and final coordinates / total time)
df['Velocity'] = df['Flipped_Mouse_Data'].apply(lambda x: euclidean_distance(x[0], x[-1]))*1000 / df['Response_Time']

df.head()

### Fitts law

import math

def FittsLaw(row):
  width = 250
  displacement = row['Velocity']*row['Response_Time']

  index_of_difficulty = math.log2(displacement/(width+1))

  return index_of_difficulty

df['Index_of_difficulty'] = df.apply(FittsLaw, axis=1)
df.head(5)

def Throughput(row):
  return row['Index_of_difficulty']/row['Response_Time']

df['Throughput'] = df.apply(Throughput, axis=1)
df.head(5)

### Direction Change Frequeny

import math

def calculate_angle(x1, y1, x2, y2):
    # Calculate the angle between two vectors using arctan
    dx = x2 - x1
    dy = y2 - y1
    return math.atan2(dy, dx)

def calculate_direction_change_frequency(mouse_data, angle_threshold):
    direction_changes = 0
    total_segments = len(mouse_data) - 1
    previous_angle = None
    # l = []

    for i in range(1, len(mouse_data)):
        x1, y1 = mouse_data[i - 1]['x'], mouse_data[i - 1]['y']
        x2, y2 = mouse_data[i]['x'], mouse_data[i]['y']

        # Calculate angle between consecutive points
        angle = calculate_angle(x1, y1, x2, y2)
        # l.append(angle)

        # Check if previous angle exists and angle change exceeds threshold
        if previous_angle is not None and abs(angle - previous_angle) > angle_threshold:
            direction_changes += 1

        previous_angle = angle
    # print(l)
    # print(direction_changes)
    # print(total_segments)
    direction_change_frequency = direction_changes / total_segments
    return direction_change_frequency

# Define threshold values
threshold_values = [10, 30, 45, 90]

# Define function to apply direction change frequency calculation for each threshold value
def apply_direction_change_frequency(df, threshold_values):
    for threshold in threshold_values:
        column_name = f'Direction_Change_Freq_{threshold}'
        df[column_name] = df['Flipped_Mouse_Data'].apply(lambda x: calculate_direction_change_frequency(x, math.radians(threshold)))
    return df

# Apply function to DataFrame
df = apply_direction_change_frequency(df, threshold_values)

df.head()

### Centroid

import pandas as pd
import math

# Define the functions to calculate centroid, midpoint, and distance
def calculate_centroid(data):
    # Extract x and y coordinates
    x_coords = [point['x'] for point in data]
    y_coords = [point['y'] for point in data]

    # Calculate centroid
    centroid_x = sum(x_coords) / len(data)
    centroid_y = sum(y_coords) / len(data)

    return centroid_x, centroid_y

def calculate_midpoint(data):
    # Extract start and end points
    start_point = data[0]
    end_point = data[-1]

    # Calculate midpoint
    midpoint_x = (start_point['x'] + end_point['x']) / 2
    midpoint_y = (start_point['y'] + end_point['y']) / 2

    return midpoint_x, midpoint_y

def calculate_distance(centroid_x, centroid_y, midpoint_x, midpoint_y):
    # Calculate Euclidean distance
    distance = math.sqrt((centroid_x - midpoint_x)**2 + (centroid_y - midpoint_y)**2)

    # Determine if centroid is on the left side of the line
    if centroid_x < midpoint_x:
        distance *= -1

    return distance

# Apply functions to each row in the 'Flipped_Mouse_Data' column
df['centroid_mp_dist'] = df['Flipped_Mouse_Data'].apply(lambda data: calculate_distance(*calculate_centroid(data), *calculate_midpoint(data)))


df.head()

### Imaginay Radius of Circle covering all points

import pandas as pd
import math

# Define the function to calculate the radius
def calculate_radius(data):
    # Calculate centroid
    centroid_x = sum(point['x'] for point in data) / len(data)
    centroid_y = sum(point['y'] for point in data) / len(data)

    # Calculate distances from centroid to each point
    distances = [math.sqrt((point['x'] - centroid_x)**2 + (point['y'] - centroid_y)**2) for point in data]

    # Radius is the maximum distance
    radius = max(distances)

    return radius

# Apply the function to each row in the 'Flipped_Mouse_Data' column
df['Img_Radius'] = df['Flipped_Mouse_Data'].apply(calculate_radius)

df.head()

### Slope and Narrowness

import pandas as pd
from sklearn.decomposition import PCA

# Function to compute slope and narrowness
def compute_slope_and_narrowness(data):
    # Prepare the data: Extract x and y coordinates
    x_coords = [point['x'] for point in data]
    y_coords = [point['y'] for point in data]
    xy_coords = np.array(list(zip(x_coords, y_coords)))

    # Perform PCA
    pca = PCA(n_components=2)
    pca.fit(xy_coords)

    # Calculate slope (direction of movement)
    first_principal_component = pca.components_[0]
    slope = first_principal_component[1] / first_principal_component[0]  # Rise over run

    # Calculate narrowness (explained variance ratio)
    narrowness = pca.explained_variance_ratio_[1]

    return pd.Series({'Slope': slope, 'Narrowness': narrowness})

# Apply the function to the DataFrame column and store output in new columns
df[['Slope', 'Narrowness']] = df['Flipped_Mouse_Data'].apply(compute_slope_and_narrowness)


df.head()

### Dissimilarity

import ast
import math
import pandas as pd

def Find_dissimilarity(first_row_mousedata):
    mouse_data_list = first_row_mousedata

    first_point = mouse_data_list[0]
    last_point = mouse_data_list[-1]

    x1, y1 = first_point['x'], first_point['y']  #, first_point['t']
    x2, y2 = last_point['x'], last_point['y']  #, last_point['t']

    slope = (y2 - y1) / (x2 - x1)

    ideal_path = []
    for point in mouse_data_list:
        x = x1 + (point['y'] - y1) / slope
        # t = point['t']
        ideal_path.append({'x': x, 'y': point['y']})   # , 't': t

    total_dissimilarity_point = 0
    total_squared_distance = 0
    for point1, point2 in zip(mouse_data_list, ideal_path):
        x1, y1 = point1['x'], point1['y']
        x2, y2 = point2['x'], point2['y']
        dissimilarity_point = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        total_dissimilarity_point += dissimilarity_point
        total_squared_distance += (x1 - x2) ** 2 + (y1 - y2) ** 2

    point_dissimilarity = total_dissimilarity_point / len(mouse_data_list)
    vector_dissimilarity = math.sqrt(total_squared_distance) / len(mouse_data_list)

    return point_dissimilarity, vector_dissimilarity

# Assuming df is your DataFrame and 'Flipped_Mouse_Data' is the column containing mouse data
df['point_dissimilarity'], df['vector_dissimilarity'] = zip(*df['Flipped_Mouse_Data'].apply(Find_dissimilarity))

df.head()

df = df.drop(['deviation_dists', 'distance'], axis=1)

df.head()

# df.to_csv("/content/drive/MyDrive/FYP/Model Comparison/Only Synthetic PosNeg.csv")

## LSTM and Tabular separate prepro

df.head()

df_for_normalization = pd.read_csv("LSTMandTabular_with_Synthetic.csv")

df_for_normalization.head()

df_for_normalization = pd.concat([df,df_for_normalization])

df_for_normalization.head()

from sklearn.preprocessing import  StandardScaler

numerical_features = ['Response_Time', 'Speed', 'Velocity', 'maximum_positive_deviation',
                      'maximum_negative_deviation', 'DTW','Direction_Change_Freq_10', 'Direction_Change_Freq_30',
                      'Direction_Change_Freq_45', 'Direction_Change_Freq_90','centroid_mp_dist',
                      'Img_Radius', 'Slope', 'Narrowness', 'point_dissimilarity', 'vector_dissimilarity'] # 'Age',
scaler = StandardScaler()
df_for_normalization[numerical_features] = scaler.fit_transform(df_for_normalization[numerical_features])

df_for_normalization.head()

df = df_for_normalization.head(1)

df.head()

test_lstm = pd.DataFrame(df['Flipped_Mouse_Data'])
test_tabular = df.drop(['Flipped_Mouse_Data'], axis=1)


test_tabular = test_tabular[['Response_Time', 'maximum_positive_deviation',
       'maximum_negative_deviation', 'DTW', 'Speed', 'Velocity',
       'Index_of_difficulty', 'Throughput', 'Direction_Change_Freq_10',
       'Direction_Change_Freq_30', 'Direction_Change_Freq_45',
       'Direction_Change_Freq_90', 'centroid_mp_dist', 'Img_Radius', 'Slope',
       'Narrowness', 'point_dissimilarity', 'vector_dissimilarity']]

# Extract features
def extract_features(data):
    features = []
    for entry in data:
        feature_entry = []
        for point in entry:
            if 't' in point:
                x = point.get('x', 0)
                y = point.get('y', 0)
                t = point['t']
                feature_entry.append([x, y, t])
                # feature_entry.append([x, y])
        features.append(feature_entry)
    return features

test_lstm = extract_features(test_lstm['Flipped_Mouse_Data'])

print(test_lstm[0])

max_len_test = 199 

from tensorflow.keras.preprocessing.sequence import pad_sequences

# Find the maximum length of sequences in X_train_lstm and test_lstm
max_len_test = 199

# Determine the maximum padding length
max_padding_length = max(max_len_train, max_len_test)

# Pad sequences using the determined maximum padding length
test_lstm = pad_sequences(test_lstm, maxlen=max_padding_length, dtype='float32', padding='post')


print(test_lstm[0].shape)

test_tabular

# print(test_lstm[0])

## Predict Input data

# Predict the output
prediction = model.predict([test_lstm, test_tabular])

# Convert the prediction to binary (0 or 1) using sigmoid activation
binary_prediction = 1 if prediction >= 0.5 else 0

print("Final result:", binary_prediction)




# Make predictions
predictions = loaded_model.predict([test_lstm, test_tabular])

# Convert the predictions to binary (0 or 1) using a threshold of 0.5
binary_predictions = (predictions >= 0.5).astype(int)

# If you want to print a single prediction, you can select one, for example, the first one
print("Final result for the first sample:", binary_predictions[0][0])

# If you want to print all predictions
print("Binary predictions for all samples:", binary_predictions.flatten())
