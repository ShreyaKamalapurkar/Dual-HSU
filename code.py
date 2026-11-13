import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Define hyperparameters
bands = 198  # Number of spectral bands (example)
input_channels = 1  # Single channel for 1D CNN
output_size = 5  # Number of endmembers

# Build the Sequential model
model = models.Sequential([
    layers.Conv1D(64, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=(bands, input_channels)),
    layers.MaxPooling1D(pool_size=2, strides=2, padding='valid'),
    
    layers.Conv1D(128, kernel_size=3, strides=1, padding='same', activation='relu'),
    layers.MaxPooling1D(pool_size=2, strides=2, padding='valid'),
    
    layers.Conv1D(256, kernel_size=3, strides=1, padding='same', activation='relu'),
    layers.MaxPooling1D(pool_size=2, strides=2, padding='valid'),
    
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(output_size)  # Output layer, no activation for regression
])

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU

# Define the 2D CNN model
def build_2d_cnn(input_shape, num_endmembers):
    inputs = Input(shape=input_shape[1:])  # Exclude batch dimension: (100, 100, 198)

    # Spectral feature reduction with 1x1 convolutions
    x = Conv2D(64, (1, 1), padding='same', activation='relu')(inputs)  # Shape: (100, 100, 64)
    x = Conv2D(32, (1, 1), padding='same', activation='relu')(x)  # Shape: (100, 100, 32)

    # Spatial feature extraction with 3x3 convolutions
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)  # Shape: (100, 100, 64)
    x = BatchNormalization()(x)

    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)  # Shape: (100, 100, 32)
    x = BatchNormalization()(x)

    # Final output layer with num_endmembers filters
    outputs = Conv2D(num_endmembers, (1, 1), activation='sigmoid', padding='same')(x)  # Shape: (100, 100, 4)

    model = Model(inputs, outputs)
    return model

# Build the model
input_shape = input_data.shape  # (1, 100, 100, 198)
num_endmembers = target_data.shape[-1]  # 4
model = build_2d_cnn(input_shape, num_endmembers)

# Display model summary
model.summary()
