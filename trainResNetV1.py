import os
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
import json
import matplotlib.pyplot as plt
import time
import pyautogui

# Define the path to the dataset
path = 'C:/Source_Code/Sample_Dataset/Datasets_300_size_potential_fusariumV1_100/datasets/train' #

# Define paths for each class
healthy_path = os.path.join(path, 'healthy')
fusarium_path = os.path.join(path, 'fusarium')

# Ensure the class directories exist
if not os.path.exists(healthy_path):
    raise FileNotFoundError(f"Healthy path '{healthy_path}' does not exist. Please create the 'healthy' directory.")
if not os.path.exists(fusarium_path):
    raise FileNotFoundError(f"Fusarium path '{fusarium_path}' does not exist. Please create the 'fusarium' directory.")

# Load image file paths and labels
def load_data():
    images = []
    labels = []
    for label, folder in enumerate(['fusarium', 'healthy']):
        folder_path = os.path.join(path, folder)
        for filename in os.listdir(folder_path):
            if filename.endswith('.png'):  # Assuming all images are in PNG format
                images.append(os.path.join(folder_path, filename))
                labels.append(label)  # Use 0 for fusarium and 1 for healthy
    return images, labels


# Load and split data
all_images, all_labels = load_data()
train_images, test_images, train_labels, test_labels = train_test_split(
    all_images, all_labels, test_size=0.2, random_state=42, shuffle=True
)  # total data 675, training 540, and testing 135: total healthy image 452, and fusarium 233, training healthy: 374, training fusarium 166, testing healthy 78 fusarium 57

# Convert labels to strings for DataFrame
train_labels_str = ['healthy' if label == 1 else 'fusarium' for label in train_labels]
test_labels_str = ['healthy' if label == 1 else 'fusarium' for label in test_labels]

# Create DataFrames for train and test data
train_df = pd.DataFrame({'filename': train_images, 'class': train_labels_str})
test_df = pd.DataFrame({'filename': test_images, 'class': test_labels_str})

# Create ImageDataGenerator instances for training and testing
datagen_train = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen_test = ImageDataGenerator(rescale=1. / 255)

# Create data generators for training and testing
train_generator = datagen_train.flow_from_dataframe(
    dataframe=train_df,
    x_col='filename',
    y_col='class',
    target_size=(100, 100),
    batch_size=32,
    # save_to_dir='C:/Source_Code/Fusarium_wilt_V1/Result_Major_revision_Aug/comparisons/augmented_images',  # Directory to save images
    save_prefix='aug',
    save_format='png',
    class_mode='binary'
)
test_generator = datagen_test.flow_from_dataframe(
    dataframe=test_df,
    x_col='filename',
    y_col='class',
    target_size=(100, 100),
    batch_size=32,
    class_mode='binary'
)

"""
The training dataset consisted of 540 images. During training, random on-the-fly augmentations were applied to these images using transformations such as rotation, 
translation, shearing, zooming, and horizontal flipping. This ensured that the model encountered a unique augmented version of the dataset in each epoch, 
effectively increasing the diversity of the training data without increasing the dataset size.

for 2 epochs:
For data augmentation, we applied several transformations, including rotation, shifting, shearing, zooming, and horizontal flipping, to artificially expand the 
training dataset. During each epoch, the original 540 images were augmented, and 32 augmented images from the first batch were saved. In total, 572 images were saved 
per epoch (32 from the first batch and 540 from the augmented dataset), resulting in 1,144 images saved over 2 epochs.
NUmber of image will be: 572 x number of epochs : original number of training image 540
"""

def build_resnet22():
    inputs = tf.keras.Input(shape=(100, 100, 3))
    x = tf.keras.layers.Conv2D(64, (7, 7), padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    def residual_block(x, filters, kernel_size=3, strides=1):
        res = x
        x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', use_bias=False, strides=strides)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        if strides != 1:
            res = tf.keras.layers.Conv2D(filters, (1, 1), padding='same', strides=strides, use_bias=False)(res)
            res = tf.keras.layers.BatchNormalization()(res)
        x = tf.keras.layers.Add()([x, res])
        x = tf.keras.layers.ReLU()(x)
        return x

    # First stage: 3 residual blocks with 64 filters
    x = residual_block(x, 64, kernel_size=3, strides=1)
    x = residual_block(x, 64, kernel_size=3, strides=1)
    x = residual_block(x, 64, kernel_size=3, strides=1)

    # Second stage: 3 residual blocks with 128 filters
    x = residual_block(x, 128, kernel_size=3, strides=2)
    x = residual_block(x, 128, kernel_size=3, strides=1)
    x = residual_block(x, 128, kernel_size=3, strides=1)

    # Third stage: 3 residual blocks with 256 filters
    x = residual_block(x, 256, kernel_size=3, strides=2)
    x = residual_block(x, 256, kernel_size=3, strides=1)
    x = residual_block(x, 256, kernel_size=3, strides=1)

    # Fourth stage: 3 residual blocks with 512 filters
    x = residual_block(x, 512, kernel_size=3, strides=2)
    x = residual_block(x, 512, kernel_size=3, strides=1)
    x = residual_block(x, 512, kernel_size=3, strides=1)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)  # Output layer for binary classification

    model = tf.keras.Model(inputs=inputs, outputs=x)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return model


# Custom callback to compute and store additional metrics
class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data
        self.history = {
            'loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'train_precision': [],
            'val_precision': [],
            'train_recall': [],
            'val_recall': [],
            'train_f1': [],
            'val_f1': []
        }

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # Store loss and val_loss
        self.history['loss'].append(logs.get('loss'))
        self.history['val_loss'].append(logs.get('val_loss'))

        # Get predictions and true labels
        train_data = self.model.predict(self.validation_data[0], verbose=0)
        train_labels = self.validation_data[1]

        val_data = self.validation_data[2]
        val_labels = self.validation_data[3]

        train_preds = (train_data > 0.5).astype('int32').flatten()
        val_preds = (self.model.predict(val_data, verbose=0) > 0.5).astype('int32').flatten()

        # Compute metrics for training data
        train_accuracy = accuracy_score(train_labels, train_preds)
        train_precision = precision_score(train_labels, train_preds)
        train_recall = recall_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds)

        # Store training metrics
        self.history['train_accuracy'].append(train_accuracy)
        self.history['train_precision'].append(train_precision)
        self.history['train_recall'].append(train_recall)
        self.history['train_f1'].append(train_f1)

        # Compute metrics for validation data
        val_accuracy = accuracy_score(val_labels, val_preds)
        val_precision = precision_score(val_labels, val_preds)
        val_recall = recall_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)

        # Store validation metrics
        self.history['val_accuracy'].append(val_accuracy)
        self.history['val_precision'].append(val_precision)
        self.history['val_recall'].append(val_recall)
        self.history['val_f1'].append(val_f1)

        print(f"Epoch {epoch + 1}: train_accuracy: {train_accuracy:.4f}, val_accuracy: {val_accuracy:.4f}, "
              f"train_precision: {train_precision:.4f}, val_precision: {val_precision:.4f}, "
              f"train_recall: {train_recall:.4f}, val_recall: {val_recall:.4f}, "
              f"train_f1: {train_f1:.4f}, val_f1: {val_f1:.4f}")




epochs=10
resnet_model = build_resnet22()
model_name = 'resnet22'
print("Current Model",model_name)


# Prepare data for the callback
train_data, train_labels = next(train_generator)
val_data, val_labels = next(test_generator)
print("Training Shape:",train_data.shape)
print("Validation Shape:",val_data.shape)

# Define the ModelCheckpoint callback to save the best weights based on validation accuracy
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
     'best_'+model_name+'_weights.h5',  # Path to save the best weights
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)


# Initialize the custom callback
metrics_callback = MetricsCallback(validation_data=(train_data, train_labels, val_data, val_labels))

# Measure inference time
start_time = time.time()
# Train the ResNet18 model
history = resnet_model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=epochs,  # Adjust the number of epochs as needed
    callbacks=[checkpoint_callback, metrics_callback]  # Add the custom metrics callback here
)
end_time = time.time()
total_time = end_time - start_time
formatted_time_training = "{:.2e}".format(total_time)
print("Total time_training:", formatted_time_training)

# Load the best weights from the saved model
resnet_model.load_weights('best_'+model_name+'_weights.h5')

# Save the entire model to a file
resnet_model.save(model_name + '_model31925.h5') # test

# Load the model from the file
loaded_model = tf.keras.models.load_model(model_name+'_model31925.h5') # using this model, that integrated resnet18_model71624

# Evaluate the ResNet18 model
test_loss, test_accuracy, test_precision, test_recall = loaded_model.evaluate(test_generator)




################################## Option 2
# Reset the generator
test_generator.reset()
# Initialize storage for all images and labels
all_images = []
all_labels = []

# Iterate through all batches
for i in range(len(test_generator)):
    images, labels = next(test_generator)
    all_images.append(images)
    all_labels.append(labels)

# Concatenate all images and labels into arrays
all_images = np.concatenate(all_images, axis=0)
all_labels = np.concatenate(all_labels, axis=0)
test_labels=all_labels
print(f"Total test images: {all_images.shape[0]}")

# Measure inference time
start_time = time.time()

test_predictions_scores = loaded_model.predict(all_images)
test_predictions = (test_predictions_scores > 0.5).astype('int32').flatten()
#########################################
end_time = time.time()
total_time = end_time - start_time
formatted_time_testing = "{:.2e}".format(total_time)
print("Total time_training:", formatted_time_testing)

ap = average_precision_score(test_labels, test_predictions_scores)  # ≈ mAP for binary class
accuracy = accuracy_score(test_labels, test_predictions)
precision = precision_score(test_labels, test_predictions)
recall = recall_score(test_labels, test_predictions)
f1 = f1_score(test_labels, test_predictions)
print("Estimated mAP / Average Precision:", ap)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Precision: {precision:.2f}")
print(f"Test Recall: {recall:.2f}")
print(f"Test F1 Score: {f1:.2f}")
# print("Predictions:",test_predictions)
# print("Actual:", test_labels)

