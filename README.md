# Step 1: Import the necessary libraries
pip install tensorflow matplotlib numpy
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

# Step 2: Load the MNIST dataset
# This dataset contains 60,000 training images and 10,000 testing images
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Step 3: Preprocess the data
# Normalize the pixel values (from 0-255) to a range of 0-1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Step 4: Build the neural network model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Flatten the 28x28 image into a 1D array
    layers.Dense(128, activation='relu'),  # Fully connected layer with 128 neurons
    layers.Dense(10, activation='softmax') # Output layer with 10 neurons (one for each digit 0-9)
])

# Step 5: Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 6: Train the model
history = model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

# Step 7: Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\\nTest accuracy: {test_acc:.4f}')

# Step 8: Make predictions
predictions = model.predict(test_images)

# Step 9: Plot some predictions
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel(f"{predicted_label} ({true_label})", color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# Plot the first 5 test images, their predicted labels, and the true labels
num_rows = 1
num_cols = 5
num_images = num_rows * num_cols
plt.figure(figsize=(2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, num_cols, i+1)
    plot_image(i, predictions, test_labels, test_images)
plt.show()
