import tensorflow as tf
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = x_train / 255.0
x_test = x_test / 255.0

print(x_train)

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
model.fit(x_train.reshape((-1, 28, 28, 1)), y_train, epochs=10, validation_data=(x_test.reshape((-1, 28, 28, 1)), y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test.reshape((-1, 28, 28, 1)), y_test, verbose=2)
print('Test accuracy:', test_acc)


def predict_digit():
    # Open a file dialog to select an image file
    file_path = filedialog.askopenfilename()

    # Load the image file using OpenCV
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    # Resize the image to 28x28
    img = cv2.resize(img, (28, 28))

    # Scale the pixel values to be between 0 and 1
    img = img / 255.0

    # Reshape the image to have a single channel
    img = np.reshape(img, (1, 28, 28, 1))

    # Feed the image into the model and get predictions
    predictions = model.predict(img)

    # Get the predicted digit and update the label
    predicted_digit = np.argmax(predictions[0])
    label.config(text="Predicted digit: " + str(predicted_digit))


# Create a Tkinter window
window = tk.Tk()

# Add a button to select an image file and test the model
button = tk.Button(window, text="Select image file", command=predict_digit)
button.pack(pady=10)

# Add a label to display the predicted digit
label = tk.Label(window, text="")
label.pack(pady=10)

# Start the Tkinter event loop
window.mainloop()