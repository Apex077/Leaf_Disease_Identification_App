import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

# Image dimensions and batch size
img_width, img_height = 150, 150
batch_size = 32

# Create an ImageDataGenerator for the training set
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

# Create an ImageDataGenerator for the validation set
validation_datagen = ImageDataGenerator(rescale=1./255)

# Load images from the training directory and preprocess them
train_generator = train_datagen.flow_from_directory('Leaf_Diseases_Dataset/Train',
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

# Load images from the validation directory and preprocess them
validation_generator = validation_datagen.flow_from_directory('Leaf_Diseases_Dataset/Validation',
                                                              target_size=(img_width, img_height),
                                                              batch_size=batch_size,
                                                              class_mode='categorical')


#Initial Code for Training the Model
""" model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
              metrics=['accuracy'])

history = model.fit(train_generator,
          steps_per_epoch=train_generator.samples // batch_size,
          validation_data=validation_generator,
          validation_steps=validation_generator.samples // batch_size,
          epochs=50)

model.save("SproutIQ_Disease_Detection_Model.keras") """

loaded_model = tf.keras.models.load_model("SproutIQ_Disease_Detection_Model.keras")

# Test the model
val_loss, val_accuracy = loaded_model.evaluate(validation_generator)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

# Initialize the ImageDataGenerator with the same preprocessing function
test_datagen = ImageDataGenerator(rescale=1./255)  # adjust this to match your preprocessing

# Create the generator
test_generator = test_datagen.flow_from_directory('Leaf_Diseases_Dataset/Test',
                                                  target_size=(img_width, img_height),
                                                  batch_size=batch_size,
                                                  class_mode='categorical')

# Making Predictions
""" predictions = loaded_model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

print(predicted_classes) """

# Get the class labels from the generator
class_labels = test_generator.class_indices
class_labels = {v: k for k, v in class_labels.items()}# key-value flip

# Get the first batch of images and labels from the test generator
test_images, test_labels = next(test_generator)

# Make predictions on the first batch of the test data
predictions = loaded_model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)

# Plot the first few images, their true labels, and their predicted labels
plt.figure(figsize=(15, 15))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(test_images[i])
    plt.colorbar()
    plt.title(f"True: {class_labels[np.argmax(test_labels[i])]}, Predicted: {class_labels[predicted_classes[i]]}",
              color = 'lightgreen', fontsize=12, fontweight='bold')
    plt.axis("off")

plt.subplots_adjust(wspace = 0.2, hspace = 0.5)
plt.show()


# Building the App
# from kivy.app import App
# from kivy.uix.label import Label
# from kivy.uix.camera import Camera
# from kivy.uix.button import Button
# from kivy.uix.boxlayout import BoxLayout
# import cv2


# class mainApp(App):
#     def build(self):
#         layout = BoxLayout(orientation='vertical')

#         # Create a camera object
#         self.camera = Camera(play=True)
#         layout.add_widget(self.camera)

#         # Create a button to take a picture
#         self.button = Button(text='Take Picture')
#         self.button.bind(on_press=self.take_picture)
#         layout.add_widget(self.button)

#         # Create a label to show the prediction
#         self.label = Label(text='Prediction: None')
#         layout.add_widget(self.label)

#         # Load the model
#         self.model = tf.keras.models.load_model('SproutIQ_Disease_Detection_Model.keras')

#         return layout
    
#     def take_picture(self, *args):
#         self.camera = Camera(play=True, resolution=(1920, 1080))
#         self.camera.export_to_png('image.png')
#         image = cv2.imread('image.png')
#         image = cv2.resize(image, (150, 150))
#         image = image / 255.0
#         image = np.expand_dims(image, axis=0)

#         prediction = self.model.predict(image)
#         predicted_class = np.argmax(prediction)

#         self.label.text = f'Prediction: {predicted_class}'
    

# mainapp = mainApp()
# mainapp.run()