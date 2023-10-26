import os
import random
import shutil
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# Callaback
fc_log_dir = "logs/fit/" + "fc_" +datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
fc_tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=fc_log_dir, histogram_freq=1)
cnn_log_dir = "logs/fit/" + "fc_" +datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
cnn_tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=cnn_log_dir, histogram_freq=1)

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Model checkpoint callback
checkpoint = ModelCheckpoint('models/best_fc_model.h5', save_best_only=True, monitor='val_accuracy')


#=============
# A. 
base_dir = './'
dataset_dir = 'dataset'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

dog_train_dir = os.path.join(train_dir,"dog")
cat_train_dir = os.path.join(train_dir,"cat")

dog_validation_dir = os.path.join(validation_dir,"dog")
cat_validation_dir = os.path.join(validation_dir,"cat")

# Create directories for training and validation data
for path in [train_dir,
             validation_dir,
             dog_train_dir,
             cat_train_dir,
             dog_validation_dir,
             cat_validation_dir]:
    if not os.path.exists(path):
        os.makedirs(path)

# Collect all image files in the 'dataset' directory
image_files = os.listdir(dataset_dir)
print(len(image_files))
# Randomly select 8,000 dog and 8,000 cat images for training
random.shuffle(image_files)

dog_train_count = len(os.listdir(dog_train_dir))
cat_train_count = len(os.listdir(cat_train_dir))
print(f"dog:{dog_train_count} \n cat:{cat_train_count}")

# Move the first 8,000 dog and 8,000 cat images to the training directory
for filename in image_files:
    if os.path.isfile(os.path.join(dataset_dir,filename)):
        if filename.startswith('dog') and dog_train_count < 8000:
            shutil.move(os.path.join(dataset_dir, filename), os.path.join(dog_train_dir, filename))
            dog_train_count += 1
        elif filename.startswith('cat') and cat_train_count < 8000:
            shutil.move(os.path.join(dataset_dir, filename), os.path.join(cat_train_dir, filename))
            cat_train_count += 1

# Move the remaining images to the validation directory
for filename in image_files:
    print(filename)
    print(os.path.isfile(os.path.join(dataset_dir,filename))) 
    if os.path.isfile(os.path.join(dataset_dir,filename)):
        if filename.startswith('dog'):
            shutil.move(os.path.join(dataset_dir, filename), os.path.join(dog_validation_dir, filename))
        elif filename.startswith('cat'):
            shutil.move(os.path.join(dataset_dir, filename), os.path.join(cat_validation_dir, filename))

# c) Preprocess the images using data augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Create image generators for training and validation
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(100, 100),
    batch_size=32,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(100, 100),
    batch_size=32,
    class_mode='binary'
)
print(tf.config.list_physical_devices('GPU'))

# Inspect the generator
"""
sample_batch, sample_labels = validation_generator.next()
for i in range(4):  # Display the first 4 images in the batch
    plt.subplot(2, 2, i + 1)
    plt.imshow(sample_batch[i])
    plt.title(f"Label: {sample_labels[i]}")
plt.show()
exit()
"""
# d) Design and build TWO deep-learning models

# Model 1: Fully-connected network model
# model_fc = keras.Sequential([
#     keras.layers.Flatten(input_shape=(150, 150, 3)),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dense(1, activation='sigmoid')
# ])

model_fc = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(100, 100, 3)),
    tf.keras.layers.Dense(256, activation='relu'),  # Increase neurons and change activatio
    tf.keras.layers.Dense(1, activation='sigmoid')
])


# e) Set up optimization, learning rate, batch size, and epoch numbers for both models

# Model 1
# Define a learning rate schedule (you can customize this schedule)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,  # Starting learning rate
    decay_steps=10000,           # Learning rate decay steps
    decay_rate=0.9               # Learning rate decay rate
)

# Create an optimizer with the learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model_fc.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_fc = model_fc.fit(train_generator, validation_data=validation_generator, epochs=40,callbacks=[fc_tensorboard_callback,early_stopping, checkpoint])
# Save Model fc
model_fc.save(f'models/fc_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.h5', save_format='h5') 

# Model 1 accuracy
val_loss_fc, val_acc_fc = model_fc.evaluate(validation_generator)
print(f"Model 1 Validation Accuracy: {val_acc_fc:.2f}")

