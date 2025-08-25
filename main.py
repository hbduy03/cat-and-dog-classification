import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

train_format = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    brightness_range=[0.8,1.2],
    zoom_range=0.2,
    horizontal_flip=True,
                              )
val_format = ImageDataGenerator(rescale=1./255)

train_data = train_format.flow_from_directory(
    './dogs_vs_cats/train',
    target_size=(150,150),
    class_mode='binary',
    batch_size=32
)
val_data = val_format.flow_from_directory(
    './dogs_vs_cats/test',
    target_size=(150,150),
    class_mode='binary',
    batch_size=32
)
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3),activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(127 ,(3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
    ])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy','precision','recall'])

result = model.fit(train_data,epochs=3,validation_data=val_data)
model.save('catanddog.h5')

fig, axes = plt.subplots(2,2, figsize=(12,4))

axes[0,0].plot(result.history['loss'], label='Training Loss')
axes[0,0].plot(result.history['val_loss'], label='Validation Loss')
axes[0,0].set_title('Loss')
axes[0,0].legend()

axes[0,1].plot(result.history['accuracy'], label='Training Accuracy')
axes[0,1].plot(result.history['val_accuracy'], label='Validation Acuracy')
axes[0,1].set_title('Accuracy')
axes[0,1].legend()

axes[1,0].plot(result.history['precision'], label='Training Precision')
axes[1,0].plot(result.history['val_precision'], label='Validation Precision')
axes[1,0].set_title('Precision')
axes[1,0].legend()

axes[1,1].plot(result.history['recall'], label='Training Recall')
axes[1,1].plot(result.history['val_recall'], label='Validation Recall')
axes[1,1].set_title('Recall')
axes[1,1].legend()
plt.show()



