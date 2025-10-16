

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt



############## Partie 1



(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()


NUM_CLASSES = 10
INPUT_SHAPE = x_train.shape[1:]


x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


y_train_cat = keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)
y_test_cat = keras.utils.to_categorical(y_test, num_classes=NUM_CLASSES)

print(f"Input data shape: {INPUT_SHAPE}")

# TODO: Print the shape of the labels after conversion
print(f"y_train shape before one-hot encoding: {y_train.shape}")
print(f"y_train shape after one-hot encoding: {y_train_cat.shape}")
print("-" * 30 + "\n")



####### PARTIE 2 EXERCICE 1



def build_basic_cnn(input_shape, num_classes):
    """Builds a simple sequential CNN model."""
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),

        # TODO: Add the Max Pooling layer
        layers.MaxPooling2D(pool_size=(2, 2)),  # Added Max Pooling

        # TODO: Add the second Conv2D layer
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),  # Added Conv2D layer

        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Flatten(),

        layers.Dense(512, activation='relu'),

        layers.Dense(num_classes, activation='softmax')
    ])
    return model



basic_cnn_model = build_basic_cnn(INPUT_SHAPE, NUM_CLASSES)
basic_cnn_model.summary()


basic_cnn_model.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])


history = basic_cnn_model.fit(
    x_train, y_train_cat,
    batch_size=64,
    epochs=10,
    validation_split=0.1
)

# TODO: Evaluate the model on x_test, y_test and display the accuracy.

score = basic_cnn_model.evaluate(x_test, y_test_cat, verbose=0)
print(f'Test loss: {score[0]:.4f}')
print(f'Test accuracy: {score[1]:.4f}')
print("-" * 30 + "\n")



###### PARTIE 2  EXERCICE 2


print("--- Part 2, Exercise 2: ResNet ---")


def residual_block(x, filters, kernel_size=(3, 3), stride=1):
    """A simplified residual block."""
    y = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', activation='relu')(x)
    y = layers.Conv2D(filters, kernel_size, padding='same')(y)


    if stride > 1 or x.shape[-1] != filters:
        x_skip = layers.Conv2D(filters, (1, 1), strides=stride)(x)
    else:
        x_skip = x

    # TODO: Complete the addition of the skip path with the main path
    z = layers.Add()([x_skip, y])
    z = layers.Activation('relu')(z)
    return z


def build_resnet_model(input_shape, num_classes):
    """Builds a small model using residual blocks."""
    # TODO: Build a small architecture using 3 consecutive residual blocks
    inputs = keras.Input(shape=input_shape)


    x = residual_block(inputs, 32)
    x = residual_block(x, 64, stride=2)
    x = residual_block(x, 64)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    return model


resnet_model = build_resnet_model(INPUT_SHAPE, NUM_CLASSES)
resnet_model.summary()
print("-" * 30 + "\n")






###### PARTIE 3 EXERCICE 4



def load_and_preprocess_image(path_to_img):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)


    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = 512 / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)

    img = img[tf.newaxis, :]
    return img


# TODO: Load and pre-process two images of your choice

content_image = load_and_preprocess_image('./image_test.png')
style_image = load_and_preprocess_image('./image_test.png')


vgg = keras.applications.VGG16(include_top=False, weights='imagenet')
vgg.trainable = False


content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']


def create_extractor(model, style_layers, content_layers):

    outputs = [model.get_layer(name).output for name in style_layers + content_layers]
    return keras.Model(inputs=model.input, outputs=outputs)


extractor = create_extractor(vgg, style_layers, content_layers)

