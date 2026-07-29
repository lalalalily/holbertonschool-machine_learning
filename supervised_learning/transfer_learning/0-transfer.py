#!/usr/bin/env python3
"""
Transfer learning on CIFAR-10 using a pre-trained Keras application (ResNet50).

Design notes / log (useful for the accompanying blog post):
- Application used: ResNet50, pre-trained on ImageNet (via keras.applications).
- CIFAR-10 images are 32x32; ResNet50 expects a much larger input, so a Lambda
  layer resizes 32x32 -> 224x224 before the images reach the frozen base model.
- Per Hint 3: instead of running the full ResNet50 forward pass on every
  training example every epoch (slow), the frozen base model's output is
  computed ONCE for the whole training/test set ("bottleneck features"), and
  a small, fast classifier head is trained on those fixed-size feature
  vectors instead. This cuts training time from hours to minutes.
- Bottleneck features are extracted in batches (not all at once) to avoid
  holding the fully resized 224x224 dataset in memory, which would be tens
  of gigabytes for the full CIFAR-10 training set.
- After the head is trained, its weights are copied into a second, full
  end-to-end model (raw 32x32x3 input -> prediction) so the saved model can
  be evaluated directly on raw CIFAR-10 images, as required.
- EarlyStopping on val_accuracy is used to avoid overfitting the head once
  validation accuracy plateaus.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras as K


def preprocess_data(X, Y):
    """
    Pre-processes CIFAR-10 data for use with a ResNet50-based model.

    X is a numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR-10
    data, where m is the number of data points.
    Y is a numpy.ndarray of shape (m,) containing the CIFAR-10 labels for X.

    Returns: X_p, Y_p
        X_p is a numpy.ndarray containing the preprocessed X (scaled and
            mean-centered the same way ResNet50 expects its inputs).
        Y_p is a numpy.ndarray containing the preprocessed Y (one-hot
            encoded).
    """
    X_p = K.applications.resnet50.preprocess_input(X.astype('float32'))
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


def _resize(images, size=(224, 224)):
    """Resizes a batch of images to the target size (bilinear interpolation)."""
    return tf.image.resize(images, size)


def _extract_features(base_model, X, batch_size=200, target_size=(224, 224)):
    """
    Runs X through the frozen base model in batches to compute bottleneck
    features, resizing each batch on the fly so the full resized dataset
    never has to be held in memory at once.
    """
    features = []
    for i in range(0, X.shape[0], batch_size):
        batch = X[i:i + batch_size]
        batch_resized = _resize(batch, target_size).numpy()
        batch_features = base_model.predict(batch_resized, verbose=0)
        features.append(batch_features)
    return np.concatenate(features, axis=0)


def main():
    """Trains a transfer-learning CIFAR-10 classifier and saves it as cifar10.h5."""
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()

    X_train_p, Y_train_p = preprocess_data(X_train, Y_train)
    X_test_p, Y_test_p = preprocess_data(X_test, Y_test)

    # 1) Frozen pre-trained base: ResNet50, ImageNet weights, no top
    #    classifier, global average pooling so the output is already a flat
    #    feature vector rather than a spatial feature map.
    base_model = K.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    base_model.trainable = False

    # 2) Compute bottleneck features ONCE for train and test sets, since a
    #    frozen model always produces the same output for the same input.
    print('Extracting training features...')
    train_features = _extract_features(base_model, X_train_p)
    print('Extracting test features...')
    test_features = _extract_features(base_model, X_test_p)

    # 3) Train a small, fast classifier head on the pre-computed features.
    feature_input = K.layers.Input(shape=train_features.shape[1:])
    x = K.layers.Dense(256, activation='relu')(feature_input)
    x = K.layers.Dropout(0.3)(x)
    head_output = K.layers.Dense(10, activation='softmax')(x)
    head_model = K.Model(feature_input, head_output)

    head_model.compile(
        optimizer=K.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        K.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=5, restore_best_weights=True
        )
    ]

    head_model.fit(
        train_features, Y_train_p,
        validation_data=(test_features, Y_test_p),
        batch_size=128,
        epochs=30,
        callbacks=callbacks,
        verbose=1
    )

    # 4) Assemble the full, end-to-end model (raw 32x32x3 input -> prediction)
    #    so the saved model can be evaluated directly on raw CIFAR-10 images.
    #    The trained head weights are copied in rather than retrained.
    full_input = K.layers.Input(shape=(32, 32, 3))
    resized = K.layers.Lambda(lambda img: _resize(img, (224, 224)))(full_input)
    base_output = base_model(resized, training=False)
    y = K.layers.Dense(256, activation='relu')(base_output)
    y = K.layers.Dropout(0.3)(y)
    full_output = K.layers.Dense(10, activation='softmax')(y)
    full_model = K.Model(full_input, full_output)

    # full_model.layers: [Input, Lambda, base_model, Dense(256), Dropout, Dense(10)]
    full_model.layers[3].set_weights(head_model.layers[1].get_weights())   # Dense(256)
    full_model.layers[-1].set_weights(head_model.layers[-1].get_weights())  # Dense(10)

    full_model.compile(
        optimizer=K.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    full_model.save('cifar10.h5')
    print('Model saved as cifar10.h5')


if __name__ == '__main__':
    main()