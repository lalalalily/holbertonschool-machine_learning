#!/usr/bin/env python3
"""
Defines a function that trains a model using mini-batch gradient descent,
handling validation data, early stopping, step-wise inverse time decay,
and saving the best iteration of the model.
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None,
                verbose=True, shuffle=False):
    """
    Trains a Keras model with options for early stopping, LR decay, and
    saving the best performing iteration.

    Args:
        network: the Keras model to train
        data: numpy.ndarray of shape (m, nx) containing the input data
        labels: one-hot numpy.ndarray of shape (m, classes) with the labels
        batch_size: size of the batch used for mini-batch gradient descent
        epochs: number of passes through the data
        validation_data: data to validate the model with, as a tuple of
                         (X_valid, Y_valid), or None
        early_stopping: boolean indicating whether early stopping should be
        patience: the patience used for early stopping
        learning_rate_decay: boolean indicating whether LR decay should be used
        alpha: the initial learning rate
        decay_rate: the decay rate for inverse time decay
        save_best: boolean indicating whether to save the model after each
                   epoch if it is the best
        filepath: file path where the model should be saved
        verbose: boolean that determines if output should be printed
        shuffle: boolean that determines whether to shuffle batches each epoch

    Returns:
        The History object generated after training the model.
    """
    callbacks = []

    # Callbacks require validation_data to be passed
    if validation_data is not None:

        # 1. Early Stopping Callback
        if early_stopping:
            early_stop_callback = K.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience
            )
            callbacks.append(early_stop_callback)

        # 2. Learning Rate Decay Callback (Inverse Time Decay)
        if learning_rate_decay:
            def scheduler(epoch):
                """Calculates learning rate based on inverse time decay."""
                return alpha / (1 + decay_rate * epoch)

            lr_scheduler = K.callbacks.LearningRateScheduler(
                schedule=scheduler,
                verbose=1
            )
            callbacks.append(lr_scheduler)

        # 3. Model Checkpoint Callback
        if save_best and filepath is not None:
            checkpoint = K.callbacks.ModelCheckpoint(
                filepath=filepath,
                monitor='val_loss',
                save_best_only=True,
                mode='min'
            )
            callbacks.append(checkpoint)

    # Train the network using mini-batch gradient descent
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=verbose,
        shuffle=shuffle
    )

    return history
