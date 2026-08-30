#!/usr/bin/env python3
"""Variational Autoencoder"""
import tensorflow.keras as keras
import tensorflow.keras.backend as K


class KLLossLayer(keras.layers.Layer):
    """Layer that computes and adds the KL divergence loss"""
    def call(self, inputs):
        """Adds the KL divergence loss and passes z through unchanged"""
        z, mu, log_sig = inputs
        kl_loss = 1 + log_sig - K.square(mu) - K.exp(log_sig)
        kl_loss = -0.5 * K.sum(kl_loss, axis=-1)
        self.add_loss(K.mean(kl_loss))
        return z


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates a variational autoencoder
    input_dims is an integer containing the dimensions of the model
    input
    hidden_layers is a list containing the number of nodes for each
    hidden layer in the encoder, respectively
    the hidden layers should be reversed for the decoder
    latent_dims is an integer containing the dimensions of the
    latent space representation
    Returns: encoder, decoder, auto
    encoder is the encoder model, which should output the latent
    representation, the mean, and the log variance, respectively
    decoder is the decoder model
    auto is the full autoencoder model
    """
    def sampling(args):
        """Samples from the latent distribution using the
        reparameterization trick
        """
        mu, log_sig = args
        batch = K.shape(mu)[0]
        dims = K.shape(mu)[1]
        epsilon = K.random_normal(shape=(batch, dims))
        return mu + K.exp(log_sig / 2) * epsilon

    encoder_inputs = keras.Input(shape=(input_dims,))
    x = encoder_inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    mu = keras.layers.Dense(latent_dims, activation=None)(x)
    log_sig = keras.layers.Dense(latent_dims, activation=None)(x)
    z = keras.layers.Lambda(sampling)([mu, log_sig])
    encoder = keras.Model(encoder_inputs, [z, mu, log_sig])

    decoder_inputs = keras.Input(shape=(latent_dims,))
    x = decoder_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    decoder_outputs = keras.layers.Dense(
        input_dims, activation='sigmoid')(x)
    decoder = keras.Model(decoder_inputs, decoder_outputs)

    z, mu, log_sig = encoder(encoder_inputs)
    z = KLLossLayer()([z, mu, log_sig])
    auto_outputs = decoder(z)
    auto = keras.Model(encoder_inputs, auto_outputs)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
