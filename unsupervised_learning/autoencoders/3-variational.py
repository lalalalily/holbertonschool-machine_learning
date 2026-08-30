#!/usr/bin/env python3
"""Variational Autoencoder module."""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates a variational autoencoder network."""
    # Encoder
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    z_mean = keras.layers.Dense(latent_dims, activation=None)(x)
    z_log_sigma = keras.layers.Dense(latent_dims, activation=None)(x)

    # Reparameterization trick
    def sampling(args):
        mu, log_sig = args
        epsilon = keras.backend.random_normal(
            shape=keras.backend.shape(mu)
        )
        return mu + keras.backend.exp(log_sig / 2.0) * epsilon

    z = keras.layers.Lambda(sampling)([z_mean, z_log_sigma])
    encoder = keras.Model(inputs, [z, z_mean, z_log_sigma], name='encoder')

    # Decoder
    latent_inputs = keras.Input(shape=(latent_dims,))
    x_dec = latent_inputs
    for nodes in reversed(hidden_layers):
        x_dec = keras.layers.Dense(nodes, activation='relu')(x_dec)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x_dec)
    decoder = keras.Model(latent_inputs, outputs, name='decoder')

    # Full VAE Model
    auto_outputs = decoder(z)
    auto = keras.Model(inputs, auto_outputs, name='auto')

    # Custom VAE Loss (Reconstruction + KL Divergence)
    def vae_loss(inputs, outputs):
        recon_loss = keras.losses.binary_crossentropy(inputs, outputs)
        recon_loss = recon_loss * input_dims
        kl_loss = -0.5 * keras.backend.sum(
            1 + z_log_sigma - keras.backend.square(z_mean) -
            keras.backend.exp(z_log_sigma),
            axis=-1
        )
        return keras.backend.mean(recon_loss + kl_loss)

    auto.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, auto
