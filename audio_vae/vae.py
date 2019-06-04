import os
import keras
import pickle

import numpy as np

from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten, Lambda
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from scipy import signal
from scipy.io import wavfile

PATH = os.path.dirname(os.path.abspath(__file__))

class VAE_Embedding():
    def __init__(self):
        input_shape = (129, 48, 1)
        intermediate_dim = 512
        latent_dim = 40
        batch_size = 16
        kernel_size = 6
        stride_size = 3
        filters = 16
        
        def sampling(args):
            """Reparameterization trick by sampling fr an isotropic unit Gaussian.
            # Arguments
                args (tensor): mean and log of variance of Q(z|X)
            # Returns
                z (tensor): sampled latent vector
            """

            z_mean, z_log_var = args
            batch = K.shape(z_mean)[0]
            dim = K.int_shape(z_mean)[1]
            # by default, random_normal has mean=0 and std=1.0
            epsilon = K.random_normal(shape=(batch, dim))
            return z_mean + K.exp(0.5 * z_log_var) * epsilon

        # VAE model = encoder + decoder
        # build encoder model
        inputs = Input(shape=input_shape, name='encoder_input')
        x = inputs
        for i in range(2):
            filters *= 2
            x = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       activation='tanh',
                       strides=3,
                       padding='valid')(x)

        # shape info needed to build decoder model
        shape = K.int_shape(x)

        # generate latent vector Q(z|X)
        x = Flatten()(x)
        x = Dense(intermediate_dim, activation='tanh')(x)
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

        # instantiate encoder model
        self.encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

        # build decoder model
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        x = Dense(shape[1] * shape[2] * shape[3], activation='tanh')(latent_inputs)
        x = Reshape((shape[1], shape[2], shape[3]))(x)

        for i in range(2):
            x = Conv2DTranspose(filters=filters,
                                kernel_size=kernel_size,
                                activation='tanh',
                                strides=3,
                                padding='valid')(x)
            filters //= 2

        outputs = Conv2DTranspose(filters=1,
                                  kernel_size=kernel_size,
                                  activation='sigmoid',
                                  padding='same',
                                  name='decoder_output')(x)

        # instantiate decoder model
        decoder = Model(latent_inputs, outputs, name='decoder')

        # instantiate VAE model
        outputs = decoder(self.encoder(inputs)[2])
        vae = Model(inputs, outputs, name='vae')

        reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))

        reconstruction_loss *= input_shape[0] * input_shape[1]
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -5e-4
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        vae.compile(optimizer='rmsprop')

        vae.load_weights(os.path.join(PATH, 'model/vae.h5'))

        self.x_mean = pickle.load(open(os.path.join(PATH, 'data/x_mean.pkl'), 'rb'))
        self.x_std = pickle.load(open(os.path.join(PATH, 'data/x_std.pkl'), 'rb'))

    def compute_features(self, filename, frame_duration=0.01, stride=0.1, window_size=0.3):
        sample_rate, samples = wavfile.read(filename)
        frame_size = int(round(frame_duration * float(sample_rate)))
        overlap_size = frame_size - int(round((frame_duration - 0.005) * float(sample_rate)))
        n_fft_points = 2
        while n_fft_points < frame_size:
            n_fft_points *= 2
        frequencies, times, spectrogram = signal.spectrogram(
            samples, fs=sample_rate, window='hamming', nperseg=frame_size, noverlap=overlap_size, nfft=n_fft_points,
            scaling='spectrum', mode='psd'
        )
        n_frames_window = int(frame_size * window_size)
        new_features = []
        padded_spectrogram = np.hstack((np.zeros((frequencies.shape[0], int(frame_size*stride))),
                                        spectrogram,
                                        np.zeros((frequencies.shape[0], int(frame_size*stride)))))
        for suck_i, time in enumerate(times*stride):
            i = int(suck_i/stride)
            chunk = padded_spectrogram[:, i:int(frame_size * window_size) + i]
            if chunk.shape == (frequencies.shape[0], int(frame_size*window_size)):
                new_features += [chunk]
        x_file = (np.stack(new_features) - self.x_mean) / self.x_std
        return self.encoder.predict(x_file.reshape(x_file.shape + (1,)))[2]

    def compute_features_dir(self, dirname):
        for audio in os.listdir(dirname):
            yield self.compute_features(os.path.join(dirname, audio))