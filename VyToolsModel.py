## VyToolsModel.py -- written by JHS 3/1/2024
##
## Utility functions for loading data from a
## dataset and training a generative model
## on it using a GAN. Much of this code was
## adapted from the following tutorial on 
## training a generative adversarial network 
## on a one-dimensional function: 
## https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-1-dimensional-function-from-scratch-in-keras/
## #############################################


from numpy import hstack
from numpy import zeros
from numpy import ones
from numpy.random import rand
from numpy.random import randn
from keras.models import Sequential
from keras.layers import Dense
import math
import os
import numpy as np
import struct
import random
from keras.models import load_model
import VyToolsConsts
import VyToolsShared
import wave


def generate_real_samples(data_path: str, training_path: str, i:int, n:int):
    '''
        Load dataset from binary files and
        generate n real samples with class labels.

        Args:

        Returns:
    '''
    curr_idx = i * n
    X11: list[np.float32] = []
    with open(training_path, 'rb') as f:
        f.seek(curr_idx)
        bytes_s = ' '
        while len(X11) < n and bytes_s:
            byte_s = f.read(4)
            if not byte_s:
                break
            float_data: np.float32 = struct.unpack('f', byte_s)
            X11.append(float_data[0])
    X2 = np.array(X11, np.float32)
    print('real data y axis vals min: ' + str(np.min(X2)))
    print('real data y axis vals mean: ' + str(np.mean(X2)))
    print('real data y axis vals max: ' + str(np.max(X2)))

    # Create label list for all files (should simply be indices from 1 to dataset_sample_size)
    _, _, dataset_files = next(os.walk(data_path))
    X22: list[np.int32] = []
    for _ in range(len(dataset_files)):
        X22.extend(range(VyToolsConsts.LATENT_SIZE))
    X1 = np.array(X22[curr_idx:curr_idx+n], np.int32)

    X1 = X1.reshape(n, 1)
    X2 = X2.reshape(n, 1)
    X = hstack((X1, X2))
    y = ones((n, 1))
    return X, y

def generate_latent_points(latent_dim: int, n: int):
    '''
        Generate x indices for points in the range [0,latent_dim]
        and reshape them into a batch of inputs for the network.

        Args: 
            latent_dim (int): dimension of the latent space.
            n (int): number of latent space samples to generate.

        Returns:
            np.array of int32 indices for x values in y = f(x) function.
    '''

    x_input = []
    for _ in range(latent_dim * n):
        x_input.append(random.randint(0,latent_dim))
    return np.array(x_input).astype(np.int32).reshape(n, latent_dim)

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n):
    x_input = generate_latent_points(latent_dim, n)
    X = generator.predict(x_input)

    print('fake data y axis vals min: ' + str(np.min(X)))
    print('fake data y axis vals mean: ' + str(np.mean(X)))
    print('fake data y axis vals max: ' + str(np.max(X)))

    y = zeros((n, 1))
    return X, y


def check_progress(rolling_average: list[float,int], new_value: np.array) -> list[float,int]:
    '''
        Check whether the rolling average (mean of generated values) is null for several training rounds.

        Args:
            rolling_average (list[float,int]): sum of N values that will be incremented.
            new_value (np.array): next value in the series N of rolling averages.
        
        Return:
            Updated list[float,int] for rolling average of generated data.
    '''
    rolling_average[1] = rolling_average[1] + 1
    rolling_average[0] = rolling_average[0] + (np.mean(new_value) - rolling_average[0]) / rolling_average[1]
    print('rolling average: ' + str(rolling_average[0]))
    if rolling_average[1] > 10 and np.all(np.abs(rolling_average) > 0.01):
        raise Exception('Error! Rolling average of values has been zero for multiple training rounds. Check input values.')
    return rolling_average

def generate_model(output_filename: str, data_path: str, training_path: str, latent_dim: int, n_epochs: int = 2548, n_batch: int = 2048) -> None:
    '''
        Set inputs == 2, since we are attempting to replicate some function y = f(x).
        We will be providing both x and y variables to the GAN. Note: this function
        can take hours or days depending upon your dataset size.

        Args:
            output_filename (str): file path where the generative .keras model will be written.
            data_path (str): folder containing all the dataset files.
            training_path (str): folder containing all the extracted features for use in training.
            latent_dim (int): the dimension of the latent space.
            n_epochs (int): number of epochs to train the GAN.

        Returns:
            None.
    '''
    num_inputs = 2
    half_batch = int(n_batch / 2)

    # Create discriminator.
    discriminator = Sequential()
    discriminator.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=num_inputs))
    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Create generator.
    generator = Sequential()
    generator.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
    generator.add(Dense(num_inputs, activation='linear'))
	
    # Create GAN.
    discriminator.trainable = False # make weights in the discriminator not trainable
    gan_model = Sequential()
    gan_model.add(generator)
    gan_model.add(discriminator)
    gan_model.compile(loss='binary_crossentropy', optimizer='adam')

    rolling_average = [0.0,0]

    # Do training. Generate both real and fake data and update the 
    # generator via the discriminator's error. Note that inverted 
    # labels are used for the fake samples so they're flagged as
    # different than the real samples.
    for i in range(n_epochs):
        x_real, y_real = generate_real_samples(data_path, training_path, i,half_batch)
        x_fake, y_fake = generate_fake_samples(generator, latent_dim, half_batch)	
        rolling_average = check_progress(rolling_average, x_fake[:, 1])
        discriminator.train_on_batch(x_real, y_real)
        discriminator.train_on_batch(x_fake, y_fake)
        gan_model.train_on_batch(generate_latent_points(latent_dim, n_batch), ones((n_batch, 1)))
	
        # Update the model by writing to file.
        generator.save(output_filename)



def generate_wav(output_filename: str, model_filepath: str, latent_dim: int, n_samples: int):
    """
        Load trained keras model and use it to generate new .wav content as 
        monaural 16-bit float data.

        Args:
            output_filename (str): name of the output generated file.
            model_filename (str): name of the model keras generative model file.
            output_filename (str): name of the output generated file.
            n_samples (int): number of samples to be generated by the generator.
 
        Returns:
            str with name of written generated file if successful, else null string.
    """

    generator = load_model(model_filepath)
    data, _ = generate_fake_samples(generator, latent_dim, n_samples)
    data = np.array(data[:, 1]).astype(np.float32)

    # normalize audio data to [-1,1].
    data = data + np.min(data)
    data = data / np.max(data)
    data = data * 1.0
    data = data - 1.0

    VyToolsShared.write_float_to_wav(output_filename,data)