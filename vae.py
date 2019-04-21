import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Lambda, Input, Dense
from keras.losses import binary_crossentropy, mse
from keras import optimizers
from keras import backend as K
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
import random
import pickle
from plotnine import *

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
# https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

#File input
f = open('store50000_15x15.pckl', 'rb')
inpt = pickle.load(f)
f.close()

#Reorganising data for model input
for mat in range(len(inpt)):
    inpt[mat] = np.array(inpt[mat]).flatten()
inpt = np.array(inpt)

inpt = inpt.reshape(inpt.shape[0], inpt.shape[1])

X_train, X_test, y_train, y_test = train_test_split(inpt, inpt, test_size=0.2, random_state=123)

#Number of neuron in input
neuron_n = 225

#------ model construction -------
# network parameters
original_dim = neuron_n
input_shape = (neuron_n, )
intermediate_dim = 100
batch_size = 25
latent_dim = 2
epochs = 50

#encoder
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
plot_model(encoder, to_file='vae_mlp_encoder_2.png', show_shapes=True)

#decoder
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='vae_mlp_decoder_2.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

print(vae.summary())

reconstruction_loss = binary_crossentropy(inputs, outputs)
reconstruction_loss *= original_dim
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')
vae.summary()
plot_model(vae,
           to_file='vae_mlp_2.png',
           show_shapes=True)

vae.fit(X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, None))

#----------------------------------

#Get final output
layer_name = 'decoder' #Selects last layer

final_layer_model = vae
final_output = final_layer_model.predict(X_test)

sample_N = 30

print("input mat \n", X_test[sample_N])
print("output mat \n", final_output[sample_N])

df = pd.DataFrame(columns=["indiv", "mut_pos", "in_val", "out_val"])

c = 0
mutN = 15
indivN = 15
for indiv in range(0,indivN):
    for mut in range(0,mutN):
        df.loc[c] = [indiv, mut, X_test[sample_N][indiv*mutN+mut], final_output[sample_N][indiv*mutN+mut]]
        c+=1

p = ggplot(df, aes("factor(mut_pos)", "factor(indiv)", fill="in_val")) + geom_tile(aes(width=.95, height=.95)) + scale_fill_gradient(low="#000000", high="#ffffff", limits=[0,1]) + xlab("Mutation positions") + ylab("Individuals")

p.save(filename = 'plots/test/15x15_heat_in_vae.png', height=5, width=10, units = 'in', res=1000)


p2 = ggplot(df, aes("factor(mut_pos)", "factor(indiv)", fill="out_val")) + geom_tile(aes(width=.95, height=.95)) + scale_fill_gradient(low="#000000", high="#ffffff", limits=[0,1]) + xlab("Mutation positions") + ylab("Individuals")

p2.save(filename = 'plots/test/15x15_heat_out_vae.png', height=5, width=10, units = 'in', res=1000)
