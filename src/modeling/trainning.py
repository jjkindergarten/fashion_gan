import tensorflow as tf

import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from src.modeling.ConGAN import make_generator_model
from src.modeling.ConGAN import make_generator_model, make_discriminator_model
import pandas as pd
from functools import partial
from src.data.utilits import make_32x32_dataset, generate_and_save_images




def discriminator_loss_gan(real_output, fake_output):

    # loss function of discriminator
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    # for the real image, the loss is the distance between 1 and the classification of all real image
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    # for the fake image, the loss is the distance between 0 and the classification of all fake image
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def discriminator_loss_wgan_gp(real_output, fake_output):
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)


def generator_loss_gan(fake_output):
    # for the generator, the loss is the distance between 1 and the classification of fake image
    # i think with the loss function using sigmoid to outcome probability from discrminator makes more sense
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    return cross_entropy(tf.ones_like(fake_output), fake_output)

def generator_loss_wgan_gp(fake_output):
    return -tf.reduce_mean(fake_output)


@tf.function
def train_step_D(generator, discriminator, discriminator_optimizer, discriminator_loss, images, noise_dim, grad_penalty_weight = 10):
    # train generator and discriminator simatelously
    noise = tf.random.normal([images.shape[0], noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        disc_loss = discriminator_loss(real_output, fake_output)

        if grad_penalty_weight != 0:
            gp = gradient_penalty(partial(discriminator, training=True), images, generated_images)
            disc_loss += grad_penalty_weight * gp

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return disc_loss, tf.reduce_mean(real_output), tf.reduce_mean(fake_output)

@tf.function
def train_step_G(generator, discriminator, generator_optimizer, generator_loss, images, noise_dim):
    # train generator only
    # in some code i saw, some separately the training function of G and D.
    # In this way, G and D use different noise to generate fake image
    # does this help ?
    noise = tf.random.normal([images.shape[0], noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    return gen_loss


def gradient_penalty(f, real, fake=None):
    # this part i dont fully understand
    # but seems like it measures the distance between fake and real images ?

    def _interpolate(a, b=None):
        shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
        alpha = tf.random.uniform(shape=shape, minval=0., maxval=1.)
        inter = a + alpha * (b - a)
        inter.set_shape(a.shape)
        return inter

    x = _interpolate(real, fake)
    with tf.GradientTape() as t:
        t.watch(x)
        pred = f(x)
    grad = t.gradient(pred, x)
    norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
    gp = tf.reduce_mean((norm - 1.)**2)

    return gp



def train(generator, discriminator, generator_optimizer, discriminator_optimizer,
          seed, dataset, epochs, checkpoint, checkpoint_prefix, save_path, noise_dim, n_critic = 4, mode = 'gan'):
    if mode == 'wgan_gp':
        penalty_weight_value = 10
        discriminator_loss = discriminator_loss_wgan_gp
        generator_loss = generator_loss_wgan_gp
    elif mode == 'gan':
        penalty_weight_value = 0
        discriminator_loss = discriminator_loss_gan
        generator_loss = generator_loss_gan

    for epoch in range(epochs):
        start = time.time()
        print('G_Loss', 'D_Loss', 'D_real', 'D_fake')
        loss_list = []
        for image_batch in dataset:
            # change the number below help adjust the frequency of update of discriminator per update of generator
            for _ in range(n_critic):
                disc_loss, real_mean, fake_mean = train_step_D(generator, discriminator, discriminator_optimizer, discriminator_loss,
                                                               image_batch, noise_dim, grad_penalty_weight = penalty_weight_value)

            gen_loss = train_step_G(generator, discriminator, generator_optimizer, generator_loss, image_batch, noise_dim)

            loss_list.append([gen_loss.numpy(),disc_loss.numpy(),real_mean.numpy(),fake_mean.numpy()])

            print(gen_loss.numpy(), disc_loss.numpy(), real_mean.numpy(), fake_mean.numpy())

        generate_and_save_images(generator, epoch, seed, save_path)
        # save the learning process data
        loss_db = pd.DataFrame(loss_list, columns=['G_Loss', 'D_Loss', 'D_real', 'D_fake'])
        loss_db.to_csv(os.path.join(save_path, 'loss_db_{}.csv'.format(epoch)))

        # Save the model weight every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    generate_and_save_images(generator, epochs, seed, save_path)


if __name__ == "__main__":

    BUFFER_SIZE = 60000
    BATCH_SIZE = 256
    EPOCHS = 40
    noise_dim = 100
    num_examples_to_generate = 16

    # change the num of test_num for each task, would help save result in different folder
    test_num = 22
    SAVE_PATH = './result/MINIST/res_{}'.format(test_num)

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    seed = tf.random.normal([num_examples_to_generate, noise_dim])

    # load data
    (train_images, train_labels), (_, _) = tf.keras.datasets.fashion_mnist.load_data()

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    # change train image size from 28*28 to 32*32
    # easy to make a more complicated network archicture
    # train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

    train_images = make_32x32_dataset(train_images)

    # Batch and shuffle the data
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    generator = make_generator_model(hidden_layer_num = 3, hidden_layer_strides1_num = 0, basic_filter_num = 64,
                                     init_dim=4, filter_size = 4, norm = 'batch', act = 'leakyrelu')
    discriminator = make_discriminator_model(img_dim = 32, hidden_layer_num = 3, basic_filter_num = 64, filter_size = 4,
                         norm = 'layer', act = 'leakyrelu', dropout_ratio = 0.3, last_layer = 'dense', if_sigmoid = False)

    # change the learning rate here
    generator_optimizer = tf.keras.optimizers.Adam(2*1e-4,  beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2*1e-4,  beta_1=0.5)

    # save checkpoint
    checkpoint_dir = os.path.join(SAVE_PATH, 'training_checkpoints')
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)
    # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    train(generator, discriminator, generator_optimizer, discriminator_optimizer, seed, train_dataset, EPOCHS, checkpoint,
          checkpoint_prefix, SAVE_PATH, noise_dim, n_critic = 3, mode = 'wgan_gp')

    generator.save(os.path.join(SAVE_PATH, 'generator.h5'))
    discriminator.save(os.path.join(SAVE_PATH, 'discriminator.h5'))

