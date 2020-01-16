import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from src.modeling.ConGAN import make_generator_model
from src.modeling.ConGAN import make_discriminator_model, make_generator_model, make_discriminator_model_sigmoid, \
    make_generator_model_relu, make_discriminator_model_layernorm, make_discriminator_model_sigmoid2, make_generator_model2
import pandas as pd
from functools import partial




def discriminator_loss(real_output, fake_output):
    # loss function of discriminator
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    # for the real image, the loss is the distance between 1 and the classification of all real image
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    # for the fake image, the loss is the distance between 0 and the classification of all fake image
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    # for the generator, the loss is the distance between 1 and the classification of fake image
    # i think with the loss function using sigmoid to outcome probability from discrminator makes more sense
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    return cross_entropy(tf.ones_like(fake_output), fake_output)

@tf.function
def train_step_wgan(generator, discriminator, generator_optimizer, discriminator_optimizer, images, noise_dim,
                    grad_penalty_weight = 10):
    noise = tf.random.normal([images.shape[0], noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = -tf.reduce_mean(fake_output)
        disc_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

        # gradient penalty here (the only difference)
        gp = gradient_penalty(partial(discriminator, training=True), images, generated_images)
        disc_loss += grad_penalty_weight * gp

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    # return these values to monitor the process of training
    return gen_loss, disc_loss, tf.reduce_mean(real_output), tf.reduce_mean(fake_output)

@tf.function
def train_step(generator, discriminator, generator_optimizer, discriminator_optimizer, images, noise_dim):
    # train generator and discriminator simatelously
    noise = tf.random.normal([images.shape[0], noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss, tf.reduce_mean(real_output), tf.reduce_mean(fake_output)

@tf.function
def train_step_G(generator, discriminator, generator_optimizer, discriminator_optimizer, images, noise_dim):
    # train generator only
    # in some code i saw, some separately the training function of G and D.
    # In this way, G and D use different noise to generate fake image
    # does this help ?
    noise = tf.random.normal([images.shape[0], noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=False)
        fake_output = discriminator(generated_images, training=False)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    return gen_loss, disc_loss, tf.reduce_mean(real_output), tf.reduce_mean(fake_output)

@tf.function
def train_step_G_wgan(generator, discriminator, generator_optimizer, discriminator_optimizer, images, noise_dim):
    # train generator only
    # in some code i saw, some separately the training function of G and D.
    # In this way, G and D use different noise to generate fake image
    # does this help ?
    noise = tf.random.normal([images.shape[0], noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=False)
        fake_output = discriminator(generated_images, training=False)

        gen_loss = -tf.reduce_mean(fake_output)
        disc_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    return gen_loss, disc_loss, tf.reduce_mean(real_output), tf.reduce_mean(fake_output)


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


def generate_and_save_images(model, epoch, test_input, save_path):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm)
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig(os.path.join(save_path, 'image_at_epoch_{:04d}.png'.format(epoch)))
    plt.show()



def train(generator, discriminator, generator_optimizer, discriminator_optimizer,
          seed, dataset, epochs, checkpoint, checkpoint_prefix, save_path, noise_dim, d_update_freq = 4, wgan = False):
    for epoch in range(epochs):
        start = time.time()
        print('G_Loss', 'D_Loss', 'D_real', 'D_fake')
        i = 0
        loss_list = []
        for image_batch in dataset:
            # change the number below help adjust the frequency of update of discriminator per update of generator
            if i % d_update_freq == 0:
                if wgan:
                    gen_loss, disc_loss, real_mean, fake_mean = train_step_wgan(generator, discriminator, generator_optimizer,
                                                                       discriminator_optimizer, image_batch, noise_dim)
                else:
                    gen_loss, disc_loss, real_mean, fake_mean = train_step(generator, discriminator,
                                                                                generator_optimizer,discriminator_optimizer,
                                                                                image_batch, noise_dim)
            else:
                if wgan:
                    gen_loss, disc_loss, real_mean, fake_mean = train_step_G_wgan(generator, discriminator, generator_optimizer,
                                                                       discriminator_optimizer, image_batch, noise_dim)
                else:
                    gen_loss, disc_loss, real_mean, fake_mean = train_step_G(generator, discriminator, generator_optimizer,
                                                                       discriminator_optimizer, image_batch, noise_dim)
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
    test_num = 18
    SAVE_PATH = './result/MINIST/res_{}'.format(test_num)

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    seed = tf.random.normal([num_examples_to_generate, noise_dim])

    # load data
    # (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    (train_images, train_labels), (_, _) = tf.keras.datasets.fashion_mnist.load_data()


    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

    # Batch and shuffle the data
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    #
    # generator = make_generator_model_relu()              # use this generator for task 7
    # discriminator = make_discriminator_model_sigmoid()  # use this discriminator for task 6
    # discriminator = make_discriminator_model_layernorm() # use this discriminator for task 11
    generator = make_generator_model()
    discriminator = make_discriminator_model_layernorm()

    noise = tf.random.normal([1, 100])
    generated_image = generator(noise, training=False)
    plt.imshow(generated_image[0, :, :, 0], cmap='gray')

    decision = discriminator(generated_image)
    print(decision)

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

    # set wgan = True for task 10 and 11
    # set d_update_freq = 4 for task 9
    train(generator, discriminator, generator_optimizer, discriminator_optimizer, seed, train_dataset, EPOCHS, checkpoint,
          checkpoint_prefix, SAVE_PATH, noise_dim, d_update_freq = 1, wgan = True)

    generator.save(os.path.join(SAVE_PATH, 'generator.h5'))
    discriminator.save(os.path.join(SAVE_PATH, 'discriminator.h5'))

