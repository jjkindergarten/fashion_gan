import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from src.modeling.ConGAN import make_generator_model
from src.modeling.ConGAN import make_discriminator_model, make_generator_model, make_discriminator_model_sigmoid, \
    make_generator_model_relu




def discriminator_loss(real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    return cross_entropy(tf.ones_like(fake_output), fake_output)

@tf.function
def train_step(generator, discriminator, generator_optimizer, discriminator_optimizer, images, noise_dim):
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


def generate_and_save_images(model, epoch, test_input, save_path):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig(os.path.join(save_path, 'image_at_epoch_{:04d}.png'.format(epoch)))
    plt.show()



def train(generator, discriminator, generator_optimizer, discriminator_optimizer,
          seed, dataset, epochs, checkpoint, checkpoint_prefix, save_path, noise_dim):
    for epoch in range(epochs):
        start = time.time()
        print('G_Loss', 'D_Loss', 'D_real', 'D_fake')
        i = 0
        for image_batch in dataset:
            if i % 2 == 0:
                gen_loss, disc_loss, real_mean, fake_mean = train_step(generator, discriminator, generator_optimizer,
                                                                       discriminator_optimizer, image_batch, noise_dim)
            else:
                gen_loss, disc_loss, real_mean, fake_mean = train_step_G(generator, discriminator, generator_optimizer,
                                                                       discriminator_optimizer, image_batch, noise_dim)

            print(gen_loss.numpy(), disc_loss.numpy(), real_mean.numpy(), fake_mean.numpy())

        generate_and_save_images(generator, epoch, seed, save_path)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    generate_and_save_images(generator, epochs, seed, save_path)


if __name__ == "__main__":

    BUFFER_SIZE = 60000
    BATCH_SIZE = 256
    EPOCHS = 50
    noise_dim = 100
    num_examples_to_generate = 16

    test_num = 8
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

    generator = make_generator_model()
    discriminator = make_discriminator_model_sigmoid()

    noise = tf.random.normal([1, 100])
    generated_image = generator(noise, training=False)
    plt.imshow(generated_image[0, :, :, 0], cmap='gray')

    decision = discriminator(generated_image)
    print(decision)

    generator_optimizer = tf.keras.optimizers.Adam(4*1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(2*1e-4)

    # save checkpoint
    checkpoint_dir = os.path.join(SAVE_PATH, 'training_checkpoints')
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    train(generator, discriminator, generator_optimizer, discriminator_optimizer, seed, train_dataset, EPOCHS, checkpoint,
          checkpoint_prefix, SAVE_PATH, noise_dim)

    generator.save(os.path.join(SAVE_PATH, 'generator.h5'))
    discriminator.save(os.path.join(SAVE_PATH, 'discriminator.h5'))

