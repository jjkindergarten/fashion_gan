import tensorflow as tf
import matplotlib.pyplot as plt
import os
import glob
import imageio

def make_32x32_dataset(img):
    img = tf.image.resize(img, [32, 32])
    img = tf.clip_by_value(img, 0, 255)
    img = img / 127.5 - 1

    return img


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


anim_file = 'dcgan.gif'
test_num = 22
SAVE_PATH = './result/MINIST/res_{}'.format(test_num)

with imageio.get_writer(os.path.join(SAVE_PATH,anim_file), mode='I') as writer:
  filenames = glob.glob(os.path.join(SAVE_PATH, 'image_at_epoch_*.png'))
  filenames = sorted(filenames)
  last = -1
  for i,filename in enumerate(filenames):
    frame = 2*(i**0.5)
    if round(frame) > round(last):
      last = frame
    else:
      continue
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)