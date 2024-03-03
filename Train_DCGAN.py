from networks import *
from PIL import Image
import numpy as np
import scipy.misc as misc
from skimage import transform
import os
import sys
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
class DCGAN:
    def __init__(self):
        self.img = tf.placeholder(tf.float32, [None, IMG_H, IMG_W, IMG_C])
        self.z = tf.placeholder(tf.float32, [None, Z_DIM])
        G = Generator("generator")
        D = Discriminator("discriminator")
        self.real_logits = D(self.img)
        self.fake_img = G(self.z)
        self.fake_logits = D(self.fake_img)
        self.D_loss = - tf.reduce_mean(tf.log(tf.sigmoid(self.real_logits + EPSILON))) - tf.reduce_mean(tf.log(1 - tf.sigmoid(self.fake_logits) + EPSILON))
        self.G_loss = - tf.reduce_mean(tf.log(tf.sigmoid(self.fake_logits) + EPSILON))
        self.Opt_G = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(self.G_loss, var_list=G.var)
        self.Opt_D = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(self.D_loss, var_list=D.var)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self):
        file_path = "/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba/"
        file_names = os.listdir(file_path)
        print(len(file_names))
        saver = tf.train.Saver()
        batch = np.zeros([BATCHSIZE, IMG_H, IMG_W, IMG_C])
        dataset_size = len(file_names)
        num_epochs=50
        num_steps = (dataset_size + BATCHSIZE - 1) // BATCHSIZE  # Calculate number of steps based on dataset size and batch size

        for epoch in range(num_epochs):
            for step in range(num_steps):
                # Generate batch indices for this step
                start_idx = step * BATCHSIZE
                end_idx = min((step + 1) * BATCHSIZE, dataset_size)
                batch_filenames = np.random.choice(file_names[start_idx:end_idx], BATCHSIZE, replace=True)
                
                for j, filename in enumerate(batch_filenames):
                    img = np.array(Image.open(file_path + filename))
                    h = img.shape[0]
                    w = img.shape[1]
                    batch[j, :, :, :] = transform.resize(img[(h // 2 - 70):(h // 2 + 70), (w // 2 - 70):(w // 2 + 70), :], [64, 64]) / 127.5 - 1.0
                z = np.random.standard_normal([BATCHSIZE, Z_DIM])
                self.sess.run(self.Opt_D, feed_dict={self.img: batch, self.z: z})
                
                z = np.random.standard_normal([BATCHSIZE, Z_DIM])
                self.sess.run(self.Opt_G, feed_dict={self.z: z})
                
                if step % 20 == 0:
                    [D_loss, G_loss, fake_img] = self.sess.run([self.D_loss, self.G_loss, self.fake_img], feed_dict={self.img: batch, self.z: z})
                    print("Epoch: %d, Step: %d, D_loss: %f, G_loss: %f"%(epoch, step, D_loss, G_loss))
                    Image.fromarray(np.uint8((fake_img[0, :, :, :] + 1.0) * 127.5)).save("/kaggle/working/Semantic-Image-Inpainting-face-inpainting-TensorFlow/result/epoch_%d.jpg" % (epoch))
                
            if epoch % 5 == 0:
                saver.save(self.sess, "/kaggle/working/Semantic-Image-Inpainting-face-inpainting-TensorFlow/save_para/dcgan_epoch_%d.ckpt" % epoch)
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
if __name__ == "__main__":
    dcgan = DCGAN()
    dcgan.train()
