import tensorflow as tf
import os
from termcolor import cprint, colored
import numpy as np
from PIL import Image
from keras.models import *
from keras.layers import *
from Sub_Functions.MSA_GAN import *
from Sub_Functions.Attention import *


class Anime_Generation(object):

    def __init__(self, images, epochs, training=True):

        self.images = images
        self.img_shape = (64, 64, 3)

        self.epochs = epochs

        self.gf = 64
        self.df = 64

        # resolution of images
        self.GENERATE_SQUARE = 96  # 96x96
        self.channels = 3 # RGB

        # Preview images for viewing samples during training
        self.PREVIEW_ROWS = 4
        self.PREVIEW_COLS = 7
        self.PREVIEW_MARGIN = 16

        # Size vector to generate images from (noise vector)
        self.SEED_SIZE = 100

        self.BATCH_SIZE = 1
        self.BUFFER_SIZE = 600
        self.training = training

        # defining losses
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        # defining optimizer
        self.generator_optimizer = tf.keras.optimizers.Adam(1.5e-4, 0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1.5e-4, 0.5)

    def Generator(self):

        # Input seed vector
        seed = Input(shape=(self.SEED_SIZE,))
        x = Dense(8 * 8 * 64)(seed)
        x = Reshape((8, 8, 64))(x)
        # First deconvolution
        x = UpSampling2D((4, 4))(x)
        x = Conv2D(128, (3, 3), (1, 1), padding="same")(x)
        x = ReLU()(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, (3, 3), (1, 1), padding="same")(x)
        x = ReLU()(x)
        x = BatchNormalization()(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(128, (3, 3), (1, 1), padding="same")(x)
        x = ReLU()(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), (1, 1), padding="same")(x)
        x = ReLU()(x)
        x = BatchNormalization()(x)
        x = Conv2D(self.channels, (3, 3), (1, 1), padding="same", activation="tanh")(x)

        # Final model
        generator = Model(inputs=seed, outputs=x)

        return generator

    @staticmethod
    def Discriminator():
        # discriminator network
        discriminator = Sequential([
            Conv2D(32, (5, 5), (2, 2), padding="same", input_shape=[64, 64, 3]),
            LeakyReLU(0.2),
            Dropout(0.3),
            Conv2D(64, (5, 5), (2, 2), padding="same"),
            LeakyReLU(0.2),
            Dropout(0.3),
            BatchNormalization(),
            Conv2D(64, (5, 5), (2, 2), padding="same"),
            LeakyReLU(0.2),
            Dropout(0.3),
            BatchNormalization(),
            Conv2D(128, (5, 5), (2, 2), padding="same"),
            LeakyReLU(0.2),
            Dropout(0.3),
            Flatten(),
            Dense(1, activation='sigmoid')
        ])

        return discriminator

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def train_step(self, images):
        seed = tf.random.normal([self.BATCH_SIZE, self.SEED_SIZE])

        generator = self.Generator()
        discriminator = self.Discriminator()

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(seed, training=True)

            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss,
                                                       generator.trainable_variables)  # calculating gradients of generator
            gradients_of_discriminator = disc_tape.gradient(disc_loss,
                                                            discriminator.trainable_variables)  # calculating gradients of discriminator

            self.generator_optimizer.apply_gradients(
                zip(gradients_of_generator, generator.trainable_variables))  # applying gradients with the optimizer
            self.discriminator_optimizer.apply_gradients(
                zip(gradients_of_discriminator,
                    discriminator.trainable_variables))  # applying gradients with the optimizer

            # OP = Optimization(generator, fake_output)
            # generator = OP.main_update_hyperparameters()
        return gen_loss, disc_loss, generator

    def dc_generator(self):

        model = Sequential()

        model.add(Dense(128 * 8 * 8, activation="relu", input_dim=self.SEED_SIZE))
        model.add(Reshape((8, 8, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        # model.summary()

        noise = Input(shape=(self.SEED_SIZE,))
        img = model(noise)

        return Model(noise, img)

    def dc_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=[64, 64, 3], padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        # model.summary()

        img = Input(shape=[64, 64, 3])
        validity = model(img)

        return Model(img, validity)

    def DCGAN(self):
        cprint("===================================", color='blue')
        print(colored("[⚠️] DCGAN Anime Generation", 'magenta', on_color='on_grey'))
        cprint("===================================", color='blue')
        train_dataset = tf.data.Dataset.from_tensor_slices(self.images).shuffle(self.BUFFER_SIZE).batch(
            self.BATCH_SIZE)

        count = 0
        out_put = []
        for image_batch in train_dataset:
            data = {}
            gen_loss_list = []
            disc_loss_list = []
            for epoch in range(self.epochs):
                seed = tf.random.normal([self.BATCH_SIZE, self.SEED_SIZE])

                generator = self.dc_generator()
                discriminator = self.dc_discriminator()

                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    generated_images = generator(seed, training=True)

                    real_output = discriminator(image_batch, training=True)
                    fake_output = discriminator(generated_images, training=True)

                    gen_loss = self.generator_loss(fake_output)
                    disc_loss = self.discriminator_loss(real_output, fake_output)

                    gradients_of_generator = gen_tape.gradient(gen_loss,
                                                               generator.trainable_variables)  # calculating gradients of generator
                    gradients_of_discriminator = disc_tape.gradient(disc_loss,
                                                                    discriminator.trainable_variables)  # calculating gradients of discriminator

                    self.generator_optimizer.apply_gradients(
                        zip(gradients_of_generator,
                            generator.trainable_variables))  # applying gradients with the optimizer
                    self.discriminator_optimizer.apply_gradients(
                        zip(gradients_of_discriminator,
                            discriminator.trainable_variables))  # applying gradients with the optimizer

                cprint(f'[⁉️] Epoch {epoch + 1}/{self.epochs}, GEN LOSS={gen_loss},DISC LOSS={disc_loss}  ',
                       color='grey',
                       on_color='on_cyan')

                gen_loss_list.append(gen_loss)
                disc_loss_list.append(disc_loss)

            g_loss = sum(gen_loss_list) / len(gen_loss_list)
            d_loss = sum(disc_loss_list) / len(disc_loss_list)

            cprint(f'[⁉️] Images {count + 1}/{len(train_dataset)}, GEN LOSS={g_loss},DISC LOSS={d_loss}  ', color='grey',
                   on_color='on_cyan')

            cprint(f'[✅] Completed Training for Images {count + 1}/{len(train_dataset)}.. ', color='grey', on_color='on_green')

            count += 1
            Generated_images = []
            for i in range(10):
                # Generate random noise for image generation
                noise = tf.random.normal([1, self.SEED_SIZE])

                # Generate an image using the trained augmentation model
                generated_image = generator(noise, training=False)

                # Normalize the generated image
                generated_image = 0.5 * generated_image + 0.5
                generated_image = generated_image[0, :, :, :]

                # Save the generated image as a JPEG file
                generated_image = np.asarray(generated_image)

                im = Image.fromarray((generated_image * 255).astype(np.uint8))
                Generated_images.append(np.array(im))

            data['original_image'] = image_batch.numpy()[0]
            data['generated_image'] = Generated_images
            out_put.append(data)
        cprint('[✅] Image Generation Completed ', color='grey', on_color='on_green')

        return out_put

    def CBAM_generator(self):

        # Input seed vector
        seed = Input(shape=(self.SEED_SIZE,))
        x = Dense(8 * 8 * 256)(seed)
        x = Reshape((8, 8, 256))(x)

        # First deconvolution
        x = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same")(x)
        x = channel_attention_module(x)
        x = ReLU()(x)
        x = BatchNormalization()(x)

        x = Conv2DTranspose(32, (5, 5), strides=(2, 2), padding="same")(x)
        x = mutual_attention(x)
        x = ReLU()(x)
        x = BatchNormalization()(x)

        x = Conv2DTranspose(3, (5, 5), strides=(2, 2), padding="same", activation="tanh")(x)

        # Final model
        generator = Model(inputs=seed, outputs=x)

        return generator

    def CBAM(self):
        cprint("===================================", color='blue')
        print(colored("[⚠️] CBAM Anime Generation", 'magenta', on_color='on_grey'))
        cprint("===================================", color='blue')
        train_dataset = tf.data.Dataset.from_tensor_slices(self.images).shuffle(self.BUFFER_SIZE).batch(
            self.BATCH_SIZE)

        count = 0
        out_put = []
        for image_batch in train_dataset:
            data = {}
            gen_loss_list = []
            disc_loss_list = []
            for epoch in range(self.epochs):
                seed = tf.random.normal([self.BATCH_SIZE, self.SEED_SIZE])

                generator = self.CBAM_generator()
                discriminator = self.Discriminator()

                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    generated_images = generator(seed, training=True)

                    real_output = discriminator(image_batch, training=True)
                    fake_output = discriminator(generated_images, training=True)

                    gen_loss = self.generator_loss(fake_output)
                    disc_loss = self.discriminator_loss(real_output, fake_output)

                    gradients_of_generator = gen_tape.gradient(gen_loss,
                                                               generator.trainable_variables)  # calculating gradients of generator
                    gradients_of_discriminator = disc_tape.gradient(disc_loss,
                                                                    discriminator.trainable_variables)  # calculating gradients of discriminator

                    self.generator_optimizer.apply_gradients(
                        zip(gradients_of_generator,
                            generator.trainable_variables))  # applying gradients with the optimizer
                    self.discriminator_optimizer.apply_gradients(
                        zip(gradients_of_discriminator,
                            discriminator.trainable_variables))  # applying gradients with the optimizer

                cprint(f'[⁉️] Epoch {epoch + 1}/{self.epochs}, GEN LOSS={gen_loss},DISC LOSS={disc_loss}  ',
                       color='grey',
                       on_color='on_cyan')

                gen_loss_list.append(gen_loss)
                disc_loss_list.append(disc_loss)

            g_loss = sum(gen_loss_list) / len(gen_loss_list)
            d_loss = sum(disc_loss_list) / len(disc_loss_list)

            cprint(f'[⁉️] Images {count + 1}/{len(train_dataset)}, GEN LOSS={g_loss},DISC LOSS={d_loss}  ', color='grey',
                   on_color='on_cyan')

            cprint(f'[✅] Completed Training for Images {count + 1}/{len(train_dataset)}.. ', color='grey', on_color='on_green')

            count += 1
            Generated_images = []
            for i in range(10):
                # Generate random noise for image generation
                noise = tf.random.normal([1, self.SEED_SIZE])

                # Generate an image using the trained augmentation model
                generated_image = generator(noise, training=False)

                # Normalize the generated image
                generated_image = 0.5 * generated_image + 0.5
                generated_image = generated_image[0, :, :, :]

                # Save the generated image as a JPEG file
                generated_image = np.asarray(generated_image)

                im = Image.fromarray((generated_image * 255).astype(np.uint8))
                Generated_images.append(np.array(im))

            data['original_image'] = image_batch.numpy()[0]
            data['generated_image'] = Generated_images
            out_put.append(data)
        cprint('[✅] Image Generation Completed ', color='grey', on_color='on_green')

        return out_put

    def style_gan_generator(self):

        # Input seed vector
        seed = Input(shape=(self.SEED_SIZE,))
        x = Dense(8 * 8 * 256)(seed)
        x = Reshape((8, 8, 256))(x)

        # First deconvolution
        x = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same")(x)
        x = mutual_attention(x)
        x = ReLU()(x)
        x = BatchNormalization()(x)

        x = Conv2DTranspose(32, (5, 5), strides=(2, 2), padding="same")(x)
        x = mutual_attention(x)
        x = ReLU()(x)
        x = BatchNormalization()(x)

        x = Conv2DTranspose(self.channels, (5, 5), strides=(2, 2), padding="same", activation="tanh")(x)

        # Final model
        generator = Model(inputs=seed, outputs=x)

        return generator

    def SRGAN(self):
        cprint("===================================", color='blue')
        print(colored("[⚠️] SRGAN Anime Generation", 'magenta', on_color='on_grey'))
        cprint("===================================", color='blue')
        train_dataset = tf.data.Dataset.from_tensor_slices(self.images).shuffle(self.BUFFER_SIZE).batch(
            self.BATCH_SIZE)

        count = 0
        out_put = []
        for image_batch in train_dataset:
            data = {}
            gen_loss_list = []
            disc_loss_list = []
            for epoch in range(self.epochs):
                seed = tf.random.normal([self.BATCH_SIZE, self.SEED_SIZE])

                generator = self.style_gan_generator()
                discriminator = self.Discriminator()

                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    generated_images = generator(seed, training=True)

                    real_output = discriminator(image_batch, training=True)
                    fake_output = discriminator(generated_images, training=True)

                    gen_loss = self.generator_loss(fake_output)
                    disc_loss = self.discriminator_loss(real_output, fake_output)

                    gradients_of_generator = gen_tape.gradient(gen_loss,
                                                               generator.trainable_variables)  # calculating gradients of generator
                    gradients_of_discriminator = disc_tape.gradient(disc_loss,
                                                                    discriminator.trainable_variables)  # calculating gradients of discriminator

                    self.generator_optimizer.apply_gradients(
                        zip(gradients_of_generator,
                            generator.trainable_variables))  # applying gradients with the optimizer
                    self.discriminator_optimizer.apply_gradients(
                        zip(gradients_of_discriminator,
                            discriminator.trainable_variables))  # applying gradients with the optimizer

                cprint(f'[⁉️] Epoch {epoch + 1}/{self.epochs}, GEN LOSS={gen_loss},DISC LOSS={disc_loss}  ',
                       color='grey',
                       on_color='on_cyan')

                gen_loss_list.append(gen_loss)
                disc_loss_list.append(disc_loss)

            g_loss = sum(gen_loss_list) / len(gen_loss_list)
            d_loss = sum(disc_loss_list) / len(disc_loss_list)

            cprint(f'[⁉️] Images {count + 1}/{len(train_dataset)}, GEN LOSS={g_loss},DISC LOSS={d_loss}  ', color='grey',
                   on_color='on_cyan')

            cprint(f'[✅] Completed Training for Images {count + 1}/{len(train_dataset)}.. ', color='grey', on_color='on_green')

            count += 1
            Generated_images = []
            for i in range(10):
                # Generate random noise for image generation
                noise = tf.random.normal([1, self.SEED_SIZE])

                # Generate an image using the trained augmentation model
                generated_image = generator(noise, training=False)

                # Normalize the generated image
                generated_image = 0.5 * generated_image + 0.5
                generated_image = generated_image[0, :, :, :]

                # Save the generated image as a JPEG file
                generated_image = np.asarray(generated_image)

                im = Image.fromarray((generated_image * 255).astype(np.uint8))
                Generated_images.append(np.array(im))

            data['original_image'] = image_batch.numpy()[0]
            data['generated_image'] = Generated_images
            out_put.append(data)
        cprint('[✅] Image Generation Completed ', color='grey', on_color='on_green')

        return out_put

    def lds_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.SEED_SIZE))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        noise = Input(shape=(self.SEED_SIZE,))
        img = model(noise)

        return Model(noise, img)

    def lds_discriminator(self):

        model = Sequential()

        model.add(Flatten(input_shape=[64, 64, 3]))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        # (!!!) No softmax
        model.add(Dense(1))

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def LDS_GAN(self):
        cprint("===================================", color='blue')
        print(colored("[⚠️] LDSGAN Anime Generation", 'magenta', on_color='on_grey'))
        cprint("===================================", color='blue')
        train_dataset = tf.data.Dataset.from_tensor_slices(self.images).shuffle(self.BUFFER_SIZE).batch(
            self.BATCH_SIZE)

        count = 0
        out_put = []
        for image_batch in train_dataset:
            data = {}
            gen_loss_list = []
            disc_loss_list = []
            for epoch in range(self.epochs):
                seed = tf.random.normal([self.BATCH_SIZE, self.SEED_SIZE])

                generator = self.lds_generator()
                discriminator = self.lds_discriminator()

                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    generated_images = generator(seed, training=True)

                    real_output = discriminator(image_batch, training=True)
                    fake_output = discriminator(generated_images, training=True)

                    gen_loss = self.generator_loss(fake_output)
                    disc_loss = self.discriminator_loss(real_output, fake_output)

                    gradients_of_generator = gen_tape.gradient(gen_loss,
                                                               generator.trainable_variables)  # calculating gradients of generator
                    gradients_of_discriminator = disc_tape.gradient(disc_loss,
                                                                    discriminator.trainable_variables)  # calculating gradients of discriminator

                    self.generator_optimizer.apply_gradients(
                        zip(gradients_of_generator,
                            generator.trainable_variables))  # applying gradients with the optimizer
                    self.discriminator_optimizer.apply_gradients(
                        zip(gradients_of_discriminator,
                            discriminator.trainable_variables))  # applying gradients with the optimizer

                cprint(f'[⁉️] Epoch {epoch + 1}/{self.epochs}, GEN LOSS={gen_loss},DISC LOSS={disc_loss}  ',
                       color='grey',
                       on_color='on_cyan')

                gen_loss_list.append(gen_loss)
                disc_loss_list.append(disc_loss)

            g_loss = sum(gen_loss_list) / len(gen_loss_list)
            d_loss = sum(disc_loss_list) / len(disc_loss_list)

            cprint(f'[⁉️] Images {count + 1}/{len(train_dataset)}, GEN LOSS={g_loss},DISC LOSS={d_loss}  ', color='grey',
                   on_color='on_cyan')

            cprint(f'[✅] Completed Training for Images {count + 1}/{len(train_dataset)}.. ', color='grey', on_color='on_green')

            count += 1
            Generated_images = []
            for i in range(10):
                # Generate random noise for image generation
                noise = tf.random.normal([1, self.SEED_SIZE])

                # Generate an image using the trained augmentation model
                generated_image = generator(noise, training=False)

                # Normalize the generated image
                generated_image = 0.5 * generated_image + 0.5
                generated_image = generated_image[0, :, :, :]

                # Save the generated image as a JPEG file
                generated_image = np.asarray(generated_image)

                im = Image.fromarray((generated_image * 255).astype(np.uint8))
                Generated_images.append(np.array(im))

            data['original_image'] = image_batch.numpy()[0]
            data['generated_image'] = Generated_images
            out_put.append(data)
        cprint('[✅] Image Generation Completed ', color='grey', on_color='on_green')

        return out_put

    def ts_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)
        d5 = conv2d(d4, self.gf*8)
        d6 = conv2d(d5, self.gf*8)
        d7 = conv2d(d6, self.gf*8)

        # Upsampling
        u1 = deconv2d(d6, d5, self.gf * 8)
        u2 = deconv2d(u1, d4, self.gf * 8)
        u3 = deconv2d(u2, d3, self.gf * 8)
        u4 = deconv2d(u3, d2, self.gf * 4)
        u5 = deconv2d(u4, d1, self.gf * 2)
        u6 = UpSampling2D(size=2)(u5)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u6)

        return Model(d0, output_img)

    def ts_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([img_A, img_B], validity)

    def FISTNet(self):
        cprint("===================================", color='blue')
        print(colored("[⚠️] FISTNet Anime Generation", 'magenta', on_color='on_grey'))
        cprint("===================================", color='blue')
        train_dataset = tf.data.Dataset.from_tensor_slices(self.images).shuffle(self.BUFFER_SIZE).batch(
            self.BATCH_SIZE)

        generator = self.ts_generator()
        discriminator = self.ts_discriminator()

        count = 0
        out_put = []
        for image_batch in train_dataset:
            data = {}
            gen_loss_list = []
            disc_loss_list = []
            for epoch in range(self.epochs):
                seed = tf.random.normal([self.BATCH_SIZE, self.img_shape[0], self.img_shape[1], self.img_shape[2]])

                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    generated_images = generator(seed, training=True)

                    real_output = discriminator([image_batch, image_batch], training=True)
                    fake_output = discriminator([generated_images, image_batch], training=True)

                    gen_loss = self.generator_loss(fake_output)
                    disc_loss = self.discriminator_loss(real_output, fake_output)

                    gradients_of_generator = gen_tape.gradient(gen_loss,
                                                               generator.trainable_variables)  # calculating gradients of generator
                    gradients_of_discriminator = disc_tape.gradient(disc_loss,
                                                                    discriminator.trainable_variables)  # calculating gradients of discriminator

                    self.generator_optimizer.apply_gradients(
                        zip(gradients_of_generator,
                            generator.trainable_variables))  # applying gradients with the optimizer
                    self.discriminator_optimizer.apply_gradients(
                        zip(gradients_of_discriminator,
                            discriminator.trainable_variables))  # applying gradients with the optimizer

                cprint(f'[⁉️] Epoch {epoch + 1}/{self.epochs}, GEN LOSS={gen_loss},DISC LOSS={disc_loss}  ',
                       color='grey',
                       on_color='on_cyan')

                gen_loss_list.append(gen_loss)
                disc_loss_list.append(disc_loss)

            g_loss = sum(gen_loss_list) / len(gen_loss_list)
            d_loss = sum(disc_loss_list) / len(disc_loss_list)

            cprint(f'[⁉️] Images {count + 1}/{len(train_dataset)}, GEN LOSS={g_loss},DISC LOSS={d_loss}  ', color='grey',
                   on_color='on_cyan')

            cprint(f'[✅] Completed Training for Images {count + 1}/{len(train_dataset)}.. ', color='grey', on_color='on_green')

            count += 1
            Generated_images = []
            for i in range(10):
                # Generate random noise for image generation
                noise = tf.random.normal([1, self.img_shape[0], self.img_shape[1], self.img_shape[2]])

                # Generate an image using the trained augmentation model
                generated_image = generator(noise, training=False)

                # Normalize the generated image
                generated_image = 0.5 * generated_image + 0.5
                generated_image = generated_image[0, :, :, :]

                # Save the generated image as a JPEG file
                generated_image = np.asarray(generated_image)

                im = Image.fromarray((generated_image * 255).astype(np.uint8))
                Generated_images.append(np.array(im))

            data['original_image'] = image_batch.numpy()[0]
            data['generated_image'] = Generated_images
            out_put.append(data)
        cprint('[✅] Image Generation Completed ', color='grey', on_color='on_green')

        return out_put

    def HSDN(self):
        cprint("===================================", color='blue')
        print(colored("[⚠️] HSDN Anime Generation", 'magenta', on_color='on_grey'))
        cprint("===================================", color='blue')
        train_dataset = tf.data.Dataset.from_tensor_slices(self.images).shuffle(self.BUFFER_SIZE).batch(
            self.BATCH_SIZE)

        count = 0
        out_put = []
        for image_batch in train_dataset:
            data = {}
            gen_loss_list = []
            disc_loss_list = []
            for epoch in range(self.epochs):
                gen_loss, disc_loss, generator = self.train_step(image_batch)

                cprint(f'[⁉️] Epoch {epoch + 1}/{self.epochs}, GEN LOSS={gen_loss},DISC LOSS={disc_loss}  ',
                       color='grey',
                       on_color='on_cyan')

                gen_loss_list.append(gen_loss)
                disc_loss_list.append(disc_loss)

            g_loss = sum(gen_loss_list) / len(gen_loss_list)
            d_loss = sum(disc_loss_list) / len(disc_loss_list)

            cprint(f'[⁉️] Images {count + 1}/{len(train_dataset)}, GEN LOSS={g_loss},DISC LOSS={d_loss}  ', color='grey',
                   on_color='on_cyan')

            cprint(f'[✅] Completed Training for Images {count + 1}/{len(train_dataset)}.. ', color='grey', on_color='on_green')

            count += 1
            Generated_images = []
            for i in range(10):
                # Generate random noise for image generation
                noise = tf.random.normal([1, self.SEED_SIZE])

                # Generate an image using the trained augmentation model
                generated_image = generator(noise, training=False)

                # Normalize the generated image
                generated_image = 0.5 * generated_image + 0.5
                generated_image = generated_image[0, :, :, :]

                # Save the generated image as a JPEG file
                generated_image = np.asarray(generated_image)

                im = Image.fromarray((generated_image * 255).astype(np.uint8))
                Generated_images.append(np.array(im))
                #
                # im.save(f'temp\\{count + 1}\\{i}.jpg')
            data['original_image'] = image_batch.numpy()[0]
            data['generated_image'] = Generated_images
            out_put.append(data)
        cprint('[✅] Image Generation Completed ', color='grey', on_color='on_green')

        return out_put

    def MSADGAN(self, opt, epochs):
        if opt == 1:
            cprint("===================================", color='blue')
            print(colored("[⚠️] MSADGAN Anime Generation", 'magenta', on_color='on_grey'))
            cprint("===================================", color='blue')
        else:
            cprint("===================================", color='blue')
            print(colored("[⚠️] GAN Anime Generation", 'magenta', on_color='on_grey'))
            cprint("===================================", color='blue')

        dataloader = AnimeDataset(self.images, device=DEVICE)
        BATCH_SIZE = 8
        train_dataloader = DataLoader(dataloader,
                                      batch_size=8,
                                      shuffle=True)
        discriminator = Discriminator(device=DEVICE)
        generator = Generator(opt, device=DEVICE)
        optimizer_d = Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer_g = Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        trainer = Trainer(generator, discriminator, optimizer_g, optimizer_d, latent_size=128, load_pretrained=False)

        for epoch in range(epochs):
            if epoch == 499:
                trainer.fit(train_dataloader, epoch, save=True)
            trainer.fit(train_dataloader, epoch, save=False)

        test_dataloader = DataLoader(dataloader,
                                     batch_size=1,
                                     shuffle=True)

        out_put = trainer.Anime_generation(test_dataloader)

        cprint('[✅] Image Generation Completed ', color='grey', on_color='on_green')

        return out_put

