import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Conv2D, Dense, Flatten, Reshape, LeakyReLU, Dropout, UpSampling2D, AveragePooling2D, \
    Conv2DTranspose, Input, Concatenate, Add, BatchNormalization, Activation, MultiHeadAttention

# подключена ли видеокарта?
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# сверточный блок нейронки( основной вычислительный блок на него
# приходится больше всего параметров)

def ResidualBlock(width):
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1, activation=keras.activations.swish)(x)
        x = layers.Conv2D(width / 2,kernel_size=1,activation=keras.activations.swish)(x)
        x = layers.Conv2D(width / 2,kernel_size=3,padding="same",activation=keras.activations.swish)(x)
        x = layers.Conv2D(width,kernel_size=1,activation=keras.activations.swish)(x)
        x = layers.Add()([x, residual])
        return x

    return apply


# понижающий блок ( понижает размер изображения в 2 раза и увеличивает
# количество фильтров до след в списке widths


def DownBlock(width, block_depth):
    def apply(x):
        x, skips = x
        height = x.shape[1]
        for _ in range(1 if height == 128 else block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
        return x

    return apply


# повышающий блок


def UpBlock(width, block_depth):
    def apply(x):
        x, skips = x
        height = x.shape[1]
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(1 if height == 128 // 2 else block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)
        return x

    return apply


# формирование архитектуры unet


def get_network(image_size, widths, block_depth):
    input_image = keras.Input(shape=(image_size, image_size, 3))
    x = input_image

    skips = []

    for width in widths[:-1]:
        x = DownBlock(width, block_depth)([x, skips])

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])(x)

    for width in reversed(widths[:-1]):
        x = UpBlock(width, block_depth)([x, skips])

    x = Concatenate()([x, input_image])
    x = layers.Conv2D(64, kernel_size=1, padding='same',
                      activation=keras.activations.swish)(x)
    x = layers.Conv2D(3, kernel_size=1, padding='same', kernel_initializer="zeros", activation='sigmoid')(x)

    return keras.Model(input_image, x, name="residual_unet")


# создаем регрессионую модель

class RegressionModel(keras.Model):
    def __init__(self, network):
        super().__init__()
        self.network = network
        self.lr = 1e-4
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    @tf.function
    def train_step(self, images):
        batch_size = tf.shape(images)[0]
        with tf.GradientTape() as tape:
            input_images = images[:, 0, :]
            output_images = images[:, 1, :]

            result = self.network(input_images, training=True)

            loss = tf.reduce_mean(
                (output_images - result) ** 2,
                axis=(1, 2, 3))

            gradients = tape.gradient(loss, self.network.trainable_weights)

        self.optimizer.apply_gradients(
            zip(gradients, self.network.trainable_weights))

        return loss


widths = [32, 64, 96, 128, 256]

block_depth = 2

batch_size = 16

img_channels = 3
clip_min = -1.0
clip_max = 1.0

image_size = 128

network = get_network(
    image_size=image_size,
    widths=widths,
    block_depth=block_depth
)

model = RegressionModel(network=network)

widths = [32, 64, 96, 128, 256]

block_depth = 2

image_size = 128

network = get_network(
    image_size=image_size,
    widths=widths,
    block_depth=block_depth
)

model.network.load_weights("regression_v1.h5")

# обучаем
epochs = 5000
hist = np.array(np.empty([0]))
from IPython.display import clear_output

for epoch in range(epochs):
    midloss = 0
    for step, x in enumerate(dataset):
        # if tf.shape(x)[0] == 14: #проверяем целостность батча
        midloss += tf.reduce_mean(model.train_step(x), axis=0)

        if (step % 100 == 0):
            clear_output(wait=True)
            print('эпоха ' + str(epoch))
            print('ошибка: ' + str(float(midloss / 10)))

        #  hist = np.append(hist, float(midloss/10))
        #  plt.plot(np.arange(0,len(hist)), hist)
        #  midloss = 0
        #  plt.show()
    #   if (step % 1000 == 0):
    #       model.ema_network.save("buffer"+str(step)+".h5")'''

model.run_generation(num_cols=4, num_rows=4)
