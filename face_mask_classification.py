import tensorflow as tf
from tensorflow.keras.applications.xception import Xception
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from datasets import Database


class FaceMaskClassification(object):
    def __init__(self):
        self.database = Database()
        self.tr_dataset = self.database.get_tf_dataset(partition='train', batch_size=64)
        self.val_dataset = self.database.get_tf_dataset(partition='val', batch_size=32)
        self.test_dataset = self.database.get_tf_dataset(partition='test', batch_size=1)

    def model(self, trainable=False):
        model = Xception(include_top=False, weights='imagenet')
        model.trainable = trainable

        inputs = tf.keras.Input(shape=(150, 150, 3))
        x = model(inputs, training=trainable)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(1)(x)
        return tf.keras.Model(inputs, outputs)

    def visualize_dataset(self, dataset):
        counter = 0
        for item in dataset:
            image, annots = item
            image = image[0, ...]
            annots = annots[0, ...]
            print(image.shape)

            plt.imshow(image)
            plt.show()
            print(annots)
            if counter == 3:
                break
            counter += 1

    def train(self):
        # model = VGG19(include_top=False, weights='imagenet')
        model = self.model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.BinaryAccuracy()]
        )

        self.visualize_dataset(self.tr_dataset)

        model.fit(self.tr_dataset, validation_data=self.val_dataset, epochs=10)
        model.save_weights('./checkpoints/model')

    def test(self):
        model = self.model()

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.BinaryAccuracy()]
        )
        model.load_weights('./checkpoints/model')
        model.evaluate(self.test_dataset)

    def demo(self):
        model = self.model()
        model.load_weights('./checkpoints/model')

        test_dataset = self.database.get_tf_dataset(partition='test', batch_size=1, all_info=True)
        for i, item in enumerate(test_dataset):
            img, annot, orig_img, y1, y2, x1, x2 = item
            x1 = x1[0, ...].numpy()
            x2 = x2[0, ...].numpy()
            y1 = y1[0, ...].numpy()
            y2 = y2[0, ...].numpy()

            prediction = model.predict(img)
            prediction = 1 / (1 + np.exp(prediction))

            prediction = prediction[0, 0]
            if annot == 1:
                prediction = 1 - prediction
                text = f'with_mask\n{prediction:0.2f}'
            else:
                text = f'no_mask\n{prediction:0.2f}'

            if prediction > 0.5:
                color = 'g'
            else:
                color = 'r'

            rect = Rectangle((y1, x1), y2 - y1, x2 - x1, fill=False, linewidth=4, edgecolor=color)

            position = list(rect.get_xy())
            position[0] += rect.get_width() / 2
            position[1] -= 20
            plt.imshow(orig_img[0, ...])
            plt.gca().add_patch(rect)
            plt.gca().annotate(text, position, color='w', weight='bold',
                               fontsize=18, ha='center', va='center')

            plt.show()
