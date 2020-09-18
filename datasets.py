import os
import random

import tensorflow as tf

import settings

import xml.etree.ElementTree as ET


class Database(object):
    def parse_object(self, obj):
        label = -1
        xmin = -1
        xmax = -1
        ymin = -1
        ymax = -1
        for child in obj:
            if child.tag == 'name':
                if child.text == 'with_mask':
                    label = 1
                else:
                    label = 0
            if child.tag == 'bndbox':
                for child_child in child:
                    if child_child.tag == 'xmin':
                        xmin = int(child_child.text)
                    elif child_child.tag == 'ymin':
                        ymin = int(child_child.text)
                    elif child_child.tag == 'xmax':
                        xmax = int(child_child.text)
                    elif child_child.tag == 'ymax':
                        ymax = int(child_child.text)

        assert(label != -1)
        assert(xmin != -1)
        assert(xmax != -1)
        assert(ymin != -1)
        assert(ymax != -1)
        return label, xmin, xmax, ymin, ymax

    def parse_annotations(self):
        objs = list()
        images = sorted(
            [os.path.join(self.images_folder, file_path) for file_path in os.listdir(self.images_folder)]
        )
        annotations = sorted(
            [os.path.join(self.annotations_folder, file_path) for file_path in os.listdir(self.annotations_folder)]
        )

        current_width = -1
        current_height = -1

        for i, annotation in enumerate(annotations):
            annotation = ET.parse(annotation)
            root = annotation.getroot()
            for child in root:
                if child.tag == 'size':
                    for child_child in child:
                        if child_child.tag == 'width':
                            current_width = int(child_child.text)
                        elif child_child.tag == 'height':
                            current_height = int(child_child.text)
                        elif child_child.tag == 'depth':
                            assert(3 == int(child_child.text))
                if child.tag == 'object':
                    label, xmin, xmax, ymin, ymax = self.parse_object(child)
                    objs.append((images[i], label, xmin, xmax, ymin, ymax, current_width, current_height))

        return objs

    def print_objs_statistics(self, objs):
        with_mask_count = 0
        without_mask_count = 0

        for obj in objs:
            label = obj[1]
            if label == 1:
                with_mask_count += 1
            else:
                without_mask_count += 1

        print(f'With mask count: {with_mask_count}')
        print(f'Without mask count: {without_mask_count}')

    def __init__(self):
        self.images_folder = os.path.join(settings.DATASET_ADDRESS, "images")
        self.annotations_folder = os.path.join(settings.DATASET_ADDRESS, "annotations")

        self.objs = self.parse_annotations()

        total = len(self.objs)
        self.train_objs = self.objs[:int(0.8 * total)]
        self.print_objs_statistics(self.train_objs)

        self.val_objs = self.objs[int(0.8 * total):int(0.9 * total)]
        self.print_objs_statistics(self.val_objs)

        self.test_objs = self.objs[int(0.9 * total):]
        self.print_objs_statistics(self.test_objs)

    def balance_objs(self, objs):
        label_0_objs = list()
        label_1_objs = list()

        for obj in objs:
            label = obj[1]
            if label == 0:
                label_0_objs.append(obj)
            else:
                label_1_objs.append(obj)

        diff = len(label_1_objs) - len(label_0_objs)

        for _ in range(diff):
            item = random.choice(label_0_objs)
            label_0_objs.append(item)

        result = label_1_objs + label_0_objs
        random.shuffle(result)
        return result

    def get_tf_dataset(self, batch_size, partition='train', all_info=False):
        """

        :param batch_size:
        :param partition:
        :param all_info:
        :return:
        """
        if all_info:
            def convert_to_img_label(img_address, annots):
                orig_img = tf.io.decode_png(tf.io.read_file(img_address), channels=3)
                img = orig_img[annots[3]:annots[4], annots[1]:annots[2], :]
                img = tf.cast(img, tf.float32) / 255
                img = tf.image.resize(img, (150, 150))

                return img, annots[0], orig_img, annots[1], annots[2], annots[3], annots[4]
        else:
            def convert_to_img_label(img_address, annots):
                orig_img = tf.io.decode_png(tf.io.read_file(img_address), channels=3)
                img = orig_img[annots[3]:annots[4], annots[1]:annots[2], :]
                img = tf.cast(img, tf.float32) / 255
                img = tf.image.resize(img, (150, 150))

                return img, annots[0]

        should_shuffle = False
        if partition == 'train':
            # objs = self.train_objs
            objs = self.balance_objs(self.train_objs)
            self.print_objs_statistics(objs)
            should_shuffle = True
        elif partition == 'val':
            objs = self.val_objs
        else:
            objs = self.test_objs

        if all_info:
            objs = self.train_objs + self.val_objs + self.test_objs
            final_objs = list()
            for obj in objs:
                if obj[0].endswith('maksssksksss521.png'):
                    final_objs.append(obj)
            objs = final_objs

        new_objs = list()
        new_objs_annot = list()
        for obj in objs:
            new_objs.append(obj[0])
            new_objs_annot.append(obj[1:])

        dataset = tf.data.Dataset.from_tensor_slices((new_objs, new_objs_annot))
        dataset = dataset.map(convert_to_img_label)
        if should_shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size)

        return dataset
