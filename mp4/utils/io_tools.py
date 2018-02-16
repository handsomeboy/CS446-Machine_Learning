"""Input and output helpers to load in data.
(This file will not be graded.)
"""

import numpy as np
import skimage
import os
from skimage import io


def read_dataset(data_txt_file, image_data_path):
    """Read data into a Python dictionary.

    Args:
        data_txt_file(str): path to the data txt file.
        image_data_path(str): path to the image directory.

    Returns:
        data(dict): A Python dictionary with keys 'image' and 'label'.
            The value of dict['image'] is a numpy array of dimension (N,8,8,3)
            containing the loaded images.

            The value of dict['label'] is a numpy array of dimension (N,1)
            containing the loaded label.

            N is the number of examples in the data split, the exampels should
            be stored in the same order as in the txt file.
    """
    data = {}
    data['image'] = None
    data['label'] = None

    start, end = None, None
    label = []

    image = None
    with open(data_txt_file) as data_file:
        for line in data_file:
            curr_label = line.split(',')
            if curr_label[1][0] == '-':
                label.append(-1)
            else:
                label.append(1)
            image_name = os.path.join(image_data_path, curr_label[0]+'.jpg')
            curr_image = io.imread(image_name)
            if image is None:
                image = curr_image.reshape(1,8,8,3)
            else:
                image = np.concatenate((image, curr_image[np.newaxis,...]), axis=0)

    label = np.array(label)
    label = label.reshape(label.shape[0],1)

    data['label'] = label
    data['image'] = image

    return data
