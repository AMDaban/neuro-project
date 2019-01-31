import copy
import cv2
import numpy as np
from skimage.measure import block_reduce


# MNN stands for 'My Neural Network'
# It is assumed that all parameters have been evaluated
# so please use MNNBuilder instead of creating object of this class directly
class MNN:
    def __init__(self, image_shape, gabor_filters, gabor_pooling_layer_kernel_shape):
        self._image_shape = image_shape
        self._gabor_filters = gabor_filters
        self.gabor_pooling_layer_kernel_shape = gabor_pooling_layer_kernel_shape

        self._gabor_filters_shape = None

        self._prepare()

    def _prepare(self):
        self._gabor_filters_shape = copy.deepcopy(self._gabor_filters[0].shape)

    def run(self, image_path, learn=False):
        image = self._read_and_validate_image(image_path)

        # filter image by given gabor filters
        gabor_layer_maps = [
            cv2.filter2D(image, cv2.CV_8UC3, kernel) for kernel in self._gabor_filters
        ]

        # perform sub-sampling (pooling)
        gabor_pooling_layer_maps = [
            block_reduce(
                filtered_image,
                self.gabor_pooling_layer_kernel_shape.get_tuple(),
                np.max
            ) for filtered_image in gabor_layer_maps
        ]

    def _read_and_validate_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise RuntimeError("failed to load image")

        if image.shape != (self._image_shape.rows, self._image_shape.columns):
            raise RuntimeError("image size is not appropriate")

        return image
