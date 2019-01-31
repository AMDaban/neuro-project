from .mnn import MNN
from .utility_models import Shape, GaborParameters
import cv2
import copy


class MNNBuilder:
    def __init__(self):
        self._image_shape = None
        self._gabor_filters = None

    def set_image_shape(self, image_shape):
        self._image_shape = image_shape
        return self

    def set_gabor_layer_filters(self, gabor_filters):
        self._gabor_filters = gabor_filters
        return self

    def build(self):
        self._validate()

        return MNN(
            image_shape=copy.deepcopy(self._image_shape),
            gabor_filters=self._create_gabor_filters()
        )

    def _create_gabor_filters(self):
        filters = []
        for parameters in self._gabor_filters:
            kernel_height = parameters.shape.rows
            kernel_width = parameters.shape.columns

            kernel = cv2.getGaborKernel(
                (kernel_height, kernel_width), parameters.sigma, parameters.theta,
                parameters.lmd, parameters.gamma, ktype=cv2.CV_32F
            )

            filters.append(kernel)

        return filters

    def _validate(self):
        self._check_types()

        unique_gabor_heights = set(map(lambda x: x.shape.rows, self._gabor_filters))
        if len(unique_gabor_heights) > 1:
            raise RuntimeError("uncoordinated gabor filters")

        unique_gabor_widths = set(map(lambda x: x.shape.columns, self._gabor_filters))
        if len(unique_gabor_widths) > 1:
            raise RuntimeError("uncoordinated gabor filters")

        gabor_width = list(unique_gabor_widths)[0]
        gabor_height = list(unique_gabor_heights)[0]
        if gabor_height > self._image_shape.rows or gabor_width > self._image_shape.columns:
            raise RuntimeError("too big gabor filters")

    def _check_types(self):
        if type(self._image_shape) is not Shape:
            raise RuntimeError("image_shape must be an instance of Shape")

        if type(self._gabor_filters) is not list:
            raise RuntimeError("gabor_filters must be a list")

        for gabor_parameters in self._gabor_filters:
            if type(gabor_parameters) is not GaborParameters:
                raise RuntimeError("gabor_filters must be a list of GaborParameters")

            if type(gabor_parameters.shape) is not Shape:
                raise RuntimeError("GaborParameters.shape must be an instance of Shape")
