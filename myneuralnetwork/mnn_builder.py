from .mnn import MNN
from .utility_models import Shape, GaborParameters
import cv2
import copy


class MNNBuilder:
    def __init__(self):
        self._image_shape = None
        self._gabor_filters = None
        self._gabor_pooling_kernel_shape = None
        self._neurons_threshold = None
        self._complex_layer_map_count = None
        self._complex_layer_kernel_shape = None
        self._random_generator = None

    def set_image_shape(self, image_shape):
        self._image_shape = image_shape
        return self

    def set_gabor_layer_filters(self, gabor_filters):
        self._gabor_filters = gabor_filters
        return self

    def set_gabor_pooling_layer_kernel_shape(self, kernel_shape):
        self._gabor_pooling_kernel_shape = kernel_shape
        return self

    def set_neurons_threshold(self, threshold):
        self._neurons_threshold = threshold
        return self

    def set_complex_layer_map_count(self, count):
        self._complex_layer_map_count = count
        return self

    def set_complex_layer_kernel_shape(self, shape):
        self._complex_layer_kernel_shape = shape
        return self

    def set_random_generator(self, generator):
        self._random_generator = generator
        return self

    def build(self):
        self._validate()

        gabor_pooling_layer_maps_shape = self._get_gabor_pooling_layer_maps_shape()

        return MNN(
            image_shape=copy.deepcopy(self._image_shape),
            gabor_filters=self._create_gabor_filters(),
            gabor_pooling_layer_kernel_shape=self._gabor_pooling_kernel_shape,
            complex_layer_map_count=self._complex_layer_map_count,
            complex_layer_map_shape=Shape(
                rows=gabor_pooling_layer_maps_shape.rows - self._complex_layer_kernel_shape.rows + 1,
                columns=gabor_pooling_layer_maps_shape.columns - self._complex_layer_kernel_shape.columns + 1,
            ),
            complex_layer_kernel_shape=self._complex_layer_kernel_shape,
            neurons_threshold=self._neurons_threshold,
            random_generator=self._random_generator
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
        if self._image_shape is None:
            raise RuntimeError("image_shape is not set")

        if self._gabor_filters is None:
            raise RuntimeError("gabor_layer_filters is not set")

        if self._gabor_pooling_kernel_shape is None:
            raise RuntimeError("gabor_pooling_layer_kernel_shape is not set")

        if self._neurons_threshold is None:
            raise RuntimeError("neurons_threshold is not set")

        if self._complex_layer_map_count is None:
            raise RuntimeError("complex_layer_map_count is not set")

        if self._complex_layer_kernel_shape is None:
            raise RuntimeError("complex_layer_kernel_shape is not set")

        if self._random_generator is None:
            raise RuntimeError("random_generator is not set")

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
            raise RuntimeError("too wide gabor filters")

        if self._image_shape.rows < self._gabor_pooling_kernel_shape.rows:
            raise RuntimeError("gabor_pooling_layer_kernel_shape height is too long")

        if self._image_shape.rows % self._gabor_pooling_kernel_shape.rows != 0:
            raise RuntimeError("image_shape height is not divisible by gabor_pooling_layer_kernel_shape height")

        if self._image_shape.columns < self._gabor_pooling_kernel_shape.columns:
            raise RuntimeError("gabor_pooling_layer_kernel_shape width is too long")

        if self._image_shape.columns % self._gabor_pooling_kernel_shape.columns != 0:
            raise RuntimeError("image_shape width is not divisible by gabor_pooling_layer_kernel_shape width")

        if self._complex_layer_kernel_shape.rows % 2 == 0 or self._complex_layer_kernel_shape.columns % 2 == 0:
            raise RuntimeError("complex_layer_kernel_shape width and height must be odd")

        gabor_pooling_layer_maps_height = self._image_shape.rows / self._gabor_pooling_kernel_shape.rows
        if self._complex_layer_kernel_shape.rows > gabor_pooling_layer_maps_height:
            raise RuntimeError("complex_layer_kernel_shape height is too long")

        gabor_pooling_layer_maps_width = self._image_shape.columns / self._gabor_pooling_kernel_shape.columns
        if self._complex_layer_kernel_shape.columns > gabor_pooling_layer_maps_width:
            raise RuntimeError("complex_layer_kernel_shape width is too long")

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

        if type(self._gabor_pooling_kernel_shape) is not Shape:
            raise RuntimeError("gabor_pooling_layer_kernel_shape must be an instance of Shape")

        if type(self._complex_layer_kernel_shape) is not Shape:
            raise RuntimeError("complex_layer_kernel_shape must be an instance of Shape")

    def _get_gabor_pooling_layer_maps_shape(self):
        return Shape(
            rows=int(self._image_shape.rows / self._gabor_pooling_kernel_shape.rows),
            columns=int(self._image_shape.columns / self._gabor_pooling_kernel_shape.columns)
        )
