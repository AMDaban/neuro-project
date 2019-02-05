import copy
import cv2
import numpy as np
from skimage.measure import block_reduce
from .utility_functions import extract_spike_profile
from .utility_models import IF


# MNN stands for 'My Neural Network'
# It is assumed that all parameters have been checked
# so please use MNNBuilder instead of creating object of this class directly
class MNN:
    def __init__(self, image_shape, gabor_filters, gabor_pooling_layer_kernel_shape, complex_layer_map_count,
                 complex_layer_map_shape, complex_layer_kernel_shape, neurons_threshold, random_generator,
                 stdp_function):
        self._image_shape = image_shape
        self._gabor_pooling_layer_kernel_shape = gabor_pooling_layer_kernel_shape
        self._complex_layer_map_count = complex_layer_map_count
        self._complex_layer_map_shape = complex_layer_map_shape
        self._complex_layer_kernel_shape = complex_layer_kernel_shape
        self._neurons_threshold = neurons_threshold
        self._random_generator = random_generator
        self._gabor_filters = gabor_filters
        self._stdp_function = stdp_function

        self._gabor_filters_shape = None
        self._complex_layer_kernels = None
        self._complex_layer_neuron_maps = None

        self._prepare()

    def _prepare(self):
        self._gabor_filters_shape = copy.deepcopy(self._gabor_filters[0].shape)

        # prepare complex layer kernels
        self._complex_layer_kernels = []
        for i in range(self._complex_layer_map_count):
            # calculate number of elements exist in one complex layer kernel
            kernel_element_count = len(self._gabor_filters)
            kernel_element_count *= self._complex_layer_kernel_shape.rows
            kernel_element_count *= self._complex_layer_kernel_shape.columns

            # generate random values
            # noinspection PyUnusedLocal
            random_values = [self._random_generator() for j in range(kernel_element_count)]

            # build and store kernels
            self._complex_layer_kernels.append(
                np.array(random_values).reshape(
                    (
                        len(self._gabor_filters),
                        self._complex_layer_kernel_shape.rows,
                        self._complex_layer_kernel_shape.columns
                    )
                )
            )

        # prepare complex layer neuron maps
        self._complex_layer_neuron_maps = []
        for i in range(self._complex_layer_map_count):
            map_height, map_width = self._complex_layer_map_shape.get_tuple()

            # noinspection PyUnusedLocal
            neurons = [IF(self._neurons_threshold) for j in range(map_height * map_width)]

            self._complex_layer_neuron_maps.append(
                np.array(neurons).reshape(map_height, map_width)
            )

    def run(self, image_path, learn=False):
        print("run on {}, learn: {}".format(image_path, learn))

        image = self._read_and_validate_image(image_path)

        # filter image by given gabor filters
        gabor_layer_maps = [
            cv2.filter2D(image, cv2.CV_8UC3, kernel) for kernel in self._gabor_filters
        ]

        # perform sub-sampling (pooling)
        gabor_pooling_layer_maps = [
            block_reduce(
                filtered_image,
                self._gabor_pooling_layer_kernel_shape.get_tuple(),
                np.max
            ) for filtered_image in gabor_layer_maps
        ]

        # extract spike profile
        spike_profile = extract_spike_profile(gabor_pooling_layer_maps)

        # apply spikes and get first complex layer spike
        first_spike, complex_layer_spike_profile = self._apply_spikes(spike_profile)

        if learn and first_spike is not None:
            self._apply_stdp(first_spike, spike_profile)

        self._reset_neurons()

        return complex_layer_spike_profile

    def _read_and_validate_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise RuntimeError("failed to load image")

        if image.shape != (self._image_shape.rows, self._image_shape.columns):
            raise RuntimeError("image size is not appropriate")

        return image

    def _apply_spikes(self, spike_profile):
        first_spike = None

        # complex_layer_spike_profile shows complex layer maps first spike time (relative to each other)
        # noinspection PyUnusedLocal
        complex_layer_spike_profile = [-1 for k in range(self._complex_layer_map_count)]

        # a simple clock that mimics the time
        clock = 0

        kernel_height, kernel_width = self._complex_layer_kernel_shape.get_tuple()
        for map_index, i, j, map_value in spike_profile:
            for n in range(kernel_height):
                for m in range(kernel_width):
                    mapped_x = i - n
                    mapped_y = j - m

                    if mapped_x < 0 or mapped_x >= self._complex_layer_map_shape.rows:
                        continue
                    if mapped_y < 0 or mapped_y >= self._complex_layer_map_shape.columns:
                        continue

                    for complex_map_index in range(self._complex_layer_map_count):
                        # calculate neuron potential change
                        potential_change = self._complex_layer_kernels[complex_map_index][map_index][n][m]

                        # extract target neuron
                        target_neuron = self._complex_layer_neuron_maps[complex_map_index][mapped_x][mapped_y]

                        # apply spike effect on target neuron
                        target_neuron.change_potential(potential_change)

                        # check if target neuron sent a spike
                        if target_neuron.hit_threshold():

                            # set first spike time in this layer
                            if complex_layer_spike_profile[complex_map_index] == -1:
                                complex_layer_spike_profile[complex_map_index] = clock

                            # set first spike time in all layers
                            if first_spike is None:
                                first_spike = (complex_map_index, mapped_x, mapped_y, clock)

            clock += 1

        return first_spike, complex_layer_spike_profile

    def _apply_stdp(self, first_spike, spike_profile):
        kernel_height, kernel_width = self._complex_layer_kernel_shape.get_tuple()

        # function to filter spike_profile with
        def check_spike(spike):
            if spike[1] < first_spike[1] or spike[1] >= first_spike[1] + kernel_height:
                return False

            if spike[2] < first_spike[2] or spike[2] >= first_spike[2] + kernel_width:
                return False

            return True

        # iterate over related spikes and apply stdp
        for time, single_spike in enumerate(spike_profile):
            if check_spike(single_spike):
                weight_change = self._stdp_function(first_spike[3] - time)
                target_kernel = self._complex_layer_kernels[first_spike[0]]
                target_weight_x = single_spike[1] - first_spike[1]
                target_weight_y = single_spike[2] - first_spike[2]
                target_kernel[single_spike[0]][target_weight_x][target_weight_y] += weight_change

    def print_complex_layer_kernels(self):
        print("MNN complex layer current kernels:")
        for index, kernel in enumerate(self._complex_layer_kernels):
            print("\tcomplex feature {}:".format(index))
            for kernel_map_index, kernel_map in enumerate(kernel):
                print("\t\tkernel-map {}:".format(kernel_map_index))
                for i in range(len(kernel_map)):
                    res = "\t\t\t"
                    for j in range(len(kernel_map[i])):
                        res += str(kernel_map[i][j]).ljust(20)
                        res += "\t\t"
                    print(res)

    def _reset_neurons(self):
        for neuron_map in self._complex_layer_neuron_maps:
            for row in neuron_map:
                for neuron in row:
                    neuron.reset()
