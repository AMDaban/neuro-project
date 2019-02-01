from myneuralnetwork import MNNBuilder, Shape, GaborParameters
import numpy as np
import random
import math

gabor_filters = []
for theta in np.arange(0, np.pi, np.pi / 16):
    parameters = GaborParameters(Shape(31, 31), 4.0, theta, 20, 0.5, 0)
    gabor_filters.append(parameters)


def random_generator():
    return random.uniform(-0.01, 0.02)


# time_diff = t_post - t_pre
def stdp_weight_change(time_diff):
    positive_learning_rate = 1 / 50
    negative_learning_rate = 1 / 50

    if time_diff >= 0:
        return positive_learning_rate * math.exp(-1 * time_diff / 10000)
    else:
        return -1 * negative_learning_rate * math.exp(-1 * time_diff / 10000)


builder = MNNBuilder()
builder.set_image_shape(Shape(170, 170)) \
    .set_gabor_layer_filters(gabor_filters) \
    .set_gabor_pooling_layer_kernel_shape(Shape(10, 10)) \
    .set_neurons_threshold(1) \
    .set_complex_layer_map_count(10) \
    .set_complex_layer_kernel_shape(Shape(11, 11)) \
    .set_random_generator(random_generator)\
    .set_stdp_function(stdp_weight_change)

mnn = builder.build()

output = mnn.run("./resources/images/photo.jpg", True)
print(output)
