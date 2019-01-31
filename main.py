from myneuralnetwork import MNNBuilder, Shape, GaborParameters
import numpy as np

gabor_filters = []
for theta in np.arange(0, np.pi, np.pi / 16):
    parameters = GaborParameters(Shape(31, 31), 4.0, theta, 20, 0.5, 0)
    gabor_filters.append(parameters)

builder = MNNBuilder()
builder.set_image_shape(Shape(640, 640)) \
    .set_gabor_layer_filters(gabor_filters) \
    .set_gabor_pooling_layer_kernel_shape(Shape(10, 10))

mnn = builder.build()

mnn.run("./resources/images/profile.jpg", False)
