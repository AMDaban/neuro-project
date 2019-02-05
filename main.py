from myneuralnetwork import MNNBuilder, Shape, GaborParameters
import numpy as np
import random
import math
from os import path, listdir
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

gabor_filters = []
for theta in np.arange(0, np.pi, np.pi / 8):
    parameters = GaborParameters(Shape(31, 31), 4.0, theta, 20, 0.5, 0)
    gabor_filters.append(parameters)


def random_generator():
    return random.uniform(-0.001, 0.009)


# time_diff = t_post - t_pre
def stdp_weight_change(time_diff):
    positive_learning_rate = 1 / 50
    negative_learning_rate = 1 / 50

    if time_diff >= 0:
        return positive_learning_rate * math.exp(-1 * time_diff / 10000)
    else:
        return -1 * negative_learning_rate * math.exp(-1 * time_diff / 10000)


builder = MNNBuilder()
builder.set_image_shape(Shape(200, 200)) \
    .set_gabor_layer_filters(gabor_filters) \
    .set_gabor_pooling_layer_kernel_shape(Shape(10, 10)) \
    .set_neurons_threshold(1) \
    .set_complex_layer_map_count(5) \
    .set_complex_layer_kernel_shape(Shape(7, 7)) \
    .set_random_generator(random_generator) \
    .set_stdp_function(stdp_weight_change)

mnn = builder.build()

# output = mnn.run("./resources/refined/car_side/image_0001.jpg", False)
# print(output)

source_dir = "./resources/refined"
first_category = "car_side"
second_category = "soccer_ball"

first_category_parent_dir = path.join(source_dir, first_category)
second_category_parent_dir = path.join(source_dir, second_category)

number_of_image_repeats = 1

# run on first category
for file_name in listdir(first_category_parent_dir):

    # run on every category image for `number_of_image_repeats` times
    for repeat in range(number_of_image_repeats):
        file_path = path.join(first_category_parent_dir, file_name)
        result = mnn.run(file_path, True)

# run on second category
for file_name in listdir(second_category_parent_dir):

    # run on every category image for `number_of_image_repeats` times
    for repeat in range(number_of_image_repeats):
        file_path = path.join(second_category_parent_dir, file_name)
        result = mnn.run(file_path, True)

test_source_dir = "./resources/refined"
first_test_category = "car_side"
second_test_category = "soccer_ball"

first_test_category_parent_dir = path.join(test_source_dir, first_test_category)
second_test_category_parent_dir = path.join(test_source_dir, second_test_category)

label_profile = []
data_profile = []

# run on first test category
for file_name in listdir(first_test_category_parent_dir):
    file_path = path.join(first_test_category_parent_dir, file_name)
    result = mnn.run(file_path, False)

    # 0 indicates first category
    label_profile.append(0)
    data_profile.append(result)

# run on second test category
for file_name in listdir(second_test_category_parent_dir):
    file_path = path.join(second_test_category_parent_dir, file_name)
    result = mnn.run(file_path, False)

    # 1 indicates second category
    label_profile.append(1)
    data_profile.append(result)

# initialize classifier
clf = SVC(gamma='auto')

# train classifier
clf.fit(data_profile, label_profile)

# predict every mnn out put
estimation_profile = clf.predict(data_profile)

# compute classifier accuracy
accuracy = accuracy_score(label_profile, estimation_profile)

print(accuracy)
