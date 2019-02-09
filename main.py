from myneuralnetwork import MNNBuilder, Shape, GaborParameters
import numpy as np
import random
import math
from os import path, listdir
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn import preprocessing
from progress.bar import Bar

gabor_filters = []
for theta in np.arange(0, np.pi, np.pi / 8):
    parameters = GaborParameters(Shape(31, 31), 4.0, theta, 20, 0.5, 0)
    gabor_filters.append(parameters)


def random_generator():
    return random.uniform(0.000, 0.1)


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

source_dir = "./resources/refined"
first_category = "car_side"
second_category = "stop_sign"

first_category_parent_dir = path.join(source_dir, first_category)
second_category_parent_dir = path.join(source_dir, second_category)

number_of_image_repeats = 3

print("training MNN on {}: ".format(first_category))
# run on first category
for file_name in Bar('Processing').iter(listdir(first_category_parent_dir)):

    # run on every category image for `number_of_image_repeats` times
    for repeat in range(number_of_image_repeats):
        file_path = path.join(first_category_parent_dir, file_name)
        mnn.run(file_path, True)

print("training MNN on {}:".format(second_category))
# run on second category
for file_name in Bar('Processing').iter(listdir(second_category_parent_dir)):

    # run on every category image for `number_of_image_repeats` times
    for repeat in range(number_of_image_repeats):
        file_path = path.join(second_category_parent_dir, file_name)
        mnn.run(file_path, True)

data_profile = []
label_profile = []

print("running MNN on {}:".format(first_category))
# run on first category
for file_name in Bar('Processing').iter(listdir(first_category_parent_dir)):
    file_path = path.join(first_category_parent_dir, file_name)
    result = mnn.run(file_path, False)

    data_profile.append(result)

    # `0` indicates first category
    label_profile.append(0)

print("running MNN on {}:".format(second_category))
# run on second category
for file_name in Bar('Processing').iter(listdir(second_category_parent_dir)):
    file_path = path.join(second_category_parent_dir, file_name)
    result = mnn.run(file_path, False)

    data_profile.append(result)

    # `1` indicates second category
    label_profile.append(1)

print("training SVM and compute accuracy by Cross-Validation...")

# scale data
data_profile = preprocessing.scale(data_profile)

# # normalize data
# data_profile = preprocessing.normalize(data_profile, "l2")

# initialize SVM
clf = SVC(gamma='auto')

# perform Cross-Validation and compute accuracy
cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
scores = cross_val_score(clf, data_profile, label_profile, cv=cv)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
