import cv2
from os import path, mkdir, listdir


def image_resize(old_path, new_height, new_width, new_path):
    # read image
    image = cv2.imread(old_path)

    # resize image
    new_image = cv2.resize(image, (new_width, new_height))

    # create parent directory if not exists
    parent_dir = path.abspath(path.dirname(new_path))
    if not path.exists(parent_dir):
        mkdir(parent_dir)

    # save image
    cv2.imwrite(new_path, new_image)


def resize_category(source_dir, category, dest_dir, new_height, new_width):
    parent_dir = path.join(source_dir, category)
    for file_name in listdir(parent_dir):
        old_path = path.join(parent_dir, file_name)
        new_path = path.join(path.join(dest_dir, category), file_name)
        image_resize(old_path, new_height, new_width, new_path)
