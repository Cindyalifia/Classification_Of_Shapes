import argparse
import numpy as np
from keras.models import load_model
from keras.preprocessing import image


def arg_parse():
    """
    Parse arguements to the detect module
    """

    parser = argparse.ArgumentParser(description='Shape detector')

    parser.add_argument(
        "--images", 
        dest='images', 
        required=True,
        help="Image / Directory containing images to perform detection upon", 
        type=str
    )

    return parser.parse_args()


def class_result(list_predict):
    """
    Function to define class of shapes
    """
    list_predict = list_predict.tolist()
    if list_predict.index(max(list_predict)) == 0:
        return 'circles'
    elif list_predict.index(max(list_predict)) == 1:
        return 'squares'
    elif list_predict.index(max(list_predict)) == 2:
        return 'triangles'


def predict_class(file_name):
    """
    Function to predict new image
    """
    test_image = image.load_img(file_name, target_size=(28, 28))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)
    print('Predicted : ', class_result(result[0]))


args = arg_parse()

# Load model
classifier = load_model('my_model.h5')

# Classify
predict_class(args.images)
