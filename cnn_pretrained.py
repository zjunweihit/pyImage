from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import MobileNet

from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input

from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

import numpy as np
import argparse
import cv2

# Models for image classification with weights trained on ImageNet:
# https://keras.io/applications/
#
#     Xception
#     VGG16
#     VGG19
#     ResNet, ResNetV2, ResNeXt
#     InceptionV3
#     InceptionResNetV2
#     MobileNet
#     MobileNetV2
#     DenseNet
#     NASNet

MODELS = {
    "vgg16": VGG16,
    "vgg19": VGG19,
    "resnet": ResNet50,
    "inception": InceptionV3,
    "xception": Xception,  # Tensorflow only
    "mobilenet": MobileNet,
}

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the input image")
ap.add_argument("-m", "--model", type=str, default="all",
                help="name of pre-trained network to use."
                     " e.g. one of {}".format(MODELS.keys()))
args = vars(ap.parse_args())

def classify_image(imageFile, modelType=args["model"]):
    if modelType not in MODELS.keys():
        raise AssertionError("The --model argument '{}' is not support!".format(args["model"]))

    # =======================================================================
    # Image Size:
    #
    # typically the image size to a CNN on ImageNet are
    #   224x224, 227x227, 256x256, 299x299
    #
    # VGG16, VGG19, ResNet accept 224x224
    # Inception V3 and Xception require 229x229
    # =======================================================================
    inputShape = (224, 224)
    preprocess = imagenet_utils.preprocess_input

    # For inception and xception, use a different image size and preprocess
    if modelType in ("inception", "xception"):
        inputShape = (299, 299)
        preprocess = preprocess_input

    print("[INFO] loading {}...".format(modelType))
    Network = MODELS[modelType]
    model = Network(weights="imagenet")

    print("[INFO] loading and pre-processing the image...")
    image = load_img(imageFile, target_size=inputShape)
    # now it will be a NumPy array (inputShape[0], inputShape[1], 3)
    image = img_to_array(image)
    # format the input image as (1, inputShape[0], inputShape[1], 3) for pre-trained network
    image = np.expand_dims(image, axis=0)
    # preprocess the image for the cnn
    image = preprocess(image)

    print("[INFO] classifying the image with '{}'...".format(modelType))
    preds = model.predict(image)
    P = imagenet_utils.decode_predictions(preds)

    for (i, (imagenetID, label, prob)) in enumerate(P[0]):
        print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))

    return P[0][0]


def putText(img, type, lable, prob, i):
    cv2.putText(img,                                                # image
                "{}: {}, {:.2f}%".format(type, label, prob * 100),  # Text
                (10, 20 * i),                                       # lefttop position
                cv2.FONT_HERSHEY_SIMPLEX,                           # font
                0.6,                                                # fontsize
                (0, 255, 0),                                        # color (B, G, R)
                2)                                                  # thickness (int only)


if args["model"] == "all":
    # compare the results of all types
    type_list = MODELS.keys()
else:
    # process the image with specified model
    type_list = [args["model"]]

i = 1
img = cv2.imread(args["image"])
for type in type_list:
    #(imagenetID, label, prob) = (1, "abc", 0.55)
    (imagenetID, label, prob)  = classify_image(args["image"], type)
    putText(img, type, label, prob, i)
    i = i + 1

cv2.imshow("Classification", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
