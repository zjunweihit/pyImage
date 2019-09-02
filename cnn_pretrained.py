from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception
from keras.applications import VGG16
from keras.applications import VGG19

from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input

from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt

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

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the input image")
ap.add_argument("-m", "--model", type=str, default="all",
                help="name of pre-trained network to use."
                     " e.g. resnet, vgg16, vgg19, xception, inception, all")
args = vars(ap.parse_args())

MODELS = {
    "vgg16": VGG16,
    "vgg19": VGG19,
    "resnet": ResNet50,
    "inception": InceptionV3,
    "xception": Xception,  # Tensorflow only
}

def classify_image(imageFile, modelType=args["model"], getImage=False):
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

    org = cv2.imread(imageFile)
    (imagenetID, label, prob) = P[0][0]
    cv2.putText(org, "[{}] Label: {}".format(modelType, label), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    if getImage == True:
        return org
    else:
        cv2.imshow("Classification", org)
        cv2.waitKey(0)

if args["model"] != "all":
    # general process
    classify_image(args["image"])
else:
    # compare the results of all types
    row, col = 2, 3

    # debug
    #img = classify_image(args["image"], "vgg16", getImage=True)
    #image_list.append(("vgg16", img))
    #img = classify_image(args["image"], "vgg19", getImage=True)
    #image_list.append(("vgg19", img))

    i = 1
    for type in MODELS.keys():
        img = classify_image(args["image"], type, getImage=True)

        plt.subplot(row, col, i)
        plt.title(type)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # remove x, y axis ticks
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img)
        i = i+1

    plt.show()
