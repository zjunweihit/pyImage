# preprocess the images for NN, e.g. resize to a fixed size, ignoring aspect ratio.
#
# Note:
#     ML algorithms, such as kNN, SVM and CNN require images to have identical widths and heights.

import cv2

class Preprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        '''
        resize to a fixed size, ignoring the aspect
        :param image: original image
        :return: resized image
        '''
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)
