# Load all image datasets from disk
# the data path format is assumed as
#   /path/to/datasets/{class}/{image}.jpg

import numpy as np
import cv2
import os

class DatasetLoader:
    def __init__(self, preprocessors=None):
        '''
        Initialize with a optional preprocessor list
        :param preprocessors: a list
        '''
        self.preprocessors = preprocessors

        if self.preprocessors == None:
           self.preprocessors = []

    def load(self, imagePathList, verbose=-1):
        data = []
        labels = []

        for (i, imagePath) in enumerate(imagePathList):
            # data path format:
            #   /path/to/dataset/{class}/{image}.jpg
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]

            if self.preprocessors != None:
                for p in self.preprocessors:
                    image = p.preprocess(image)

            data.append(image)
            labels.append(label)

            # show info every verbose images
            if verbose > 0 and i > 0 and (i+1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i+1, len(imagePathList)))

        # convert the list(where item is a nparray) to a ndarray
        # and return the tuple of ndarray of data and labels
        return (np.array(data), np.array(labels))
