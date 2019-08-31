from skimage.exposure import rescale_intensity
import numpy as np
import cv2

class Convolve:
    def __init__(self, imageFile, kernel):
        '''
        :param imageFile: the image file will be read by opencv and converted to gray
        :param kernel: the kernel to apply the input image
        '''
        # the average of S, 1/S, S is the total number of enties in the matrix
        self.smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
        self.largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))

        self.sharpen = np.array(
            ([0, -1, 0],
             [-1, 5, -1],
             [0, -1, 0]), dtype="int")

        self.laplaceian = np.array(
            ([0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]), dtype="int")

        self.sobelX = np.array(
            ([-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]), dtype="int")

        self.sobelY = np.array(
            ([-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]), dtype="int")

        self.emboss = np.array(
            ([-2, -1, 0],
             [-1, 1, 1],
             [0, 1, 2]), dtype="int")
        self.kernelList = {
            "small_blur" : self.smallBlur,
            "large_blur" : self.largeBlur,
            "sharpen"    : self.sharpen,
            "laplaceian" : self.laplaceian,
            "sobel_x"    : self.sobelX,
            "sobel_y"    : self.sobelY,
            "emboss"     : self.emboss}

        if kernel not in self.kernelList.keys():
            raise ValueError("Not support the kernel: {}".format(kernel))

        self.kernel = kernel
        self.image = cv2.imread(imageFile)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        return

    def convolve(self):
        self.convolve_gray(self.image, self.kernel)


    @staticmethod
    def convolve_gray(image, kernel):
        (iH, iW) = image.shape[:2]
        (kH, kW) = kernel.shape[:2]

        # add padding to the image to output the same size as original image's
        pad = (kW - 1) // 2
        image_pad = cv2.copyMakeBorder(image, pad, pad, pad, pad,
                                   cv2.BORDER_REPLICATE)
        output = np.zeros((iH, iW), dtype="float")

        # loop over the original image
        # (pad, pad) is the (0,0) in the original image
        for y in np.arange(pad, iH + pad):
            for x in np.arange(pad, iW + pad):
                # get the ROI of the image
                roi = image_pad[y - pad:y + pad + 1, x - pad:x + pad + 1]

                # convolve roi with the kernel
                k_conv = (roi * kernel).sum()

                # store the output results
                # output has the same size as original image's
                # (0,0) is the lefttop
                output[y - pad, x - pad] = k_conv

        # rescale the output image in range [0, 255]
        output = rescale_intensity(output, in_range=(0, 255))
        output = (output * 255).astype("uint8")

        return output

    @staticmethod
    def get_kernel(name):
        smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
        largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))

        sharpen = np.array(
            ([0, -1, 0],
             [-1, 5, -1],
             [0, -1, 0]), dtype="int")

        laplaceian = np.array(
            ([0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]), dtype="int")

        sobelX = np.array(
            ([-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]), dtype="int")

        sobelY = np.array(
            ([-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]), dtype="int")

        emboss = np.array(
            ([-2, -1, 0],
             [-1, 1, 1],
             [0, 1, 2]), dtype="int")
        kernelList = {
            "small_blur" : smallBlur,
            "large_blur" : largeBlur,
            "sharpen"    : sharpen,
            "laplaceian" : laplaceian,
            "sobel_x"    : sobelX,
            "sobel_y"    : sobelY,
            "emboss"     : emboss}
        if name not in kernelList.keys():
            raise ValueError("Not support the kernel: {}".format(name))
        return kernelList[name]



