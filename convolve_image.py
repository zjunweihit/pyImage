from pyimage.nn.convolve import Convolve
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the input image")
args = vars(ap.parse_args())

Kernels = [
    "small_blur",
    "large_blur",
    "sharpen",
    "laplaceian",
    "sobel_x",
    "sobel_y",
    "emboss"]

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

for kernel in Kernels:
    print("[INFO] applying {} kernel".format(kernel))
    K = Convolve.get_kernel(kernel)
    convOutput = Convolve.convolve_gray(gray, K)
    opencvOutput = cv2.filter2D(gray, -1, K)

    cv2.imshow("Original", gray)
    cv2.imshow("{} - convolve".format(kernel), convOutput)
    cv2.imshow("{} - opencv".format(kernel), opencvOutput)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
