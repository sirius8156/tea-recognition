# coding=utf-8

import cv2


def cropping_square(img_path, x_start, y_start, length):
    ori_img = cv2.imread(img_path)
    cropped = ori_img[x_start:x_start + length, y_start:y_start + length]
    return cropped


if __name__ == '__main__':
    jpg = "/Users/philip.du/Documents/Projects/research/tea-recognition/sample_1/1a.JPG"

    square = cropping_square(jpg, 1500, 800, 300)
    cv2.imwrite("cropped_1.jpg", square)

    image = cv2.imread("cropped_1.jpg")
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    m = cv2.getRotationMatrix2D(center, 45, 0.75)
    rotated = cv2.warpAffine(image, m, (w, h))
    cv2.imwrite("cropped_2.jpg", rotated)
