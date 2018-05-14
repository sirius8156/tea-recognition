# coding=utf-8

import cv2
import numpy as np
from matplotlib import pyplot as plt


def get_gray_img(img_path):
    ori_img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2GRAY)
    return gray_img


def get_sobel_img(img_path):
    ori_img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2GRAY)

    # Sobel函数求完导数后会有负值，还有会大于255的值。
    # 而原图像是uint8，即8位无符号数，所以Sobel建立的图像位数不够，会有截断。
    # 因此要使用16位有符号的数据类型，即cv2.CV_16S
    x = cv2.Sobel(gray_img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(gray_img, cv2.CV_16S, 0, 1)

    # 转回uint8
    abs_x = cv2.convertScaleAbs(x)
    abs_y = cv2.convertScaleAbs(y)

    sobel_img = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
    return sobel_img


def get_laplacian_img(img_path):
    ori_img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2GRAY)

    # laplacian_img = cv2.Laplacian(gray_img, cv2.CV_16S)
    laplacian_img = cv2.Laplacian(gray_img, cv2.CV_16S, 5)

    return laplacian_img


def get_canny_img(img_path):
    ori_img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2GRAY)

    # 先进行高斯滤波降噪。
    blur_img = cv2.GaussianBlur(gray_img, (3, 3), 0)

    # 在进行抠取轮廓，其中apertureSize默认为3。
    return cv2.Canny(blur_img, 50, 150, L2gradient=1)


def get_fast_fourier_img(img_path):
    ori_img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray_img)
    f_shift = np.fft.fftshift(f)
    fft_img = np.log(np.abs(f_shift))
    return fft_img

def pretty_show_img(img):
    weight = img.shape[0]
    height = img.shape[1]

    show_able_img = img

    # 下采样只到分辨率接近1000*1000
    while weight >= 800 or height >= 800:
        show_able_img = cv2.pyrDown(show_able_img)
        weight = show_able_img.shape[0]
        height = show_able_img.shape[1]

    return show_able_img


if __name__ == '__main__':
    jpg = "/Users/philip.du/Documents/Projects/research/tea-recognition/sample_1/1a.JPG"
    ori_image = cv2.imread(jpg)
    gray_image = get_gray_img(jpg)
    sobel_img = get_sobel_img(jpg)
    laplacian_img = get_laplacian_img(jpg)
    canny_img = get_canny_img(jpg)
    fft_img = get_fast_fourier_img(jpg)

    plt.subplot(211), plt.imshow(ori_image, 'ori')
    plt.subplot(212), plt.imshow(gray_image, 'gray')
    plt.subplot(221), plt.imshow(sobel_img, 'sobel')
    plt.subplot(222), plt.imshow(canny_img, 'canny')

    # cv2.imshow('ori_image', pretty_show_img(ori_image))
    # cv2.imshow('gray_image', pretty_show_img(gray_image))
    # cv2.imshow('sobel_img', pretty_show_img(sobel_img))
    # cv2.imshow('laplacian_img', pretty_show_img(laplacian_img))
    # cv2.imshow('canny_img', pretty_show_img(canny_img))
    # cv2.imshow('fft_img', pretty_show_img(fft_img))

    # cv2.waitKey(0)

    cv2.imwrite("gray_image.jpg", gray_image)
    cv2.imwrite("sobel_img.jpg", sobel_img)
    cv2.imwrite("canny_img.jpg", canny_img)
