# coding=utf-8

import cv2

import tools.feature_extract as ife


def orb_img(img, features_count):
    orb = cv2.ORB_create(features_count)
    orb_key_points, orb_desc = orb.detectAndCompute(img, None)
    orb_signed_img = cv2.drawKeypoints(img, orb_key_points, None)

    return orb_key_points, orb_desc, orb_signed_img


def sift_img(img):
    sift = cv2.xfeatures2d.SIFT_create()
    sift_key_points, sift_desc = sift.detectAndCompute(img, None)
    sift_signed_img = cv2.drawKeypoints(img, sift_key_points, None)

    return sift_key_points, sift_desc, sift_signed_img


def surf_img(img):
    surf = cv2.xfeatures2d.SURF_create()
    surf_key_points, surf_desc = surf.detectAndCompute(img, None)
    surf_signed_img = cv2.drawKeypoints(img, surf_key_points, None)

    return surf_key_points, surf_desc, surf_signed_img


if __name__ == '__main__':
    jpg = "/Users/philip.du/Documents/Projects/research/tea-recognition/sample_1/1a.JPG"
    # jpg = "/Users/philip.du/Downloads/image1.JPG"

    ## ----  orb --- ##

    # gray_img = ife.get_gray_img(jpg)
    # key_points, desc, signed_img = orb_img(gray_img, 5000)
    # print("gray# kps: {}, descriptors: {}".format(len(key_points), desc.shape))
    # cv2.imwrite("gray_img.signed.jpg", signed_img)
    #
    # sobel_img = ife.get_sobel_img(jpg)
    # key_points, desc, signed_img = orb_img(sobel_img, 5000)
    # print("sobel# kps: {}, descriptors: {}".format(len(key_points), desc.shape))
    # cv2.imwrite("sobel_img.signed.jpg", signed_img)
    #
    # canny_img = ife.get_canny_img(jpg)
    # key_points, desc, signed_img = orb_img(canny_img, 5000)
    # print("canny# kps: {}, descriptors: {}".format(len(key_points), desc.shape))
    # cv2.imwrite("canny_img.signed.jpg", signed_img)

    ## ----  sift --- ##

    gray_img = ife.get_gray_img(jpg)
    key_points, desc, signed_img = sift_img(gray_img)
    print("gray# kps: {}, descriptors: {}".format(len(key_points), desc.shape))
    cv2.imwrite("gray_img.signed.jpg", signed_img)

    sobel_img = ife.get_sobel_img(jpg)
    key_points, desc, signed_img = sift_img(sobel_img)
    print("sobel# kps: {}, descriptors: {}".format(len(key_points), desc.shape))
    cv2.imwrite("sobel_img.signed.jpg", signed_img)

    canny_img = ife.get_canny_img(jpg)
    key_points, desc, signed_img = sift_img(canny_img)
    print("canny# kps: {}, descriptors: {}".format(len(key_points), desc.shape))
    cv2.imwrite("canny_img.signed.jpg", signed_img)

    ## ----  surf --- ##

    # gray_img = ife.get_gray_img(jpg)
    # key_points, desc, signed_img = surf_img(gray_img)
    # print("gray# kps: {}, descriptors: {}".format(len(key_points), desc.shape))
    # cv2.imwrite("gray_img.signed.jpg", signed_img)
    #
    # sobel_img = ife.get_sobel_img(jpg)
    # key_points, desc, signed_img = surf_img(sobel_img)
    # print("sobel# kps: {}, descriptors: {}".format(len(key_points), desc.shape))
    # cv2.imwrite("sobel_img.signed.jpg", signed_img)
    #
    # canny_img = ife.get_canny_img(jpg)
    # key_points, desc, signed_img = surf_img(canny_img)
    # print("canny# kps: {}, descriptors: {}".format(len(key_points), desc.shape))
    # cv2.imwrite("canny_img.signed.jpg", signed_img)
