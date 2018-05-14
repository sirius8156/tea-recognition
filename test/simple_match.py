# coding=utf-8
import cv2
import numpy as np

import tools.feature_extract as ife
import tools.feature_key_point as fkp


def test_match(img_path_1, img_path_2):
    canny_img_1 = ife.get_canny_img(img_path_1)
    key_points_1, desc_1, signed_img_1 = fkp.sift_img(canny_img_1)

    canny_img_2 = ife.get_canny_img(img_path_2)
    key_points_2, desc_2, signed_img_2 = fkp.sift_img(canny_img_2)

    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(desc_1, desc_2, k=2)
    matches = sorted(matches, key=lambda x: x[0].distance)
    good = [m1 for (m1, m2) in matches if m1.distance < 0.9 * m2.distance]

    good = np.expand_dims(good, 1)
    img3 = cv2.drawMatchesKnn(canny_img_1, key_points_1, canny_img_2, key_points_2, good[0:20], outImg=None, flags=2)
    return good, img3


if __name__ == '__main__':
    compare_1_good, compare_1_img = test_match("cropped_1.jpg", "cropped_2.jpg")
    compare_2_good, compare_2_img = test_match("cropped_1.jpg", "cropped_3.jpg")
    compare_3_good, compare_3_img = test_match("cropped_1.jpg", "cropped_4.jpg")
    compare_4_good, compare_3_img = test_match("cropped_2.jpg", "cropped_3.jpg")
    compare_5_good, compare_3_img = test_match("cropped_2.jpg", "cropped_4.jpg")

    print("compare_1#good: {}, compare_2#good: {}, compare_3#good: {}, compare_4#good: {}, compare_5#good: {}"
          .format(len(compare_1_good),
                  len(compare_2_good),
                  len(compare_3_good),
                  len(compare_4_good),
                  len(compare_5_good)))

    # TODO 目前观察阈值len(good)卡在100, 可以区分出特征差异
    # TODO 后续需要大量实验sample块的大小, 验证阈值
    # TODO canny特征向量数量小, 如果用sobel差异会更明显, 这里也需要实验, 找到合适的算子, 在速度和复杂度中找到优化

    cv2.imshow("compare_1", compare_1_img)
    cv2.imshow("compare_2", compare_2_img)
    cv2.imshow("compare_3", compare_3_img)
    cv2.imshow("compare_4", compare_3_img)
    cv2.imshow("compare_5", compare_3_img)

    cv2.waitKey(0)

    # matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), {})
    #
    # knn_matches = matcher.knnMatch(desc_1, desc_2, 2)
    # knn_matches = matches = sorted(knn_matches, key=lambda x: x[0].distance)
    #
    # good = [m1 for (m1, m2) in knn_matches if m1.distance < 0.7 * m2.distance]
    #
    # src_pts = np.float32([key_points_1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    # dst_pts = np.float32([key_points_2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    #
    # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    #
    # h, w = canny_img_1.shape[:2]
    # pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    # dst = cv2.perspectiveTransform(pts, M)
    #
    # canvas = canny_img_2.copy()
    # cv2.polylines(canvas, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
    # matched = cv2.drawMatches(canny_img_1, key_points_1, canvas, key_points_2, good, None)  # ,**draw_params)
    #
    # h, w = canny_img_1.shape[:2]
    # pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    # dst = cv2.perspectiveTransform(pts, M)
    # perspectiveM = cv2.getPerspectiveTransform(np.float32(dst), pts)
    # found = cv2.warpPerspective(canny_img_2, perspectiveM, (w, h))
    #
    # cv2.imshow("matched", matched)
    # cv2.imshow("found", found)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
