import cv2
import numpy as np
import random


# def get_cartoon(p, l=0.1, s=0.95):
#     p = cv2.bilateralFilter(p, 5, 150, 150)
#
#     gray = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
#     p2 = cv2.Canny(gray, 30, 100)
#
#     p = p.astype(np.float32) / 255.0
#     p = cv2.cvtColor(p, cv2.COLOR_BGR2HLS)
#
#     p[:, :, 1] = (1.0 + l) * p[:, :, 1]
#     p[:, :, 1][p[:, :, 1] > 1] = 1
#     # 饱和度
#     p[:, :, 2] = (1.0 + s) * p[:, :, 2]
#     p[:, :, 2][p[:, :, 2] > 1] = 1
#     # HLS2BGR
#     p = cv2.cvtColor(p, cv2.COLOR_HLS2BGR) * 255
#     p = np.array(p, dtype=np.int16)
#     p[:, :, 0] = np.clip(p[:, :, 0] - p2, 0, 255)
#     p[:, :, 1] = np.clip(p[:, :, 1] - p2, 0, 255)
#     p[:, :, 2] = np.clip(p[:, :, 2] - p2, 0, 255)
#     p = np.array(p, dtype=np.uint8)
#     return p



