from cv_bridge import CvBridge, CvBridgeError

import numpy as np
import heapq
import cv2

import copy
import os


class ImageTools(object):
    def __init__(self, img_id=0):
        self.cv2_img = None
        self.img_id = img_id

    def callback(self, msg):
        try:
            bridge = CvBridge()
            self.cv2_img = bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError, e:
            print e
        else:
            print "successfully receive an image"

    def sampling_image(self, patch_size=224, num_patches=400):
        height, width, _ = self.cv2_img.shape
        c_h = np.random.randint(patch_size / 2 + 20, (width - patch_size / 2), num_patches)  # 20:remove the grasper
        c_w = np.random.randint(patch_size / 2, (height - patch_size / 2), num_patches)  # patch_size/2
        threshold = 25
        images = []
        coors = []
        for i in xrange(num_patches):
            img = self.cv2_img[c_h - patch_size / 2: c_h + patch_size / 2 + patch_size % 2,
                               c_w - patch_size / 2: c_w + patch_size / 2 + patch_size % 2, :]
            if np.std(img) > threshold:
                images.append(img)
                coors.append((c_h[i], c_w[i]))
        coors = np.array(coors)
        images = np.concatenate(images, axis=0)
        return coors, images

    def resampling_image(self, scores, coors, patch_size=224):
        min_value_index = np.where(scores == scores.min())[0]
        indices = np.where(scores <= heapq.nsmallest(37, scores.flatten())[-1])[0]
        min_value_coor = coors[min_value_index][0]
        coors = coors[indices]
        new_coors = [coor for coor in coors if
                     abs(coor[0] - min_value_coor[0]) < 20 and abs(coor[1] - min_value_coor[1]) < 20]
        new_coor = np.mean(new_coors, axis=0, dtype=np.int)
        # get the best patch's location
        new_patch = self.cv2_img[(new_coor[0] - patch_size / 2):(new_coor[0] + patch_size / 2) + patch_size % 2,
                                 (new_coor[1] - patch_size / 2):(new_coor[1] + patch_size / 2) + patch_size % 2, :]
        img = np.expand_dims(new_patch, axis=0).astype(np.float32)
        return img, new_coor

    def save_image(self, path):
        cv2.imwrite(os.path.join(path, '{:06d}.jpg'.format(self.img_id)), self.cv2_img)
        self.img_id += 1

    def display_image(self):
        cv2.imshow('wirst_image', self.cv2_img)
        print 'q to quit.'
        while 1:
            if cv2.waitKey(1) & 0xff == ord('q'):
                break

    def process_and_draw_rect(self, grasp_angle, sx, sy):
        img_temp = copy.deepcopy(self.cv2_img)
        grasp_l = 200 / 3.0
        grasp_w = 200 / 6.0
        points = np.array([[-grasp_l, -grasp_w],
                           [grasp_l, -grasp_w],
                           [grasp_l, grasp_w],
                           [-grasp_l, grasp_w]])
        rotate_matrix = np.array([[np.cos(grasp_angle), -np.sin(grasp_angle)],
                      [np.sin(grasp_angle), np.cos(grasp_angle)]])
        rot_points = np.dot(rotate_matrix, points.transpose()).transpose()
        temp = np.array([[sx, sy],
                         [sx, sy],
                         [sx, sy],
                         [sx, sy]])
        im_points = (rot_points + temp).astype(np.int)
        cv2.line(img_temp, tuple(im_points[0]), tuple(im_points[1]), color=(0, 255, 0), thickness=5)
        cv2.line(img_temp, tuple(im_points[1]), tuple(im_points[2]), color=(0, 0, 255), thickness=5)
        cv2.line(img_temp, tuple(im_points[2]), tuple(im_points[3]), color=(0, 255, 0), thickness=5)
        cv2.line(img_temp, tuple(im_points[3]), tuple(im_points[0]), color=(0, 0, 255), thickness=5)
        return img_temp

    def process_and_draw_rect_patch(self, sx, sy, color):
        img_temp = copy.deepcopy(self.cv2_img)
        color = (0, 0, 255) if color=='blue' else (0, 255, 0)
        a1 = sx - 113
        b1 = sy - 113
        a2 = sx + 113
        b2 = sy - 113
        a3 = sx - 113
        b3 = sy + 113
        a4 = sx + 113
        b4 = sy + 113
        temp = np.array([[a1, b1],
                         [a2, b2],
                         [a3, b3],
                         [a4, b4]]).astype(np.int)
        cv2.line(img_temp, tuple(temp[0]), tuple(temp[1]), color=color, thickness=1)
        cv2.line(img_temp, tuple(temp[1]), tuple(temp[3]), color=color, thickness=1)
        cv2.line(img_temp, tuple(temp[2]), tuple(temp[3]), color=color, thickness=1)
        cv2.line(img_temp, tuple(temp[0]), tuple(temp[2]), color=color, thickness=1)
        return img_temp
