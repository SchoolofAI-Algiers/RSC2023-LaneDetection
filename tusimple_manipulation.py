# import required packages
import json
import os.path

import numpy as np
import cv2
import matplotlib.pyplot as plt# read each line of json file
json_gt = [json.loads(line) for line in open('./datasets/tusimple/TUSimple/train_set/label_data_0313.json')]
gt = json_gt[100]
gt_lanes = gt['lanes']

# print(gt_lanes)
y_samples = gt['h_samples']
print(y_samples)
raw_file = gt['raw_file']# see the image
data_root = '/mnt/Travail/DLProjects/RSC/LaneRSC/datasets/tusimple/TUSimple/train_set'
img_path = os.path.join(data_root, raw_file)
# print(img_path)
img = cv2.imread(img_path)


gt_lanes_vis = [[(x, y) for (x, y) in zip(lane, y_samples)
                  if x >= 0] for lane in gt_lanes]
img_vis = img.copy()

# print(gt_lanes_vis)
for lane in gt_lanes_vis:
    # print(lane)
    cv2.polylines(img_vis, np.int32([lane]), isClosed=False,
                   color=(0,255,0), thickness=5)

plt.figure(figsize=(16, 18))
plt.imshow(img_vis)
plt.show()

# cv2.imshow('image',img)
# cv2.WaitKey(5000)
# cv2.destroyAllWindows()

