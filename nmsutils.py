import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import shutil
import math

# Function to generate a random bounding box
def generate_random_bbox(canvas_size):
    x1 = random.randint(0, 758)
    y1 = random.randint(0, 758)

    x2 = random.randint(x1, 768)
    y2 = random.randint(y1, 768)
    return [x1, y1, x2, y2]

# Function to generate a similar bounding box with some offset
def generate_similar_bbox(gt_bbox, offset_range):

    new_x1 = random.randint(max(0, gt_bbox[0] - offset_range), min(gt_bbox[0] + offset_range, 758))
    new_y1 = random.randint(max(0, gt_bbox[1] - offset_range), min(gt_bbox[1] + offset_range, 758))
    while True:
        try:
            new_x2 = random.randint(max(0, gt_bbox[2] - offset_range), min(gt_bbox[2] + offset_range, 758))
            if new_x2-new_x1 < 10:
                new_x2+=10
            new_y2 = random.randint(max(0, gt_bbox[3] - offset_range), min(gt_bbox[3] + offset_range, 758))
            if new_y2-new_y1 < 10:
                new_y2+=10
        except ValueError:
            new_x2 = 768
            new_y2 = 768
        break
    if new_x1 > new_x2:
        new_x1, new_x2 = new_x2, new_x1
    if new_y1 > new_y2:
        new_y1, new_y2 = new_y2, new_y1
    if new_x1 == new_x2:
        new_x2 +=10
    if new_y1 == new_y2:
        new_y2 +=10

    return [new_x1,new_y1, new_x2, new_y2]

def augment_bbox_big(bbox):
    augment_range_x = 0.2*(bbox[2]-bbox[0])
    augment_range_y = 0.2*(bbox[3]-bbox[1])
    new_x1 = max(0,bbox[0]-augment_range_x)
    new_y1 = max(0,bbox[1]-augment_range_y)
    new_x2 = min(768,bbox[2]+augment_range_x)
    new_y2 = min(768,bbox[3]+augment_range_y)
    return [new_x1,new_y1,new_x2,new_y2]

def augment_bbox_small(bbox):
    augment_range_x = 0.2*(bbox[2]-bbox[0])
    augment_range_y = 0.2*(bbox[3]-bbox[1])
    new_x1 = max(0,bbox[0]-augment_range_x)
    new_y1 = max(0,bbox[1]-augment_range_y)
    new_x2 = min(768,bbox[2]+augment_range_x)
    new_y2 = min(768,bbox[3]+augment_range_y)
    return [new_x1,new_y1,new_x2,new_y2]



def xyxytoxywh(bbox):
    return[bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]

