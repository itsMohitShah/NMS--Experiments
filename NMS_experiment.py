import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import shutil
import math
from nmsutils import *
import pandas as pd



def calculate_iou(bbox1, bbox2):

    """
    Box1 and Box2 need to be of type [x1,y1,x2,y2]
    """
    xA = min(bbox1[2], bbox2[2])
    xB = max(bbox1[0], bbox2[0])

    yA = min(bbox1[3], bbox2[3])
    yB = max(bbox1[1], bbox2[1])

    intersection = max(0, xA-xB) * max(0, yA-yB)
    w1, h1 = bbox1[2]-bbox1[0], bbox1[3]-bbox1[1]
    w2, h2 = bbox2[2]-bbox2[0], bbox2[3]-bbox2[1]

    box1_area = (w1)*(h1)
    box2_area = (w1)*(h2)
    union = (box1_area)+(box2_area) - intersection
    if union == 0:
        print(f"Union of following boxes = 0:: Box1: {bbox1}\nBox2: {bbox2}")
    iou = intersection/union
    eps=1e-7

    cw = max(bbox1[2],bbox2[2]) - min(bbox1[0],bbox2[0])  # convex (smallest enclosing box) width
    ch = max(bbox1[3],bbox2[3]) - min(bbox1[1],bbox2[1])  # convex height

    c2 = cw**2 + ch**2 + eps  # convex diagonal squared
    rho2 = (
        (bbox2[0] + bbox2[2] - bbox1[0] - bbox1[2])**2 + (bbox2[1] + bbox2[3] - bbox1[1] - bbox1[3])**2) / 4  # center dist**2

    v = (4 / math.pi**2) * (np.arctan(w2 / h2) - np.arctan(w1 / h1))**2            
    alpha = v / (v - iou + (1 + eps))
    ciou =  iou - (rho2 / c2 + v * alpha)  # CIoU
    diou =  iou - rho2 / c2  # DIoU
    c_area = cw * ch + eps  # convex area
    giou = iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou, giou, diou, ciou




def main(iterations = 5,clearall = False):
    path_images_folder = 'Images_NMS'
    path_bothsuppressed_folder = os.path.join(path_images_folder,'BothSuppressed')
    path_nonesuppressed_folder = os.path.join(path_images_folder,'NoneSuppressed')
    path_xyxy_folder = os.path.join(path_images_folder,'XYXY')
    if clearall == True:
        if os.path.exists(path_images_folder):
            shutil.rmtree(path_images_folder)
    os.makedirs(path_images_folder, exist_ok=True)
    os.makedirs(path_bothsuppressed_folder, exist_ok=True)
    os.makedirs(path_nonesuppressed_folder, exist_ok=True)
    os.makedirs(path_xyxy_folder, exist_ok=True)
    df = pd.DataFrame({'Image':[],
                       'GT':[],
                       'BBox1':[],
                       'BBox2':[],
                       'Status':[],
                       'IoU':[],
                       'GIoU':[],
                       'DIoU':[],
                       'CIoU':[]})
    for i in tqdm(range(iterations)):
        position_x = 20
        position_y = 20
        font = cv2.FONT_HERSHEY_DUPLEX
        font_size = 0.6
        font_thickness = 1
        # Canvas size and bounding box size
        canvas_size = (768, 768)
        # bbox_size = (random.randint(10,768), random.randint(10,768))
        offset_range = random.randint(1,100)

        # Create a blank white canvas
        canvas = np.ones((canvas_size[0], canvas_size[1], 3), dtype=np.uint8) * 255

        # Generate ground truth bounding box
        gt_bbox = generate_random_bbox(canvas_size)

        # Generate two similar bounding boxes
        bbox1 = generate_similar_bbox(gt_bbox, offset_range)
        bbox2 = generate_similar_bbox(gt_bbox, offset_range)
        iou, giou, diou, ciou = calculate_iou(bbox1,bbox2)

        # Draw the bounding boxes on the canvas
        cv2.rectangle(canvas, (gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3]), (0, 255, 0), 2)  # Ground truth in green

        
        canvas = cv2.putText(canvas, f"GT: {gt_bbox}", (position_x, position_y),font, font_size, (0, 255, 0), font_thickness)
        canvas = cv2.putText(canvas, f"Bbox1: {bbox1}", (position_x, position_y+20),font, font_size,(255, 0, 0), font_thickness)
        canvas = cv2.putText(canvas, f"Bbox2: {bbox2}", (position_x, position_y+40),font, font_size, (0, 0, 255), font_thickness)
        canvas = cv2.putText(canvas, "IoU: {:.2f}".format(iou), (position_x, position_y+60),font, font_size, (0, 0, 0), font_thickness)
        canvas = cv2.putText(canvas, "GIoU: {:.2f}".format(giou), (position_x, position_y+80),font, font_size, (0, 0, 0), font_thickness)
        canvas = cv2.putText(canvas, "DIoU: {:.2f}".format(diou), (position_x, position_y+100),font, font_size, (0, 0, 0), font_thickness)
        canvas = cv2.putText(canvas, "CIoU: {:.2f}".format(ciou), (position_x, position_y+120),font, font_size, (0, 0, 0), font_thickness)
        
        canvas_correctNMS = canvas.copy()
        canvas_customNMS = canvas.copy()        

        # cv2.rectangle(canvas, (bbox1[0], bbox1[1]), (bbox1[2], bbox1[3]), (255, 0, 0), 2)  # Similar box 1 in blue
        # cv2.rectangle(canvas, (bbox2[0], bbox2[1]), (bbox2[2], bbox2[3]), (0, 0, 255), 2)  # Similar box 2 in red

        bbox1_xywh = xyxytoxywh(bbox1)
        bbox2_xywh = xyxytoxywh(bbox2)
        #! Swapping the Bbox values. They are now in xywh format
        bbox1_xyxy = bbox1
        bbox2_xyxy = bbox2

        bbox1 = bbox1_xywh
        bbox2 = bbox2_xywh


        list_bbox = [bbox1_xywh,bbox2_xywh]
        list_bbox_xyxy = [bbox1_xyxy,bbox2_xyxy]
        list_conf = [0.90,0.65]
        list_colour = [(255, 0, 0),(0, 0, 255)]
        idx_CorrectNMS = cv2.dnn.NMSBoxes(list_bbox,list_conf,0.3,0.5)
        idx_NMSXYXY = cv2.dnn.NMSBoxes(list_bbox_xyxy,list_conf,0.3,0.5)




        # Draw the bounding boxes on the canvas
        for idx in range(len(list_bbox_xyxy)):
            cv2.rectangle(canvas, (list_bbox_xyxy[idx][0], list_bbox_xyxy[idx][1]), (list_bbox_xyxy[idx][2], list_bbox_xyxy[idx][3]), list_colour[idx], 2)  

        # Draw the bounding boxes on the canvas
        for idx in idx_CorrectNMS:
            cv2.rectangle(canvas_correctNMS, (list_bbox_xyxy[idx][0], list_bbox_xyxy[idx][1]), (list_bbox_xyxy[idx][2], list_bbox_xyxy[idx][3]), list_colour[idx], 2)  
        # Draw the bounding boxes on the canvas
        for idx in idx_NMSXYXY:
            # if idx in idx_CustomNMS2:
            cv2.rectangle(canvas_customNMS, (list_bbox_xyxy[idx][0], list_bbox_xyxy[idx][1]), (list_bbox_xyxy[idx][2], list_bbox_xyxy[idx][3]), list_colour[idx], 2)



        # Display the result
        plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        plt.title(f'{i+1}{iterations}\nBlue: {bbox1}\nRed: {bbox2}\nIoU: {iou}', fontsize = 10)
        if len(idx_CorrectNMS) == len(idx_NMSXYXY):
            if len(idx_CorrectNMS) == 2:
                filename = os.path.join(path_nonesuppressed_folder,"NOTSuppressedinBothCases_image_{}.png")
                status = 'NotSuppressedinBothCases'
            else:
                filename = os.path.join(path_bothsuppressed_folder,"SuppressedinBothCases_image_{}.png")
                status = 'SuppressedinBothCases'
        else:
            filename = os.path.join(path_xyxy_folder,"XYXY_image_{}.png")
            status = 'XYXY'
        filecounter = 0
        while os.path.isfile(filename.format(filecounter)):
            filecounter+=1
        filename = filename.format(filecounter)

        finalimage = plt.figure(figsize=(20,20))
        ax1 = plt.subplot2grid(shape=(3,2),loc=(0,0),colspan=2, rowspan= 2)
        ax2 = plt.subplot2grid(shape=(3,2),loc=(2,0))
        ax3 = plt.subplot2grid(shape=(3,2),loc=(2,1))


        ax1.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        ax2.set_title("IoU = 0.5")
        ax2.imshow(cv2.cvtColor(canvas_correctNMS, cv2.COLOR_BGR2RGB))
        ax3.set_title("IoU = 0.5")
        ax3.imshow(cv2.cvtColor(canvas_customNMS, cv2.COLOR_BGR2RGB))



        # plt.imsave(filename,finalimage)

        finalimage.savefig(filename)
        plt.close()
        list_allinfo = [os.path.split(filename)[-1],gt_bbox,bbox1_xyxy,bbox2_xyxy,status,iou,giou,diou,ciou]
        df.loc[len(df)] = list_allinfo

        df.to_excel(os.path.join(path_images_folder,'Results.xlsx'))

if __name__ == "__main__":
    main(clearall=True, iterations= 1000)

