import os
import mohitutils
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import shutil




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
            new_y2 = random.randint(max(0, gt_bbox[3] - offset_range), min(gt_bbox[3] + offset_range, 758))
        except ValueError:
            new_x2 = 768
            new_y2 = 768
        break
    if new_x1 > new_x2:
        new_x1, new_x2 = new_x2, new_x1
    if new_y1 > new_y2:
        new_y1, new_y2 = new_y2, new_y1

    return [new_x1,new_y1, new_x2, new_y2]


def calculate_iou(bbox1, bbox2):

    """
    Box1 and Box2 need to be of type [x1,y1,x2,y2]
    """
    xA = min(bbox1[2], bbox2[2])
    xB = max(bbox1[0], bbox2[0])

    yA = min(bbox1[3], bbox2[3])
    yB = max(bbox1[1], bbox2[1])

    intersection = max(0, xA-xB) * max(0, yA-yB)

    box1_area = (bbox1[2]-bbox1[0])*(bbox1[3]-bbox1[1])
    box2_area = (bbox2[2]-bbox2[0])*(bbox2[3]-bbox2[1])
    union = (box1_area)+(box2_area) - intersection
    if union == 0:
        print(f"Union of following boxes = 0:: Box1: {bbox1}\nBox2: {bbox2}")
    iou = intersection/union
    return iou


def xyxytoxywh(bbox):
    return[bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]



def main(iterations = 5,clearall = False):
    destination_folder = 'Images'
    if clearall == True:
        if os.path.exists(destination_folder):
            shutil.rmtree(destination_folder)
    os.makedirs(destination_folder, exist_ok=True)
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
        iou = calculate_iou(bbox1,bbox2)

        # Draw the bounding boxes on the canvas
        cv2.rectangle(canvas, (gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3]), (0, 255, 0), 2)  # Ground truth in green

        
        canvas = cv2.putText(canvas, f"GT: {gt_bbox}", (position_x, position_y),
                            font, font_size, (0, 255, 0), font_thickness)
        canvas = cv2.putText(canvas, f"Bbox1: {bbox1}", (position_x, position_y+20),
                            font, font_size,(255, 0, 0), font_thickness)
        canvas = cv2.putText(canvas, f"Bbox2: {bbox2}", (position_x, position_y+40),
                            font, font_size, (0, 0, 255), font_thickness)
        canvas = cv2.putText(canvas, "IoU: {:.2f}".format(iou), (position_x, position_y+60),
                            font, font_size, (0, 0, 0), font_thickness)
        
        canvas_correctNMS = canvas.copy()
        canvas_customNMS = canvas.copy()        
        canvas_correctNMS00 = canvas.copy()
        canvas_customNMS00 = canvas.copy()
        cv2.rectangle(canvas, (bbox1[0], bbox1[1]), (bbox1[2], bbox1[3]), (255, 0, 0), 2)  # Similar box 1 in blue
        cv2.rectangle(canvas, (bbox2[0], bbox2[1]), (bbox2[2], bbox2[3]), (0, 0, 255), 2)  # Similar box 2 in red

        bbox1_xywh = xyxytoxywh(bbox1)
        bbox2_xywh = xyxytoxywh(bbox2)

        list_bbox_xywh = [bbox1_xywh,bbox2_xywh]
        list_bbox = [bbox1,bbox2]
        list_conf = [0.90,0.65]
        list_colour = [(255, 0, 0),(0, 0, 255)]
        idx_CorrectNMS = cv2.dnn.NMSBoxes(list_bbox_xywh,list_conf,0.3,0.5)
        idx_CustomNMS = cv2.dnn.NMSBoxes(list_bbox,list_conf,0.3,0.5)

        idx_CorrectNMS00 = cv2.dnn.NMSBoxes(list_bbox_xywh,list_conf,0.3,0.0)
        idx_CustomNMS00 = cv2.dnn.NMSBoxes(list_bbox,list_conf,0.3,0.0)

        # Draw the bounding boxes on the canvas
        for idx in range(len(list_bbox)):
            cv2.rectangle(canvas, (list_bbox[idx][0], list_bbox[idx][1]), (list_bbox[idx][2], list_bbox[idx][3]), list_colour[idx], 2)  

        # Draw the bounding boxes on the canvas
        for idx in idx_CorrectNMS:
            cv2.rectangle(canvas_correctNMS, (list_bbox[idx][0], list_bbox[idx][1]), (list_bbox[idx][2], list_bbox[idx][3]), list_colour[idx], 2)  
        # Draw the bounding boxes on the canvas
        for idx in idx_CustomNMS:
            cv2.rectangle(canvas_customNMS, (list_bbox[idx][0], list_bbox[idx][1]), (list_bbox[idx][2], list_bbox[idx][3]), list_colour[idx], 2)  


        # Draw the bounding boxes on the canvas
        for idx in idx_CorrectNMS00:
            cv2.rectangle(canvas_correctNMS00, (list_bbox[idx][0], list_bbox[idx][1]), (list_bbox[idx][2], list_bbox[idx][3]), list_colour[idx], 2)  
        # Draw the bounding boxes on the canvas
        for idx in idx_CustomNMS00:
            cv2.rectangle(canvas_customNMS00, (list_bbox[idx][0], list_bbox[idx][1]), (list_bbox[idx][2], list_bbox[idx][3]), list_colour[idx], 2)  







        # Display the result
        plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        plt.title(f'{i+1}{iterations}\nBlue: {bbox1}\nRed: {bbox2}\nBoxes with IoU: {iou}', fontsize = 10)
        # plt.show()
        if len(idx_CorrectNMS) == len(idx_CustomNMS):
            if len(idx_CustomNMS) == 2:
                filename = os.path.join(destination_folder,"NOTSuppressedinBothCases_image_{}.png")
            else:
                filename = os.path.join(destination_folder,"SuppressedinBothCases_image_{}.png")
        else:
            filename = os.path.join(destination_folder,"XYXY_image_{}.png")
        filecounter = 0
        while os.path.isfile(filename.format(filecounter)):
            filecounter+=1
        filename = filename.format(filecounter)

        finalimage = plt.figure(figsize=(20,20))
        ax1 = plt.subplot2grid(shape=(4,2),loc=(0,0),colspan=2, rowspan= 2)
        ax2 = plt.subplot2grid(shape=(4,2),loc=(2,0))
        ax3 = plt.subplot2grid(shape=(4,2),loc=(2,1))
        ax4 = plt.subplot2grid(shape=(4,2),loc=(3,0))
        ax5 = plt.subplot2grid(shape=(4,2),loc=(3,1))

        ax1.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        ax2.set_title("IoU = 0.5")
        ax2.imshow(cv2.cvtColor(canvas_correctNMS, cv2.COLOR_BGR2RGB))
        ax3.set_title("IoU = 0.5")
        ax3.imshow(cv2.cvtColor(canvas_customNMS, cv2.COLOR_BGR2RGB))

        ax4.set_title("IoU = 0.0")
        ax4.imshow(cv2.cvtColor(canvas_correctNMS00, cv2.COLOR_BGR2RGB))
        ax5.set_title("IoU = 0.0")
        ax5.imshow(cv2.cvtColor(canvas_customNMS00, cv2.COLOR_BGR2RGB))

        # plt.imsave(filename,finalimage)

        finalimage.savefig(filename)
        plt.close()

if __name__ == "__main__":
    main(clearall=False, iterations= 1000)

