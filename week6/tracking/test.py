from pycocotools import coco
import cv2
import numpy as np
import imageio
import matplotlib.pylab as plt
from new_tracking import tracking_iou
import copy
import glob
from metrics.mAP import calculate_ap
import json

def frame_with_2bb(gt_bb, det_bb, frame_path, f_val):
    lst_gt = [item[0] for item in gt_bb]
    lst_nogt = [item[0] for item in det_bb]
    frame1 = cv2.imread((frame_path + '/{:06d}.png').format(f_val))

    args_gt = [i for i, num in enumerate(lst_gt) if num == f_val]

    for ar in args_gt:
        # Ground truth bounding box in green
        cv2.rectangle(frame1, (int(gt_bb[ar][3]), int(gt_bb[ar][4])),
                      (int(gt_bb[ar][5]), int(gt_bb[ar][6])), (0, 255, 0), 2)

    args_nogt = [i for i, num in enumerate(lst_nogt) if num == f_val]
    np.random.seed(34)
    r = np.random.randint(0, 256, len(args_nogt), dtype = int)
    g = np.random.randint(0, 256, len(args_nogt), dtype = int)
    b = np.random.randint(0, 256, len(args_nogt), dtype = int)
    for i, ar in enumerate(args_nogt):
        # guessed GT in blue
        cv2.rectangle(frame1, (int(det_bb[ar][3]), int(det_bb[ar][4])),
                      (int(det_bb[ar][5]), int(det_bb[ar][6])),
                      (int(r[i]),int(g[i]),int(b[i])), 2)

#    frame1 = cv2.resize(frame1, (int(1920 / 4), int(1080 / 4)))
#    frame1 = cv2.resize(frame1, (int(1920), int(1080)))
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    imageio.imsave('frame{}_with_2bb.png'.format(f_val), frame1)
    return frame1


def animation_tracks(det_bb, idd, ini_frame, end_frame, frames_path):
    """
    This function creates an animation of the tracks assigning a different color
    to eack track. It also draws a number because the gif animation seems to be
    changing color for the same track when it is not.
    """
    np.random.seed(34)
    r = np.random.randint(0, 256, idd, dtype = int)
    g = np.random.randint(0, 256, idd, dtype = int)
    b = np.random.randint(0, 256, idd, dtype = int)

    images = []

    lst_bb = [item[0] for item in det_bb]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for f_val in range(ini_frame, end_frame):
        frame_bb = [det_bb[i] for i, num in enumerate(lst_bb) if num == f_val]
        frame1 = cv2.imread((frames_path + '/{:06d}.png').format(f_val))
        for fr in frame_bb:
            cv2.rectangle(frame1, (int(fr[3]), int(fr[4])),
                  (int(fr[5]), int(fr[6])),
                  (int(r[fr[2]]), int(g[fr[2]]), int(b[fr[2]])), 2)
            cv2.putText(frame1, str(fr[2]), (int(fr[3]),
                        int(fr[4]) - 10), font, 0.75,
                        (int(r[fr[2]]), int(g[fr[2]]), int(b[fr[2]])), 2, cv2.LINE_AA)
#        frame1 = cv2.resize(frame1, (int(1920 / 2), int(1080 / 2)))
        images.append(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
#    imageio.mimsave('tracking4.gif', images)
    imageio.mimsave('fgs.tiff', images)
    

def bbox_rle(file):
    segm = {
            "counts": file[-1].strip().encode(encoding='UTF-8'),
            "size": [int(file[3]), int(file[4])]
            }

    box = coco.maskUtils.toBbox(segm)
    box[2:] = box[2:] + box[:2]
    box = box.tolist()
    return box

def define_list_gt(fields, box):
    test_list = [int(fields[0]), #frame
                 int(fields[2]), #id (1-> car, 2-> pedestrian, other -> no idea)
                 int(fields[1][1:]), #id
                 int(box[0]), # xTopLeft
                 int(box[1]), # yTopLeft
                 int(box[2]), # xBottomRight
                 int(box[3])] # yBottomRight
    return test_list


def read_gt_txt(path, mode=None):
    lines = open(path).read().splitlines()
    bb = []
    for l in lines:
        fields = l.split(" ")
        box = bbox_rle(fields)
        if mode == 'car':      
            if int(fields[2]) == 1:         
                test_list = define_list_gt(fields, box)
                bb.append(test_list)            
        elif mode == 'pedestrian':
            if int(fields[2]) == 2:         
                test_list = define_list_gt(fields, box)
                bb.append(test_list)            
        else:
            if int(fields[2]) == 2 or int(fields[2]) == 1:         
                test_list = define_list_gt(fields, box)
                bb.append(test_list)            
    return bb



def define_list_det(p):
    if p['category_id'] == 0:
        cat = 1
    else:
        cat = 2
    test_list = [int(str(p['image_id'])[-6:]), # Frame
                 int(cat), # id (1-> car, 2-> pedestrian, other -> no idea)
                 0, # id
                 int(p['bbox'][0]), # xTopLeft
                 int(p['bbox'][1]), # yTopLeft
                 int(p['bbox'][0])+int(p['bbox'][2]), # xBottomRight
                 int(p['bbox'][1])+int(p['bbox'][3]), # yBottomRight
                 float(p['score']), # Confidence
                 p["segmentation"]["size"][0],
                 p["segmentation"]["size"][1],
                  p["segmentation"]["counts"]]
    return test_list

def read_det(cat):
    bb = []
    with open("coco_city_kitti.json") as json_file:
        data = json.load(json_file)
        for p in data:
            if len(str(p['image_id'])) == 8 and int(str(p['image_id'])[0:2])==14:
                if cat == 'car':
                    if p['category_id'] == 0:
                        test_list = define_list_det(p)                 
                        bb.append(test_list)
                elif cat == 'pedestrian':
                    if p['category_id'] == 1:
                        test_list = define_list_det(p)
                        bb.append(test_list)
                else:
                    if p['category_id'] == 2 or p['category_id'] == 1:
                        test_list = define_list_det(p)
                        bb.append(test_list)    
    return bb




def number_of_images(path):
    """
    Computes how many images are in the given path
    """
    return len(glob.glob1(path, "*.png"))


path = 'C:/Users/Sara/Documents/GitHub/Visual-recognition/week6/testing_sara/KITTI-MOTS/instances_txt/0014.txt'    
gt_bb = read_gt_txt(path, mode='car')
det_bb = read_det(cat = 'car')


frame_path = "C:/Users/Sara/Documents/GitHub/Visual-recognition/week6/testing_sara/KITTI-MOTS/training/image_02/0014"


video_n_frames = number_of_images(frame_path)
det_bb_max_iou, idd = tracking_iou(frame_path, copy.deepcopy(det_bb), video_n_frames, mode='normal')

ap_max_iou = calculate_ap(det_bb_max_iou, copy.deepcopy(gt_bb), 0, video_n_frames, mode='area')

frame_with_2bb(gt_bb, det_bb, frame_path, 10)

animation_tracks(det_bb_max_iou, idd, 0, video_n_frames, frame_path)

f=open('0014.txt','w')
for ele in det_bb_max_iou:
    f.write(str(ele[0])+' ')
    f.write(str(ele[2])+' ')
    f.write(str(ele[1])+' ')
    f.write(str(ele[8])+' ')
    f.write(str(ele[9])+' ')
    f.write(str(ele[10])+'\n')
    

f.close()
#C:\Users\Sara\Documents\GitHub\Visual-recognition\week6\testing_sara\KITTI-MOTS\instances_txt

