#Title: Inference.py
#Data: 20 05 2021
#Author:Fischer @ Cyberlabs
#Usage: python3 inference.py

#score = mean values of Z axis

import argparse
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from tools.DPC3_NET import DPC3_NET
import csv
import io
from retinaface.retinaface import RetinaFace
import numpy as np
import cv2

def transform_image(img_crop):
    my_transforms = transforms.Compose ([transforms.Resize([224, 224]),
                                         transforms.ToTensor(),
                                         # Change this according to your normalization defined on Training
                                         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    
    image = Image.fromarray(img_crop)
    return my_transforms(image).unsqueeze(0)
#===========================================
def prediction(model, img_crop, device, threshold=0.5):
    tensor = transform_image(img_crop)
    tensor = tensor.to(device)
    with torch.no_grad():
        output = model.forward(tensor)
        print(torch.mean(output, dim=(2)))
        score = torch.mean(output, dim=(1, 2))
        pred = (score > threshold)

    return pred, score

def load_model(network, model_path):

    state = torch.load(model_path)

    network.load_state_dict(state['state_dict'])
    network.eval()
    
    return network

def main():
    print("Starting inference")
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",type=str,default="experiments/model.pth", help="input trained model")
    parser.add_argument("--img", type=str, help="input image to inference")
    args=parser.parse_args()
    #-------------------------------------------

    if args.img is None and args.folder is None:
        print("Error no image or folder were given")
        print("Usage: python3 inference.py --img [PATH]")
        return
    elif args.img is not None and args.folder is not None:
        print("Error image and folder were given, you can only execute one")
        print("Usage: python3 inference.py --img [PATH]")
        return

    if(args.img is not None):
        print("Inferencing on an image:" + args.img)
        mode="image"

    if(args.model is None): 
        print("Error no model available")
        return

    print("Reading model: " + args.model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    network = DPC3_NET()
    
    #if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    network = torch.nn.DataParallel(network)
    network = network.to(device)
    
    network = load_model(network, args.model)
    print("OK")
    
    retina = RetinaFace('retinaface/mnet.25', 0, 0, 'net3')
            
    img = cv2.imread(args.img)
        
    boxes, _ = retina.detect(img, threshold=0.5)
    
    # Take the box with highest confidence
    scores = boxes[:, -1]
    bbs = boxes[:, :-1]
    i = np.argmax(scores)
    # best_score = scores[i]
    best_box = bbs[i]
    
    # print(best_box)
    
    xmin, xmax, ymin, ymax = best_box
    xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)
    img_crop = img[xmin:xmax, ymin:ymax, ::-1]# Crop and convert from BGR to RGB
    
    # cv2.imwrite('data/test_crop.jpg', img_crop[:, :, ::-1])
    
    print("Result")
    print(prediction(network, img_crop, device))
    

#========================================================
if __name__=="__main__":
    main()
