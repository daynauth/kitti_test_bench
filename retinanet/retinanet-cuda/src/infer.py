import sys
import torch
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from PIL import Image
import torch.nn.functional as F
import numpy as np
import os
import torchvision.transforms as transforms
import cv2
import time
import argparse
import pandas as pd
import glob
import requests

from shared.utils import rest_put



sys.path.append("/workspace/retinanet")

kitti_classes = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc"]

COLORS = np.random.uniform(0, 255, size=(len(kitti_classes), 3))

from retinanet.model import Model

def parse(args):
    parser = argparse.ArgumentParser(description='Retinanet Kitti Detection')

    parser.add_argument('--model', type=str, help='path to output model or checkpoint to resume from', default = '/models/retinanet-kitti-rn50fpn_fp.pth')
    parser.add_argument('--images', type=str, help='directory where test images are located', default='/dataset/Kitti-Coco/validate/data')
    parser.add_argument('--output-images', type=str, help='directory where result images are stored', default='/output/images')
    parser.add_argument('--output-files', type=str, help='directory where result files are stored', default='/output/files')
    parser.add_argument('--ground-truth', type=str, help='directory where ground truth files are located', default='/dataset/Kitti-Coco/validate/labels')
    parser.add_argument('--full-precision', help='inference in full precision', action='store_true')
    parser.add_argument('--threshold', help='threshold for determining bbox', default=0.5)
    return parser.parse_args(args)

def basename(file):
    return os.path.basename(file)

def file_name(file):
    file = basename(file)
    return os.path.splitext(file)

def pre_process_data(im, device = 'cpu', resize = 800,  max_size = 1333, stride=128):
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225]

    #im = Image.open(path).convert("RGB")

    ratio = resize / min(im.size)
    if ratio * max(im.size) > max_size:
        ratio = max_size / max(im.size)
    im = im.resize((int(ratio * d) for d in im.size), Image.BILINEAR)

    data = torch.ByteTensor(torch.ByteStorage.from_buffer(im.tobytes()))
    data = data.float().div(255).view(*im.size[::-1], len(im.mode))
    data = data.permute(2, 0, 1)

    for t, m, s in zip(data, mean, std):
        t.sub_(m).div_(s)

    pw, ph = ((stride - d % stride) % stride for d in im.size)
    data = F.pad(data, (0, pw, 0, ph))

    data = data.unsqueeze(0)

    data = data.cuda(non_blocking=True)

    return data, ratio

def draw_box(image, results, output_dir, image_name):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    for scores, box, classes in results:
        color = COLORS[classes]
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, lineType=cv2.LINE_AA)

        cv2.putText(image, kitti_classes[classes], (int(box[0]), int(box[1]-5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, 
            lineType=cv2.LINE_AA)
        #cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), -1)

    output = os.path.join(output_dir, image_name + '.png')
    print('output image saved to, ', output)
    cv2.imwrite(output, image)

def detect(model, image, ratio):
    with torch.no_grad():
        scores, boxes, classes = model(image)


    scores = scores[0].detach().cpu().numpy()
    boxes = boxes[0].detach().cpu().numpy()/ratio
    classes = classes[0].int().detach().cpu().numpy()

    
    return scores, boxes, classes

def valid_detection(threshold, scores, boxes, classes):
    results = []

    for s, b, c in zip(scores, boxes, classes):
        if s > threshold:
            results.append((s, b, c))

    return results

def load_model(file_name, mixed_precision = False):
    model, state = Model.load(file_name)
    stride = model.module.stride if isinstance(model, DDP) else model.stride

    #initialize model
    model.share_memory()

    if torch.cuda.is_available():
        model = model.cuda()    
    
    model = amp.initialize(model, None,
                                    opt_level='O2' if mixed_precision else 'O0',
                                    keep_batchnorm_fp32=True,
                                    verbosity=0)

    model.eval()

    return model, stride

def write_results(results, directory, basename):
    output = os.path.join(directory, basename + '.txt')
    print('writing results to ', output)
    with open(output, 'w') as file:
        for result in results:
            x1, y1, x2, y2 = result[1]
            prob = result[0]
            label = kitti_classes[result[2]]
            file.write('%s %s %s %s %s %s\n' % (label, prob, x1, y1, x2, y2))
    
def run_test(model, stride):
    print('running initial test ...')
    img_ptr = 'test_image.png'
    image = Image.open(img_ptr).convert("RGB")
    processed_image, ratio = pre_process_data(image, stride=stride)
    _, _, _ = detect(model, processed_image, ratio)

    print('test passed ...')
    #TODO throw some kind of an error here

def calculate_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def calculate_precision(results, path, overlap = 0.5):
    with open(path, 'r') as file:
        content = file.readlines()

    content = [x.strip() for x in content]

    gt_bbox = []
    

    for line in content:
        if len(line.split()) == 6:
            class_name, prob, top, left, bottom, right = line.split()
        elif len(line.split()) == 5:
            class_name, top, left, bottom, right = line.split()
        else:
            raise RuntimeError("Ground truth file not in correct format")

        gt_bbox.append([float(top), float(left), float(bottom), float(right)])


    if len(gt_bbox) == 0:
        return 0

    else:    
        count = 0

        for result in results:
            for gt_box in gt_bbox:
                iou = calculate_iou(result[1], gt_box)

                if iou > overlap:
                    count += 1

        return count/len(gt_bbox)

        
def run_inferences(images, model, stride, mixed_precision, output_results_dir, output_image_dir, threshold = 0.5, ground_truth = None):
    image_list = [os.path.join(images, f) for f in os.listdir(images) if os.path.isfile(os.path.join(images, f))]
   
    for img in image_list:
        image = Image.open(img).convert("RGB")

        #id to push to database
        id, _ = file_name(img)

        processed_image, ratio = pre_process_data(image, stride=stride)


        start = time.time()
        scores, boxes, classes = detect(model, processed_image, ratio)
        total = (time.time() - start) * 1000 #inference time to push to database

        results = valid_detection(threshold, scores, boxes, classes)

        write_results(results, output_results_dir, id)
        draw_box(image, results, output_image_dir, id)

        gt_path = os.path.join(ground_truth, id + '.txt')

        precision = 0
        if os.path.exists(gt_path):
            precision = calculate_precision(results, gt_path)
            print(precision)      

        data = {
            'image': os.path.basename(img),
            'device': 'gpu',
            'model': 'retinanet',
            'bits' : 32,
            'inference_time' : total,
            'precision' : precision,
            'energy' : 0
        }

        #print(data)
        rest_put(data) 



def main(args = None):
    args = parse(args or sys.argv[1:])

    #output director to store the image file
    
    output_results_dir = args.output_files
    output_image_dir = args.output_images
    threshold = args.threshold
    mixed_precision = not args.full_precision
    images = args.images
    ground_truth = args.ground_truth

    #load model from model file
    model, stride = load_model(args.model, mixed_precision)

    run_test(model, stride)
    run_inferences(images, model, stride, mixed_precision, output_results_dir, output_image_dir, ground_truth=ground_truth)


if __name__ == '__main__':
    main()