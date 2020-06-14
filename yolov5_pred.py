import argparse

from utils.datasets import *
from utils.utils import *
#import cv2
#import numpy as np
import json
import time

class YoloV5(object):
    def __init__(self, gpuid='0'):
        #os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuid)
        self.device = torch_utils.select_device(gpuid)
        self.weights = 'weights/yolov5l.pt'
        self.img_size = 640
        self.half = False
        self.conf_thres = 0.4
        self.iou_thres = 0.5
        self.agnostic_nms = False
        self.augment = False
        self.classes = None

        self.model = torch.load(self.weights, map_location=self.device)['model']
        self.model.to(self.device).eval()
        self.names = self.model.names if hasattr(self.model, 'names') else self.model.modules.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]

    #dataset = LoadImages(source, img_size=imgsz)
    
    def letterbox(self, img, new_shape=(416, 416), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):                                               
        # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = new_shape
            ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)


    def process(self, input, output={}):
        input = json.loads(input)
        process_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        if "url" in input:
            url = input["url"]
            img_name = os.path.basename(url)

            output["url"] = url
            output["img_name"] = img_name
            output["yolov5"] = {"bboxes":[], "status":"ERROR", 'time':process_time}
        else:
            return
        
        #load img
        if os.path.exists(url):
            #print('load from locall')
            img0 = cv2.imread(url)
        else:#from network
            #print('load from network')
            img0 = cv2.imread(url)
        img_h, img_w  = img0.shape[:2]
        output['img_w'] = img_w
        output['img_h'] = img_h

        img = self.letterbox(img0, new_shape=self.img_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        with torch.no_grad():
            pred = self.model(img, augment=self.augment)[0]

        # to float
        if self.half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres,
                                   fast=True, classes=self.classes, agnostic=self.agnostic_nms)

        ## Apply Classifier
        #if classify:
        #    pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        rs = []
        for i, det in enumerate(pred):  # detections per image
            p, s = url, ''

            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, self.names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    xyxy = [int(t) for t in (torch.tensor(xyxy).view(1, 4)).view(-1).tolist()]  # xyxy
                    #xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    conf = round(float(conf), 4)
                    cls = int(cls)
                    name = self.names[int(cls)]
                    #print(xyxy, ' ', conf, ' ', cls, ' ', name)
                    t_dict = {'bbox_xyxy':xyxy, 'prob':conf, 'cls_id':str(cls), 'cls_name':name}
                    rs.append(t_dict)
        output['yolov5']['bboxes'] = rs
        output['yolov5']['status'] = 'OK'
        return json.dumps(output)


if __name__ == '__main__':
    model = YoloV5('1')
    input = {'url':'/home/tang/linux/code/detection/yolov5/inference/images/bus.jpg'}
    input = json.dumps(input)
    output = model.process(input)
    print(output)
