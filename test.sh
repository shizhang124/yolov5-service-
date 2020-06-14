#python test.py --weights weights/yolov5s.pt --data data/coco.yaml --img-size 640 --conf-thres 0.1 --device 0
#python test.py --weights weights/yolov5m.pt --data data/coco.yaml --img-size 640 --conf-thres 0.1 --device 1
#python test.py --weights weights/yolov5l.pt --data data/coco.yaml --img-size 640 --conf-thres 0.1 --device 0
#python test.py --weights weights/yolov5x.pt --data data/coco.yaml --img-size 640 --conf-thres 0.1 --device 1

python test.py --weights weights/yolov5s.pt --data data/coco.yaml --img-size 736 --conf-thres 0.001 --device 1
#python test.py --weights weights/yolov5m.pt --data data/coco.yaml --img-size 736 --conf-thres 0.001 --device 1
#python test.py --weights weights/yolov5l.pt --data data/coco.yaml --img-size 736 --conf-thres 0.001 --device 0
#python test.py --weights weights/yolov5x.pt --data data/coco.yaml --img-size 736 --conf-thres 0.001 --device 1
