import flask
import json
import os
import logging
from yolov5_pred import YoloV5

os.environ['OMP_THEAD_NUM']='1'
#gpuID = int(os.environ['APP_WORKER_ID'])%2
gpuID = 1

app = flask.Flask(__name__)
app.clf = YoloV5(str(gpuID))

@app.route('/det', methods=['GET'])
def newdet():
    url = flask.request.args.get('url', type=str, default=None)
    input = {"url":url}
    output = app.clf.process(json.dumps(input))
    return output
