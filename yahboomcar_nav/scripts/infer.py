#!/usr/bin/env python3
# -*-coding: UTF-8 -*
"""
An example that uses TensorRT's Python api to make inferences.
"""
import ctypes
import os
import shutil
import random
import sys
import threading
import time

import re
import requests
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import rospy
from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Int32,Bool,String
from sensor_msgs.msg import Image

import sys
import json
import base64
import time
import pyaudio
import wave

IS_PY3 = sys.version_info.major == 3

if IS_PY3:
    from urllib.request import urlopen
    from urllib.request import Request
    from urllib.error import URLError
    from urllib.parse import urlencode
    timer = time.perf_counter
else:
    from urllib2 import urlopen
    from urllib2 import Request
    from urllib2 import URLError
    from urllib import urlencode
    if sys.platform == "win32":
        timer = time.clock
    else:
        # On most other platforms the best timer is time.time()
        timer = time.time

API_KEY = 'z5fFSSWPpAp5jScWZUhLgua1'
SECRET_KEY = '5AKkaHMXIDyAXkGEMOxxTddtKsWnIOxs'


AUDIO_FILE = './output.wav'  

FORMAT = AUDIO_FILE[-3:]  

CUID = '123456PYTHON'

RATE = 16000  



DEV_PID = 1537 
ASR_URL = 'http://vop.baidu.com/server_api'
SCOPE = 'audio_voice_assistant_get' 


CONF_THRESH = 0.5
IOU_THRESHOLD = 0.4
LEN_ALL_RESULT = 38001
LEN_ONE_RESULT = 38
pubDest = 0
pubLabel = 0
destID = -1
mode = -1
categories = []
last = 0
msgs_from_ui = ""
stop = False


class DemoError(Exception):
    pass


"""  TOKEN start """

TOKEN_URL = 'http://aip.baidubce.com/oauth/2.0/token'


def fetch_token():
    params = {'grant_type': 'client_credentials',
              'client_id': API_KEY,
              'client_secret': SECRET_KEY}
    post_data = urlencode(params)
    if (IS_PY3):
        post_data = post_data.encode( 'utf-8')
    req = Request(TOKEN_URL, post_data)
    try:
        f = urlopen(req)
        result_str = f.read()
    except URLError as err:
        print('token http response http code : ' + str(err.code))
        result_str = err.read()
    if (IS_PY3):
        result_str =  result_str.decode()

    print(result_str)
    result = json.loads(result_str)
    print(result)
    if ('access_token' in result.keys() and 'scope' in result.keys()):
        print(SCOPE)
        if SCOPE and (not SCOPE in result['scope'].split(' ')): 
            raise DemoError('scope is not correct')
        print('SUCCESS WITH TOKEN: %s  EXPIRES IN SECONDS: %s' % (result['access_token'], result['expires_in']))
        return result['access_token']
    else:
        raise DemoError('MAYBE API_KEY or SECRET_KEY not correct: access_token or scope not found in token response')

"""  TOKEN end """
def record(time):  
  
    CHUNK = 1024 
    FORMAT = pyaudio.paInt16  
    CHANNELS = 1 
    RATE = 16000 
    RECORD_SECONDS = time  
    WAVE_OUTPUT_FILENAME = AUDIO_FILE 
    device_index = 11
    p = pyaudio.PyAudio()  
    stream = p.open(format=FORMAT, 
                    channels=CHANNELS,  
                    rate=RATE,
                    input=True, 
                    frames_per_buffer=CHUNK,
                    input_device_index = device_index)  
    print("* recording")  
    frames = []  
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)): 
        data = stream.read(CHUNK) 
        frames.append(data)  
    print("* done recording")
    stream.stop_stream()  
    stream.close()  
    p.terminate() 

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')  
    wf.setnchannels(CHANNELS)  
    wf.setsampwidth(p.get_sample_size(FORMAT)) 
    wf.setframerate(RATE) 
    wf.writeframes(b''.join(frames)) 
    wf.close() 

def asr():
    speech_data = []
    with open(AUDIO_FILE, 'rb') as speech_file:
        speech_data = speech_file.read()

    length = len(speech_data)
    if length == 0:
        raise DemoError('file %s length read 0 bytes' % AUDIO_FILE)
    speech = base64.b64encode(speech_data)
    if (IS_PY3):
        speech = str(speech, 'utf-8')
    params = {'dev_pid': DEV_PID,
             #"lm_id" : LM_ID, 
              'format': FORMAT,
              'rate': RATE,
              'token': token,
              'cuid': CUID,
              'channel': 1,
              'speech': speech,
              'len': length
              }
    post_data = json.dumps(params, sort_keys=False)
    # print post_data
    req = Request(ASR_URL, post_data.encode('utf-8'))
    req.add_header('Content-Type', 'application/json')
    try:
        begin = timer()
        f = urlopen(req)
        result_str = f.read()
        print ("Request time cost %f" % (timer() - begin))
    except URLError as err:
        print('asr http response http code : ' + str(err.code))
        result_str = err.read()

    if (IS_PY3):
        result_str = str(result_str, 'utf-8')
    result_dict = json.loads(result_str)  

    result_list = result_dict.get("result", [])
    #print(result_list)  
    return  result_list[0]



def get_img_path_batches(batch_size, img_dir):
    ret = []
    batch = []
    for root, dirs, files in os.walk(img_dir):
        for name in files:
            if len(batch) == batch_size:
                ret.append(batch)
                batch = []
            batch.append(os.path.join(root, name))
    if len(batch) > 0:
        ret.append(batch)
    return ret

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param: 
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return

    """
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


class YoLov5TRT(object):
    """
    description: A YOLOv5 class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, engine_file_path):
        # Create a Context on this device,
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            print('bingding:', binding, engine.get_binding_shape(binding))
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = engine.max_batch_size

    def infer(self, raw_image_generator):
        global destID
        global mode
        global last
        #
        threading.Thread.__init__(self)
        # Make self the active context, pushing it on top of the context stack.
        self.ctx.push()
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        # Do image preprocess
        batch_image_raw = []
        batch_origin_h = []
        batch_origin_w = []
        batch_input_image = np.empty(shape=[self.batch_size, 3, self.input_h, self.input_w])
        for i, image_raw in enumerate(raw_image_generator):
            input_image, image_raw, origin_h, origin_w = self.preprocess_image(image_raw)
            batch_image_raw.append(image_raw)
            batch_origin_h.append(origin_h)
            batch_origin_w.append(origin_w)
            np.copyto(batch_input_image[i], input_image)
        batch_input_image = np.ascontiguousarray(batch_input_image)

        # Copy input image to host buffer
        np.copyto(host_inputs[0], batch_input_image.ravel())
        start = time.time()
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.execute_async(batch_size=self.batch_size, bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        # Synchronize the stream
        stream.synchronize()
        end = time.time()
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        # Here we use the first row of output in that batch_size = 1
        output = host_outputs[0]
        # Do postprocess
       
        for i in range(self.batch_size):
            result_boxes, result_scores, result_classid = self.post_process(
                output[i * LEN_ALL_RESULT: (i + 1) * LEN_ALL_RESULT], batch_origin_h[i], batch_origin_w[i]
            )
            
        if mode == 0 :
            if len(result_boxes) == 1:
              destID = result_classid[0]
          
              box = result_boxes[0]
              plot_one_box(
                    box,
                    batch_image_raw[i],
                    label = "Target:({})".format(categories[int(destID)]),
                )
            else:
              return batch_image_raw, end - start
         
        if mode == 1:
                '''
                if len(result_boxes) == 0:
                        current = time.time()
                        if(current-last >=3):
                            last = current
                        
                        
                            array_msg = Float32MultiArray()
                            array_msg.data.append(-1)
                            array_msg.data.append(-1)
                            array_msg.data.append(-1)
                            array_msg.data.append(-1)
                            array_msg.data.append(-1)
                            pub_label.publish(array_msg)
                            #time.sleep(3)
                '''
                flag_find = False
                i = 0
                min_err = (batch_origin_h[i]**2 + batch_origin_w[i]**2)**(1/2)
                min_array_msg = Float32MultiArray()
                for j in range(len(result_classid)):
                    if result_classid[j] == destID:
                        box = result_boxes[j]
                        center_x = (box[0] + box[2]) / 2
                        center_y = (box[1] + box[3]) / 2
                        array_msg = Float32MultiArray()
                        array_msg.data.append(center_x)
                        array_msg.data.append(center_y)
                        array_msg.data.append(batch_origin_h[i])
                        array_msg.data.append(batch_origin_w[i])
                        array_msg.data.append(result_classid[j])

                        
                        plot_one_box(
                            box,
                            batch_image_raw[i],
                            label="Target:({}, {}, {})".format(categories[int(destID)],center_x, center_y),
                        )
                        flag_find = True
                        if ((center_x - batch_origin_w[i]/2)**2 + (center_y - batch_origin_h[i]/2)**2)**(1/2) <min_err:
                            min_err = ((center_x - batch_origin_w[i]/2)**2 + (center_y - batch_origin_h[i]/2)**2)**(1/2)
                            min_array_msg = array_msg
                       #  break 
                if(flag_find is False):
                        current = time.time()
                        if(current-last >=3):
                            last = current
                        
                            array_msg = Float32MultiArray()
                            array_msg.data.append(-1)
                            array_msg.data.append(-1)
                            array_msg.data.append(-1)
                            array_msg.data.append(-1)
                            array_msg.data.append(-1)
                            pub_label.publish(array_msg)
                else:
                    pub_label.publish(min_array_msg)

        return batch_image_raw, end - start

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        
    def get_raw_image(self, image_path_batch):
        """
        description: Read an image from image path
        """
        for img_path in image_path_batch:
            yield cv2.imread(img_path)
        
    def get_raw_image_zeros(self, image_path_batch=None):
        """
        description: Ready data for warmup
        """
        for _ in range(self.batch_size):
            yield np.zeros([self.input_h, self.input_w, 3], dtype=np.uint8)

    def preprocess_image(self, raw_bgr_image):
        """
        description: Convert BGR image to RGB,
                     resize and pad it to target size, normalize to [0,1],
                     transform to NCHW format.
        param:
            input_image_path: str, image path
        return:
            image:  the processed image
            image_raw: the original image
            h: original height
            w: original width
        """
        image_raw = raw_bgr_image
        h, w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        # Calculate widht and height and paddings
        r_w = self.input_w / w
        r_h = self.input_h / h
        if r_h > r_w:
            tw = self.input_w
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((self.input_h - th) / 2)
            ty2 = self.input_h - th - ty1
        else:
            tw = int(r_h * w)
            th = self.input_h
            tx1 = int((self.input_w - tw) / 2)
            tx2 = self.input_w - tw - tx1
            ty1 = ty2 = 0
        # Resize the image with long side while maintaining ratio
        image = cv2.resize(image, (tw, th))
        # Pad the short side with (128,128,128)
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, None, (128, 128, 128)
        )
        image = image.astype(np.float32)
        # Normalize to [0,1]
        image /= 255.0
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w

    def xywh2xyxy(self, origin_h, origin_w, x):
        """
        description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        param:
            origin_h:   height of original image
            origin_w:   width of original image
            x:          A boxes numpy, each row is a box [center_x, center_y, w, h]
        return:
            y:          A boxes numpy, each row is a box [x1, y1, x2, y2]
        """
        y = np.zeros_like(x)
        r_w = self.input_w / origin_w
        r_h = self.input_h / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h

        return y

    def post_process(self, output, origin_h, origin_w):
        """
        description: postprocess the prediction
        param:
            output:     A numpy likes [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...] 
            origin_h:   height of original image
            origin_w:   width of original image
        return:
            result_boxes: finally boxes, a boxes numpy, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a numpy, each element is the score correspoing to box
            result_classid: finally classid, a numpy, each element is the classid correspoing to box
        """
        # Get the num of boxes detected
        num = int(output[0])
        # Reshape to a two dimentional ndarray
        pred = np.reshape(output[1:], (-1, LEN_ONE_RESULT))[:num, :]
        pred = pred[:, :6]
        # Do nms
        boxes = self.non_max_suppression(pred, origin_h, origin_w, conf_thres=CONF_THRESH, nms_thres=IOU_THRESHOLD)
        result_boxes = boxes[:, :4] if len(boxes) else np.array([])
        result_scores = boxes[:, 4] if len(boxes) else np.array([])
        result_classid = boxes[:, 5] if len(boxes) else np.array([])
        return result_boxes, result_scores, result_classid

    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        """
        description: compute the IoU of two bounding boxes
        param:
            box1: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))
            box2: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))            
            x1y1x2y2: select the coordinate format
        return:
            iou: computed iou
        """
        if not x1y1x2y2:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        # Get the coordinates of the intersection rectangle
        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)
        # Intersection area
        inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * \
                     np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        return iou

    def non_max_suppression(self, prediction, origin_h, origin_w, conf_thres=0.5, nms_thres=0.4):
        """
        description: Removes detections with lower object confidence score than 'conf_thres' and performs
        Non-Maximum Suppression to further filter detections.
        param:
            prediction: detections, (x1, y1, x2, y2, conf, cls_id)
            origin_h: original image height
            origin_w: original image width
            conf_thres: a confidence threshold to filter detections
            nms_thres: a iou threshold to filter detections
        return:
            boxes: output after nms with the shape (x1, y1, x2, y2, conf, cls_id)
        """
        # Get the boxes that score > CONF_THRESH
        boxes = prediction[prediction[:, 4] >= conf_thres]
        # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
        boxes[:, :4] = self.xywh2xyxy(origin_h, origin_w, boxes[:, :4])
        # clip the coordinates
        boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w -1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w -1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h -1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h -1)
        # Object confidence
        confs = boxes[:, 4]
        # Sort by the confs
        boxes = boxes[np.argsort(-confs)]
        # Perform non-maximum suppression
        keep_boxes = []
        while boxes.shape[0]:
            large_overlap = self.bbox_iou(np.expand_dims(boxes[0, :4], 0), boxes[:, :4]) > nms_thres
            label_match = boxes[0, -1] == boxes[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            keep_boxes += [boxes[0]]
            boxes = boxes[~invalid]
        boxes = np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])
        return boxes


class inferThread(threading.Thread):
    def __init__(self, yolov5_wrapper, image_path_batch):
        threading.Thread.__init__(self)
        self.yolov5_wrapper = yolov5_wrapper
        self.image_path_batch = image_path_batch

    def run(self):
        batch_image_raw, use_time = self.yolov5_wrapper.infer(self.yolov5_wrapper.get_raw_image(self.image_path_batch))
        for i, img_path in enumerate(self.image_path_batch):
            parent, filename = os.path.split(img_path)
            save_name = os.path.join('output', filename)
            # Save image
            cv2.imwrite(save_name, batch_image_raw[i])
        print('input->{}, time->{:.2f}ms, saving into output/'.format(self.image_path_batch, use_time * 1000))


class warmUpThread(threading.Thread):
    def __init__(self, yolov5_wrapper):
        threading.Thread.__init__(self)
        self.yolov5_wrapper = yolov5_wrapper

    def run(self):
        batch_image_raw, use_time = self.yolov5_wrapper.infer(self.yolov5_wrapper.get_raw_image_zeros())
        print('warm_up->{}, time->{:.2f}ms'.format(batch_image_raw[0].shape, use_time * 1000))

def parse_response(text):
    matches = re.findall(r'-?\d+', text)
    for match in matches:
        num = int(match)
        if num in {-1, 0, 1, 2, 3, 4}:
            return num
    return -1

def get_product_number(description):
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    print(system_prompt)
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": description}
        ],
        "temperature": 0,
        "max_tokens": 1  # limit length
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            reply = response.json()['choices'][0]['message']['content'].strip()
            return parse_response(reply)
        print(f"API request fail! resuest code: {response.status_code}")
        return None
    except Exception as e:
        print(f"request fail!:{str(e)}")
        return None
def cv2_to_imgmsg(cv_image):
    img_msg = Image()
    img_msg.height = cv_image.shape[0]
    img_msg.width = cv_image.shape[1]
    img_msg.encoding = "bgr8"
    img_msg.is_bigendian = 0
    img_msg.data = cv_image.tostring()
    img_msg.step = len(img_msg.data) // img_msg.height # That double line is actually integer division, not a comment
    return img_msg

def mode_callback(msg):
    global mode
    global stop
    global msgs_from_ui  
    global destID
    break_flag = False
        # Car motion control, subscriber callback function
    if not isinstance(msg, Int32): return
      
        # Issue linear vel and angular vel
    if msg.data==1:
        mode = 1
        yolov5_wrapper = YoLov5TRT(engine_file_path)
        fps = 0
        st = time.time()
        fc = 0
        # cap = cv2.VideoCapture("/home/ydr/tensorrtx-master/yolov/home/ydr/workspace/rd_wa/src/yahboomcar_nav/scripts/infer.py5/WeChat.mp4")
        # input("Press Enter to open the camera...")
        cap = cv2.VideoCapture(0)
        # cap.set(3,960)
        # cap.set(4,540)
        if cap.isOpened():
            rospy.loginfo("Camera Open")
            print("Camera Open")
        while (cap.isOpened() and not (stop)):
            _, frame = cap.read()
            current = time.time
            img = frame
            # print(len(img))
            batch_image_raw, use_time = yolov5_wrapper.infer([frame])
            
            #img = batch_image_raw[0]
            img = cv2.resize(batch_image_raw[0],(256,256))
            fc += 1
            dt = time.time() - st
            if dt >= 1:
                st = time.time()
                fps = fc / dt
                fc = 0
            h, w, _ = img.shape
            cv2.putText(img, f"FPS: {fps:.2f}  {w}X{h}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("1", img)
            
            cv2.waitKey(1)
        stop = False
        cap.release()
        cv2.destroyAllWindows()
        yolov5_wrapper.destroy()

    elif msg.data==0:
        mode = 0
        yolov5_wrapper = YoLov5TRT(engine_file_path)

        fps=0
        st = time.time()
        fc=0
    #cap = cv2.VideoCapture("/home/ydr/tensorrtx-master/yolov/home/ydr/workspace/rd_wa/src/yahboomcar_nav/scripts/infer.py5/WeChat.mp4")
    #input("Press Enter to open the camera...")
        cap = cv2.VideoCapture(0)
    #cap.set(3,960)
    #cap.set(4,540)
        if cap.isOpened():
            print("Camera Open")
            rospy.loginfo("Camera Open")
            
        # input("Press Enter to set your goal...") 
        while cap.isOpened():
            _,frame = cap.read()
            current = time.time
            img = frame
        #print(len(img))
            batch_image_raw, use_time = yolov5_wrapper.infer([frame])
            img = cv2.resize(batch_image_raw[0],(256,256))
            fc+=1
            dt = time.time() - st
            if dt>=1:
                st = time.time()
                fps = fc/dt
                fc=0
            h,w,_ = img.shape
            cv2.putText(img,f"FPS: {fps:.2f}",(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            cv2.imshow("1",img)
            processed_msg = cv2_to_imgmsg(img)
            pub_image.publish(processed_msg)
            if (cv2.waitKey(1) and True) and (msgs_from_ui == "q"):
                break
            if (cv2.waitKey(1) and True) and (msgs_from_ui == "s"):
                msgs_from_ui = ""
                rospy.loginfo("Ensure your goal... (y=yes, n=no, q=quit)")
                print("Ensure your goal... (y=yes, n=no, q=quit)")



                while True:
                    ans = msgs_from_ui
                    if ans == "":
                        time.sleep(0.1)
                    else:
                        if ans == 'y':
                            dest = Float32()
                            dest.data = destID
                            pub_destIndex.publish(dest)
                            print("Goal established")
                            rospy.loginfo("Goal established")
                            break_flag = True
                            break
                        elif ans == 'n':
                            break 
                        elif ans == 'q':
                            break_flag = True
                            break
                        else:
                            print("invalid input, again!")
                        
                if break_flag:
                    break
                    
        cap.release()
        cv2.destroyAllWindows()
        yolov5_wrapper.destroy()
    elif msg.data==2:
      mode = 2
      print("inquiring system start up! (press q to exit)")  
      while True:
        # input
        
        # user_input = input("\nplease enter the description of what you want to buy (press q to exit): ").strip()
        ''' 
        ui_mode = Int32()
        ui_mode.data = 1
        pub_mode2.publish(ui_mode)
        '''
        print("\nplease enter the description of what you want to buy (press q to exit): ") 
        user_input = ""
        while(True):
            if(msgs_from_ui == ""):
                time.sleep(0.1)
            elif msgs_from_ui == "q":
                break_flag = True
                break
            else:
                user_input = msgs_from_ui
                msgs_from_ui = ""
                break
        if(break_flag):
            break
        print(user_input)
        # get result
        result = None
        
        while result is None:
            result = get_product_number(user_input)
            if result is not None:
                break
        
        id_result = Int32()
        id_result.data = result
        pub_mode2.publish(id_result)
        print(f"get result: {result}")
        
        # user feedback
        print("Is this what you want to buy? (y=yes, n=no, q=quit)") 
        while True:
            # feedback = input("Is this what you want to buy? (y=yes, n=no, q=quit)").lower()
            if msgs_from_ui == "" :
                time.sleep(0.1)
            else:
                feedback = msgs_from_ui
                msgs_from_ui = ""
                if feedback == 'y':
                    dest =  Float32()
                    dest.data = float(result)
                    pub_destIndex.publish(dest)
                    break_flag = True
                    break 
                elif feedback == 'n':
                    break 
                elif feedback == 'q':
                    break_flag = True
                    break
                else:
                    print("invalid input, again!")
            
        if break_flag:
            break
        
        #
      print("exited!")
    elif msg.data==3:
      mode = 3
      print("voice inquiring system start up! (press q to exit)")  
      while True:
        # input
        
        # user_input = input("\nplease enter the description of what you want to buy (press q to exit): ").strip()
        ''' 
        ui_mode = Int32()
        ui_mode.data = 1
        pub_mode2.publish(ui_mode)
        '''
        print("\nplease speak the description of what you want to buy (press q to exit): ") 
        user_input = ""
        while(True):
            if(msgs_from_ui == ""):
                time.sleep(0.1)
            elif msgs_from_ui == "q":
                break_flag = True
                break
            elif msgs_from_ui == "say":
                msgs_from_ui = ""
                break
        if(break_flag):
            break
        record(10)
        user_input = asr()
        print(user_input)
        
        ans_result = String()
        ans_result.data = user_input
        pub_mode3.publish(ans_result)
        
        # get result
        result = None
        
        while result is None:
            result = get_product_number(user_input)
            if result is not None:
                break
        
        id_result = Int32()
        id_result.data = result
        pub_mode2.publish(id_result)
        print(f"get result: {result}")
        
        # user feedback
        print("Is this what you want to buy? (y=yes, n=no, q=quit)") 
        while True:
            # feedback = input("Is this what you want to buy? (y=yes, n=no, q=quit)").lower()
            if msgs_from_ui == "" :
                time.sleep(0.1)
            else:
                feedback = msgs_from_ui
                msgs_from_ui = ""
                if feedback == 'y':
                    dest =  Float32()
                    dest.data = float(result)
                    pub_destIndex.publish(dest)
                    break_flag = True
                    break 
                elif feedback == 'n':
                    break 
                elif feedback == 'q':
                    break_flag = True
                    break
                else:
                    print("invalid input, again!")
            
        if break_flag:
            break
        
        #
      print("exited!")

    msgs_from_ui = ""
    mode = -1 

def subStop(msg):
    global stop
    stop = msg.data

def sub_ui(msg):
    global msgs_from_ui
    msgs_from_ui = msg.data
    print()
    
if __name__ == "__main__":
    
    # load custom plugin and engine
    token = fetch_token()
    rospy.init_node('productInfo_publisher', anonymous=False)
    PLUGIN_LIBRARY = "/home/ydr/tensorrtx-yolov5-v7.0/yolov5/build/libmyplugins.so"
    engine_file_path = "/home/ydr/tensorrtx-yolov5-v7.0/yolov5/build/best.engine"
    ctypes.CDLL(PLUGIN_LIBRARY)
    
    # load coco labels
    DEEPSEEK_API_KEY = "sk-69d4739bf87d4b1599e62874867dab08"  
    API_URL = "https://api.deepseek.com/v1/chat/completions"
    system_prompt = '''
You are a product classification assistant. Based on the user's description, return the corresponding product number. The product list is as follows:
0: potato_chips_in_bag
1: friend_pie
2: hotpot
3: potato_chips_in_bottle
4: shimp_flakers
5: water
6: pocky

Please respond according to the following rules:
1. Directly return the corresponding number (0-6)
2. Return 1024 if there is no matching product
3. Do not include any extra content
4. user may say chinese, so you need to translate it into english first.

Examples:
User: a bag of potato chips
Answer: 0
User: I_want_to_eat_hotpot
Answer: 2
User: I am thirsty
Answer: 5
User: I want to eat pocky
Answer: 6
User: I need a bottle of potato chips
Answer: 3
User: friend pie is what I need to buy
Answer: 1
User: I want to eat vegetable
Answer: 1024
'''

    categories = ["leshi_bag", "friend pie", "hotpot", "leshi_bottle", "shimp_flakers", "water", "pocky"]

    sub_camera_mod = rospy.Subscriber('mode', Int32, mode_callback, queue_size=10)
    sub_stop = rospy.Subscriber("stop_mode", Bool, subStop, queue_size=1)
    sub_ui = rospy.Subscriber("sub_ui", String, sub_ui, queue_size=10)

    pub_image = rospy.Publisher("/image_publisher/processed_image",Image,queue_size = 10)
    pub_label = rospy.Publisher("label_location", Float32MultiArray, queue_size=100)
    pub_destIndex = rospy.Publisher("destination_index", Float32, queue_size=1)
    pub_mode2 = rospy.Publisher("ctrl_mode2", Int32, queue_size=1)
    pub_mode3 = rospy.Publisher("ctrl_mode3", String, queue_size=1)

    rospy.spin()
        
        #img = cv2.resize(tuple(frame),400,300)
       
'''
        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break
        ans = input("Press y/n to ensure your goal...")
        if ans == 'y':
            pub_label.publish(destID)
            print("Goal established")
            break
        elif ans == 'n':
            continue
        elif ans == 'q':
            break
        else:
            print("Invalid input")
            continue'''
    # while cap.isOpened():
    #     _,frame = cap.read()
    #     batch_image_raw, use_time = yolov5_wrapper.infer([frame])
    #     img = batch_image_raw[0]
    #     fc+=1
    #     dt = time.time() - st
    #     if dt>=1:
    #         st = time.time()
    #         fps = fc/dt
    #         fc=0
    #     h,w,_ = img.shape
    #     cv2.putText(img,f"FPS: {fps:.2f}  {w}X{h}",(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2 )
    #     cv2.imshow("1",img)
    #     cv2.waitKey(1)

