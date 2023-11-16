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
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
from torchvision import transforms as T
from torchvision.utils import make_grid
import torch
import tkinter as tk
import tkinter.filedialog as filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import serial

CONF_THRESH = 0.5
IOU_THRESHOLD = 0.4

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
    color = color or [128 for _ in range(3)]
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

class GanDep(object):
    def __init__(self, engine_file_path):
        # Create a Context on this device,
        self.ctx = cuda.Device(0).make_context()
        self.stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        G_LOGGER = trt.Logger(trt.Logger.WARNING)
        # 反序列化引擎
        with open(engine_file_path, "rb") as f, trt.Runtime(G_LOGGER) as runtime:
            self.gan_engine = runtime.deserialize_cuda_engine(f.read())
        self.gan_context = self.gan_engine.create_execution_context()

        self.gan_d_input = cuda.mem_alloc(4 * 3 * 480 * 640)
        self.gan_d_output = cuda.mem_alloc(4 * 3 * 480 * 640)
        self.gan_output = np.empty((1, 3, 480, 640), dtype=np.float32)
        self.gan_bindings = [int(self.gan_d_input), int(self.gan_d_output)]
        
    def infer(self, raw_image):
        # Make self the active context, pushing it on top of the context stack.
        self.ctx.push()
        # Restore
        # Do image preprocess
        image_gan = self.preprocess_image(raw_image)

        cuda.memcpy_htod_async(self.gan_d_input, image_gan, self.stream)
        # Execute model
        self.gan_context.execute_async(1, bindings=self.gan_bindings,stream_handle= self.stream.handle)
        # Transfer predictions backcc
        cuda.memcpy_dtoh_async(self.gan_output, self.gan_d_output, self.stream)
        self.stream.synchronize()
        # Remove any context from the top of the context stack, deactivating it.
        gan_outimg = self.postprocess_gan(self.gan_output)
        return gan_outimg

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()

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
        image_gan = self.Loader(cv2.cvtColor(raw_bgr_image, cv2.COLOR_BGR2RGB))
        return image_gan

    def Loader(self, image_RGB):
        transform=T.Compose([T.ToTensor(), T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
        image_RGB = transform(image_RGB)
        return image_RGB.unsqueeze(0).numpy().astype(np.float32)

    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def postprocess_gan(self, clean_fake1):
        clean_fake1 = torch.as_tensor(clean_fake1)
        img_tensor = self.denorm(clean_fake1)
        grid = make_grid(img_tensor, nrow=1, padding=0)
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        # print(type(ndarr))
        img = cv2.cvtColor(ndarr, cv2.COLOR_RGB2BGR)
        # print(type(img))
        return img 
    

class YoloDep(object):
    """
    description: A YOLOv5 class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, engine_file_path):
        # Create a Context on this device,
        self.ctx = cuda.Device(0).make_context()
        self.stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        G_LOGGER = trt.Logger(trt.Logger.WARNING)
        # 反序列化引擎
        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.bindings = []

        for binding in self.engine:
            print('bingding:', binding, self.engine.get_binding_shape(binding))
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            self.bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                self.input_w = self.engine.get_binding_shape(binding)[-1]
                self.input_h = self.engine.get_binding_shape(binding)[-2]
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)
        self.batch_size = self.engine.max_batch_size

    def infer(self, raw_image):
        # threading.Thread.__init__(self)
        # Make self the active context, pushing it on top of the context stack.
        self.ctx.push()
        # Do image preprocess
        input_image, origin_h, origin_w = self.preprocess_image(raw_image)
        # Copy input image to host buffer
        np.copyto(self.host_inputs[0], input_image.ravel())
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
        # Run inference.
        self.context.execute_async(batch_size=self.batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)

        self.stream.synchronize()
        # Remove any context from the top of the context stack, deactivating it.

        # Do postprocess
        return self.post_process(
            self.host_outputs[0][0 * 6001: (0 + 1) * 6001], origin_h,origin_w
        )
    
    
    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()

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
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
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
        return image, h, w


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
        pred = np.reshape(output[1:], (-1, 6))[:num, :]
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


def Motor_level_set(delta):
    global level_angle, vertical_angle, ser
    level_angle = delta
    ser.write(("@" + str(level_angle).zfill(3) + str(vertical_angle).zfill(3) + "\r\n").encode())
    
    
def Motor_vertical_set(delta):
    global level_angle, vertical_angle, ser
    vertical_angle = delta
    ser.write(("@" + str(level_angle).zfill(3) + str(vertical_angle).zfill(3) + "\r\n").encode())
    
def is_detect_set():
    global is_detect
    is_detect = not is_detect

def is_enhance_set():
    global is_enhance
    is_enhance = not is_enhance

def is_stop_set():
    global is_stop
    is_stop = not is_stop

def interface():
    # 界面设置
    global main_window, is_detect, is_enhance, is_stop
    #------------- 选项按钮
    cb1 = tk.Checkbutton(main_window, text='进行识别检测', command=is_detect_set)
    cb2 = tk.Checkbutton(main_window, text="进行图像增强", command=is_enhance_set)
    cb3 = tk.Checkbutton(main_window, text="暂停", command=is_stop_set)

    #-------------- 舵机控制滑条
    level_sc = tk.Scale(main_window, label='level_set', from_=0, to=270,
                  orient=tk.HORIZONTAL, length=200,
                  showvalue=True, tickinterval=90, resolution=0.1,
                  command=Motor_level_set)
    vertical_sc = tk.Scale(main_window, label='vertical_set', from_=0, to=270,
                        orient=tk.VERTICAL, length=200,
                        showvalue=True, tickinterval=90, resolution=0.1,
                        command=Motor_vertical_set)
    level_sc.set(0)
    vertical_sc.set(0)
    cb1.pack()
    cb2.pack()
    cb3.pack()
    level_sc.pack()
    vertical_sc.pack()


def Enhance():
    global SpiralGan, is_enhance, is_detect, fin_img, cap, is_stop
    window = cv2.namedWindow("fin_img", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("fin_img", 640, 480)
    while(cap.isOpened()):
        _, ori_img = cap.read()
        start = time.time()
        # print(cap.isOpened())
        # print("is_stop:{}, is_enhance:{}, is_detect:{}".format(is_stop, is_enhance, is_detect))
        if is_stop:
            continue
        if is_enhance:
            fin_img = SpiralGan.infer(ori_img)
        else:
            fin_img = ori_img
        if not is_detect:
            cv2.imshow('fin_img', fin_img)
        # print(time.time() - start)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('stop_dispose')
            break
    cap.release()

def Detect():
    global YoloDet, is_detect, ori_img, fin_img, categories, is_stop, cap
    while(cap.isOpened()):
        if is_stop:
            continue
        if is_detect:
            result_boxes, result_scores, result_classid = YoloDet.infer(ori_img)
            # Draw rectangles and labels on the original image
            for j in range(len(result_boxes)):
                box = result_boxes[j]
                plot_one_box(
                    box,
                    fin_img,
                    label="{}:{:.2f}".format(
                        categories[int(result_classid[j])], result_scores[j]
                    ),
                )
            cv2.imshow('fin_img', fin_img)

def save():
    
    pass

# def Motor_Control(level_angle, vertical_angle):
    
    


if __name__ == "__main__":
    # load custom plugin and engine
    PLUGIN_LIBRARY = "build0608/libmyplugins.so"
    yolo_engine_file_path = "build0608/best0608.engine"#"build/yolov5s.engine"
    gan_engine_file_path = "/home/spring/Desktop/interface/interface/yolov5_with_trt_gan/torch33_1.engine"
    if len(sys.argv) > 1:
        engine_file_path = sys.argv[1]
    if len(sys.argv) > 2:
        PLUGIN_LIBRARY = sys.argv[2]

    ctypes.CDLL(PLUGIN_LIBRARY)

    # load coco labels
    categories = ['holothurian']
    # model set
    YoloDet = YoloDep(yolo_engine_file_path)
    SpiralGan = GanDep(gan_engine_file_path)
    # capvideo set
    # cap = cv2.VideoCapture('/home/spring/Desktop/56.avi')
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) #'''
    
    # global variable
    fin_img = np.zeros((480, 640, 3), dtype=np.uint8)
    ori_img = np.zeros((480, 640, 3), dtype=np.uint8)
    main_window = tk.Tk()  # 创建实例化对象
    main_window.title('This is main window!')
    main_window.geometry("500x500") # 1024x600
    save_file_type = tk.StringVar()
    save_file_type.set((".jpg", ".png"))
    is_detect = False
    is_enhance = False
    is_stop = False
    # serial set
    ser = serial.Serial("/dev/ttyTHS1", 9600)
    level_angle, vertical_angle = 0, 0
    # thread set
    # interface()
    thread1 = threading.Thread(target=Enhance)
    thread2 = threading.Thread(target=Detect)
    thread3 = threading.Thread(target=interface)
    thread1.start()
    thread2.start()
    thread3.start()
    thread2.join()
    # main_window.mainloop()
    cap.release()
    YoloDet.destroy()
    SpiralGan.destroy()
    cv2.destroyAllWindows()
