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
        # Store
        # self.stream = stream
        # self.gan_context = gan_context
        # self.gan_engine = gan_engine
        # self.gan_bindings = gan_bindings
        # self.gan_d_output = gan_d_output
        # self.gan_d_input = gan_d_input
        # self.gan_output = gan_output
        # self.ctx.pop()
        
    def infer(self, raw_image):
        # threading.Thread.__init__(self)
        # Make self the active context, pushing it on top of the context stack.
        self.ctx.push()
        # Restore
        # stream = self.stream
        # Do image preprocess
        image_gan = self.preprocess_image(raw_image)

        # stream1 = cuda.Stream()
        # start = time.time()
        cuda.memcpy_htod_async(self.gan_d_input, image_gan, self.stream)
        # Execute model
        self.gan_context.execute_async(1, bindings=self.gan_bindings,stream_handle= self.stream.handle)
        # Transfer predictions backcc
        cuda.memcpy_dtoh_async(self.gan_output, self.gan_d_output, self.stream)
        self.stream.synchronize()

        end = time.time()
        # Remove any context from the top of the context stack, deactivating it.
        gan_outimg = self.postprocess_gan(self.gan_output)
        end = time.time()
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
        # image_raw = raw_bgr_image
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


def show_ori_img():
    global ori_img, main_window
    img_show = Image.fromarray(cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB))
    render = ImageTk.PhotoImage(img_show)
    img = tk.Label(main_window, image=render)
    img.image = render
    img.place(x=20, y=200)


def show_fin_img():
    global fin_img, main_window
    img_show = Image.fromarray(cv2.cvtColor(fin_img, cv2.COLOR_BGR2RGB))
    render = ImageTk.PhotoImage(img_show)
    img = tk.Label(main_window, image=render)
    img.image = render
    img.place(x=876, y=200)


def Motor_level_set(delta):
    global level_angle, vertical_angle, ser
    level_angle = delta
    ser.write(("@" + str(level_angle).zfill(3) + str(vertical_angle).zfill(3) + "\r\n").encode())
    
    
def Motor_vertical_set(delta):
    global level_angle, vertical_angle, ser
    vertical_angle = delta
    ser.write(("@" + str(level_angle).zfill(3) + str(vertical_angle).zfill(3) + "\r\n").encode())
    

def interface(): # 界面设置
    global main_window, is_detect, is_enhance, is_stop
    #------------- 选项按钮
    cb1 = tk.Checkbutton(main_window, text='进行识别检测',variable=is_detect,
                        onvalue=1,offvalue=0,
                        command=None)
    cb2 = tk.Checkbutton(main_window, text="进行图像增强",variable=is_enhance,
                        onvalue=1,offvalue=0,
                        command=None)
    cb3 = tk.Checkbutton(main_window, text="暂停",variable=is_stop,
                        onvalue=1,offvalue=0,
                        command=None)

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


def Depose():
    global YoloDet, SpiralGan, is_enhance, is_detect, ori_img, fin_img, categories, cap, event, main_window
    while(cap.isOpened()):
        _, ori_img = cap.read()
        # if is_stop.get():
        #     continue # '''
        # if is_enhance.get():
        print(1)
        fin_img = SpiralGan.infer(ori_img)
        cv2.imshow('1', cv2.UMat(fin_img))
        print(2)
        if cv2.waitKey(30) & 0xFF == ord('q'):
                print('stop_dispose')
                break # '''
    cap.release()

def Dep():
    global YoloDet, SpiralGan, is_enhance, is_detect, ori_img, fin_img, categories, cap, event, main_window
    # while(cap.isOpened()):
    _, ori_img = cap.read()
    # if is_stop.get():
    #     continue # '''
    # if is_enhance.get():
    fin_img = SpiralGan.infer(ori_img)
    show_fin_img()
    main_window.after(30, Dep)
    '''if cv2.waitKey(1) & 0xFF == ord('q'):
                print('stop_dispose')
                break # '''
    # cap.release()

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
    # YoloDet = YoloDep(yolo_engine_file_path)
    SpiralGan = GanDep(gan_engine_file_path)

    # YoloDet = 1
    # SpiralGan = 2

    cap = cv2.VideoCapture('/home/spring/Desktop/6.avi')
    '''cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)# '''
    event = threading.Event()
    # global variable
    main_window = tk.Tk()  # 创建实例化对象
    main_window.title('This is main window!')
    main_window.geometry("1536x1080") # 1024x600
    width, height = 640, 480
    var_flip = tk.IntVar()
    tx, ty = 0, 0
    save_file_type = tk.StringVar()
    save_file_type.set((".jpg", ".png"))
    is_detect = tk.IntVar()
    is_enhance = tk.IntVar()
    is_stop = tk.IntVar()
    ori_img = np.zeros((480, 640, 3), dtype=np.uint8)
    fin_img = np.zeros((480, 640, 3), dtype=np.uint8)
    #-----------------------------------------------#
    ser = serial.Serial("/dev/ttyTHS1", 9600)
    level_angle, vertical_angle = 0, 0
    #-----------------------------------------------#
    # cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)   # 接收来自摄像头的数据
    # thread1 = threading.Thread(target=Depose)
    # thread1.start()
    interface()
    Depose()
    # main_window.after(30, Dep)
    # main_window.mainloop()
    cap.release()
    cv2.destroyAllWindows()
