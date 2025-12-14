import tensorflow as tf
import numpy as np
from config import *
from PIL import Image
import cv2
import time
import os
import traceback
import math
import argparse
from scipy.ndimage import map_coordinates # Requires SciPy to be installed


parser = argparse.ArgumentParser()
parser.add_argument('--model-dir')
parser.add_argument('--model-name')
parser.add_argument('--before-ch', type=int)
#parser.add_argument('--after-ch', type=int)
parser.add_argument('--output-dir', default='data_video_local')
parser.add_argument('--infer-with-stable', action='store_true')
parser.add_argument('--infer-with-last', action='store_true')
parser.add_argument('--test-list', nargs='+', default=['data_video/test_list', 'data_video/train_list_deploy'])
#parser.add_argument('--train-list', default='data_video/train_list_deploy')
parser.add_argument('--prefix', default='data_video')
parser.add_argument('--max-span', type=int, default=1)
parser.add_argument('--random-black', type=int, default=None)
#parser.add_argument('--indices', type=int, nargs='+', required=True)
parser.add_argument('--start-with-stable', action='store_true')
parser.add_argument('--refine', type=int, default=1)
parser.add_argument('--no_bm', type=int, default=1)
parser.add_argument('--gpu_memory_fraction', type=float, default=0.1)
parser.add_argument('--deploy-vis', action='store_true')
args = parser.parse_args()

MaxSpan = args.max_span
args.indices = indices[1:]


sess = tf.Session(config=tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)))

model_dir = args.model_dir#'models/vbeta-1.1.0/'
model_name = args.model_name#'model-5000'
before_ch = max(args.indices)#args.before_ch
after_ch = max(1, -min(args.indices) + 1)
#after_ch = args.after_ch
#after_ch = 0
new_saver = tf.train.import_meta_graph(model_dir + model_name + '.meta')
new_saver.restore(sess, model_dir + model_name)
graph = tf.get_default_graph()
x_tensor = graph.get_tensor_by_name('stable_net/input/x_tensor:0')
#output = graph.get_tensor_by_name('stable_net/SpatialTransformer/_transform/Reshape_7:0')
#black_pix = graph.get_tensor_by_name('stable_net/SpatialTransformer/_transform/Reshape_6:0')
output = graph.get_tensor_by_name('stable_net/inference/SpatialTransformer/_transform/output_img:0')
black_pix = graph.get_tensor_by_name('stable_net/inference/SpatialTransformer/_transform/black_pix:0')
#theta_mat_tensor = graph.get_tensor_by_name('stable_net/feature_loss/Reshape:0')
Hs_tensor = graph.get_tensor_by_name('stable_net/inference/SpatialTransformer/_transform/get_Hs/Hs:0')
x_map = graph.get_tensor_by_name("stable_net/inference/SpatialTransformer/_transform/x_map:0")
y_map = graph.get_tensor_by_name("stable_net/inference/SpatialTransformer/_transform/y_map:0")


#black_pix = graph.get_tensor_by_name('stable_net/img_loss/StopGradient:0')

#list_f = open('data_video/test_list_deploy', 'r')

video_list = []

for list_path in args.test_list:
    if os.path.isfile(list_path):
        print('adding '+list_path)
        list_f = open(list_path, 'r')
        temp = list_f.read()
        video_list.extend(temp.split('\n'))

def make_dirs(path):
    if not os.path.exists(path): os.makedirs(path)

cvt_train2img = lambda x: ((np.reshape(x, (height, width)) + 0.5) * 255).astype(np.uint8)

def draw_imgs(net_output, stable_frame, unstable_frame, inputs):
    cvt2int32 = lambda x: x.astype(np.int32)
    assert(net_output.ndim == 2)
    assert(stable_frame.ndim == 2)
    assert(unstable_frame.ndim == 2)

    net_output = cvt2int32(net_output)
    stable_frame = cvt2int32(stable_frame)
    unstable_frame = cvt2int32(unstable_frame)
    last_frame = cvt2int32(cvt_train2img(inputs[..., 0]))
    output_minus_input  = abs(net_output - unstable_frame)
    output_minus_stable = abs(net_output - stable_frame)
    output_minus_last   = abs(net_output - last_frame)
    img_top    = np.concatenate([net_output,         output_minus_stable], axis=1)
    img_bottom = np.concatenate([output_minus_input, output_minus_last], axis=1)
    img = np.concatenate([img_top, img_bottom], axis=0).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

def getNext(delta, bound, speed = 5):
    tmp = delta + speed
    if tmp >= bound or tmp < 0: speed *= -1
    return delta + speed, speed
    # return np.random.randint(0, bound), 5

def cvt_theta_mat(theta_mat):
    # theta_mat * x = x'
    # ret * scale_mat * x = scale_mat * x'
    # ret = scale_mat * theta_mat * scale_mat^-1
    scale_mat = np.eye(3)
    scale_mat[0, 0] = width / 2.
    scale_mat[0, 2] = width / 2.
    scale_mat[1, 1] = height / 2.
    scale_mat[1, 2] = height / 2.
    assert(theta_mat.shape == (3, 3))
    from numpy.linalg import inv
    return np.matmul(np.matmul(scale_mat, theta_mat), inv(scale_mat))

def warpRev(img, theta):
    assert(img.ndim == 3)
    assert(img.shape[-1] == 3)
    theta_mat_cvt = cvt_theta_mat(theta)
    return cv2.warpPerspective(img, theta_mat_cvt, dsize=(width, height), flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR)


def cvt_theta_mat_bundle(Hs):
    # theta_mat * x = x'
    # ret * scale_mat * x = scale_mat * x'
    # ret = scale_mat * theta_mat * scale_mat^-1
    scale_mat = np.eye(3)
    scale_mat[0, 0] = width / 2.
    scale_mat[0, 2] = width / 2.
    scale_mat[1, 1] = height / 2.
    scale_mat[1, 2] = height / 2.

    Hs = Hs.reshape((grid_h, grid_w, 3, 3))
    from numpy.linalg import inv

    return np.matmul(np.matmul(scale_mat, Hs), inv(scale_mat))

def smooth_mapping(x_map, y_map):
    rate = 4
    x_map = cv2.resize(cv2.resize(x_map, (int(output_width / rate), int(output_height / rate))), (output_width, output_height))
    y_map = cv2.resize(cv2.resize(y_map, (int(output_width / rate), int(output_height / rate))), (output_width, output_height))
    x_map = (x_map + 1) / 2 * output_width
    y_map = (y_map + 1) / 2 * output_height
    return x_map, y_map

def warpRevBundle2(img, x_map, y_map):
    assert(img.ndim == 3)
    assert(img.shape[-1] == 3)
    
    dst = cv2.remap(img, x_map, y_map, cv2.INTER_LINEAR)
    assert(dst.shape == (output_height, output_width, 3))
    return dst

def warpRevBundle(img, Hs):
    assert(img.ndim == 3)
    assert(img.shape[-1] == 3)
    Hs_cvt = cvt_theta_mat_bundle(Hs)

    gh = int(math.floor(height / grid_h))
    gw = int(math.floor(width / grid_w))
    img_ = []
    for i in range(grid_h):
        row_img_ = []
        for j in range(grid_w):
            H = Hs_cvt[i, j, :, :]
            sh = i * gh
            eh = (i + 1) * gh - 1
            sw = j * gw
            ew = (j + 1) * gw - 1
            if (i == grid_h - 1):
                eh = height - 1
            if (j == grid_w - 1):
                ew = width - 1
            temp = cv2.warpPerspective(img, H, dsize=(width, height), flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR)
            row_img_.append(temp[sh:eh+1, sw:ew+1, :])
        img_.append(np.concatenate(row_img_, axis=1))
    img = np.concatenate(img_, axis=0)    
    assert(img.shape == (height, width, 3))
    return img




def get_bbox(mask):
    x1 = float("inf")
    y1 = float("inf")
    x2 = float("-inf")
    y2 = float("-inf")
    
    y_indexes, x_indexes = np.where(mask == 1)

    if len(x_indexes) == 0:
        # print("hi")
        return None
    
    x1 = float(min(x_indexes))
    y1 = float(min(y_indexes))
    x2 = float(max(x_indexes))
    y2 = float(max(y_indexes))

    return x1, y1, x2, y2


def remap(x1, y1, x2, y2, x_map, y_map):
    # TODO hcange to zeroes carefullly
    mask = np.zeros((output_height, output_width))
    # mask = mask.astype(np.uint8)
    
    # mask = cv2.resize(mask, (output_width, output_height))
    
    cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 1, -1)
    # display_resized(mask, width, height)

    # print(mask)
    remapped_mask = cv2.remap(mask, x_map, y_map, cv2.INTER_LINEAR)
    # print(remapped_mask)

    
    p1_p2 = get_bbox(remapped_mask)
    # print(p1_p2)
    
    return p1_p2



production_dir = os.path.join(args.output_dir, 'output')
make_dirs(production_dir)

# visual_dir = os.path.join(args.output_dir, 'output-vis')
# make_dirs(visual_dir)


from copy import deepcopy

class VideoStabilizer():
    def __init__(self, fps, before_ch, after_ch):
        self.fps = fps
        
        self.before_ch = before_ch
        self.after_ch = after_ch
        
        self.reset()

    def stabilization_start(self, frame_unstable, bboxs_coords):
        if self.before_ch_i < self.before_ch:
            while self.before_ch_i < self.before_ch:
                self.before_frames.append(cvt_img2train(frame_unstable, crop_rate))
                self.before_masks.append(np.zeros([1, height, width, 1], dtype=np.float))
            
                # TODO probably not needed
                # temp = before_frames[i]
                # temp = ((np.reshape(temp, (height, width)) + 0.5) * 255).astype(np.uint8)

                # temp = np.concatenate([temp, np.zeros_like(temp)], axis=1)
                # temp = np.concatenate([temp, np.zeros_like(temp)], axis=0)
                
                self.before_ch_i += 1
            return cv2.resize(frame_unstable, (output_width, output_height)), bboxs_coords, False
        
        if self.after_ch_i < self.after_ch:
            self.after_temp.append(frame_unstable)
            self.after_frames.append(cvt_img2train(frame_unstable, 1))
            
            self.after_ch_i += 1
            
            if self.after_ch_i < self.after_ch:
                return frame_unstable, bboxs_coords, False

        self.in_xs = []

        self.all_black = np.zeros([height, width], dtype=np.int64)

        dh = int(height * 0.8 / 2)
        dw = int(width * 0.8 / 2)

        black_mask = np.zeros([dh, width], dtype=np.float)
        temp_mask = np.concatenate([np.zeros([height - 2 * dh, dw], dtype=np.float), np.ones([height - 2 * dh, width - 2 * dw], dtype=np.float), np.zeros([height - 2 * dh, dw], dtype=np.float)], axis=1)
        self.black_mask = np.reshape(np.concatenate([black_mask, temp_mask, black_mask], axis=0),[1, height, width, 1]) 

        return None, None, False
    
    def get_stable_frame_infer(self, frame_unstable, bboxs_coords):
        if self.black is not None:
            self.before_frames.append(self.frame)
            self.before_masks.append(self.black.reshape((1, height, width, 1)))
            
            self.before_frames.pop(0)
            self.before_masks.pop(0)
            self.after_frames.append(cvt_img2train(frame_unstable, 1))
            self.after_frames.pop(0)
            self.after_temp.append(frame_unstable)
            self.after_temp.pop(0)
        
        in_x = []
        if input_mask:
            for i in args.indices:
                if (i > 0):
                    in_x.append(self.before_masks[-i])
        for i in args.indices:
            if (i > 0):
                in_x.append(self.before_frames[-i])
        in_x.append(self.after_frames[0])
        for i in args.indices:
            if (i < 0):
                in_x.append(self.after_frames[-i])
        if (args.no_bm == 0):
            in_x.append(self.black_mask)


        in_x = np.concatenate(in_x, axis = 3)

        if MaxSpan != 1:
            self.in_xs.append(in_x)
            if len(self.in_xs) > MaxSpan: 
                self.in_xs = self.in_xs[-1:]
                print('cut')
            in_x = self.in_xs[0].copy()
            in_x[0, ..., self.before_ch] = self.after_frames[0][..., 0]
        tmp_in_x = in_x.copy()
        for j in range(args.refine):
            # start = time.time()
            img, self.black, Hs, x_map_, y_map_ = sess.run([output, black_pix, Hs_tensor, x_map, y_map], feed_dict={x_tensor:tmp_in_x})
            # tot_time += time.time() - start
            
            self.black = self.black[0, :, :]
            xmap = x_map_[0, :, :, 0]
            ymap = y_map_[0, :, :, 0]
            self.all_black = self.all_black + np.round(self.black).astype(np.int64)
            img = img[0, :, :, :].reshape(height, width)
            self.frame = img + self.black * (-1)
            self.frame = self.frame.reshape(1, height, width, 1)
            tmp_in_x[..., -1] = self.frame[..., 0]
        img = ((np.reshape(img, (height, width)) + 0.5) * 255).astype(np.uint8)
        
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        

        smoothed_x_map, smoothed_y_map = smooth_mapping(xmap, ymap)
        
        stable_bboxs_coords = deepcopy(bboxs_coords)
        for i, bbox_coords in enumerate(bboxs_coords):
            (x1, y1), (x2, y2) = bbox_coords
            remapping_res = remap(x1, y1, x2, y2, smoothed_x_map, smoothed_y_map)
            if remapping_res is None:
                # TODO handle out of bounds
                new_x1, new_y1, new_w, new_h = (10000, 10000, x2-x1, y2-y1)
                new_x2 = new_x1 + new_w
                new_y2 = new_y1 + new_h
            else:
                new_x1, new_y1, new_x2, new_y2 = remapping_res
            stable_bboxs_coords[i] = [(new_x1, new_y1), (new_x2, new_y2)]




        img_warped = warpRevBundle2(cv2.resize(self.after_temp[0], (output_width, output_height)), smoothed_x_map, smoothed_y_map)


        stable_frame = img_warped
        return stable_frame, stable_bboxs_coords

    def get_stable_frame(self, frame_unstable, bboxs_coords):
        
        # TODO check the logic
        if not self.has_started:
            stable_frame, stable_bboxs_coords, is_stable = self.stabilization_start(frame_unstable, bboxs_coords)
            
            if stable_frame is not None:
                return stable_frame, stable_bboxs_coords, is_stable
            
            self.has_started = True
        
        stable_frame, stable_bboxs_coords = self.get_stable_frame_infer(frame_unstable, bboxs_coords)

        return stable_frame, stable_bboxs_coords, True

    def reset(self):
        self.before_frames = []
        self.before_masks = []
        self.after_frames = []
        self.after_temp = []
        
        self.black = None
        self.frame = None
        
        self.in_xs = None
        self.black_mask = None
        self.all_black = None
        
        self.after_ch_i = 0
        self.before_ch_i = 0
        self.has_started = False
        


# process_b_receiver.py
import cv2
import zmq
import numpy as np

# 1. Setup ZeroMQ
context = zmq.Context()
socket = context.socket(zmq.REP) # REP (Reply) matches REQ
socket.bind("tcp://*:5555")

print("Receiver: Server started on port 5555. Waiting for frames...")

print('inference with {}'.format(args.indices))
are_videos_left = True
while are_videos_left:
    tot_time = 0

    # stable_cap = cv2.VideoCapture(os.path.join(args.prefix,'stable', video_name)) 
    # unstable_cap = cv2.VideoCapture(os.path.join(args.prefix,'unstable', video_name))
    
    recv_obj = socket.recv_pyobj()

    # Check for stop signal
    if isinstance(recv_obj, str):
        if recv_obj == "STOP":
            print("Receiver: Stop signal received. Shutting down.")
            socket.send_string("ACK")
            break
    video_info = recv_obj
    

    print("video info received")
    socket.send_string("ACK")
    
    fps = video_info["fps"]
    output_width = video_info["output_width"]
    output_height = video_info["output_height"]


    

    print("fps: {}, width: {}, height: {}".format(fps, output_width, output_height))

    # TODO check first frames handling

    stabilizer = VideoStabilizer(fps, before_ch, after_ch)

    length = 0
    full_time_started = time.time()
    
    try:
        while(True):
            recv_obj = socket.recv_pyobj()

            # Check for stop signal
            if isinstance(recv_obj, str):
                if recv_obj == "STOP":
                    print("Receiver: Stop signal received. Shutting down.")
                    socket.send_string("ACK")
                    are_videos_left = False
                    break
                if recv_obj == "NEXT VIDEO":
                    print("Receiver: next video signal received.")
                    socket.send_string("ACK")
                    break
            
            frame_unstable, bboxs_coords = recv_obj
            
            
            start_time = time.time()
            stable_frame, stable_bboxs_coords, is_stable = stabilizer.get_stable_frame(frame_unstable, bboxs_coords)
            tot_time += time.time() - start_time
            
            # videoWriter.write(stable_frame)
            
            socket.send_pyobj((stable_frame, stable_bboxs_coords))
            
            
            
            length = length + 1
            if (length % 10 == 0):
                print("length: " + str(length))      
                print('fps={}'.format(length / tot_time))
                print('full time={}'.format(time.time() - full_time_started))
    except Exception as e:
        traceback.print_exc()
    finally:
        print('total length={}'.format(length))

        # TODO do we need cut frames
