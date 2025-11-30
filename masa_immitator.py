# process_a_sender.py
import cv2
import zmq
import time
import os


import argparse

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

video_list = []

for list_path in args.test_list:
    if os.path.isfile(list_path):
        print('adding '+list_path)
        list_f = open(list_path, 'r')
        temp = list_f.read()
        video_list.extend(temp.split('\n'))

# 1. Setup ZeroMQ
context = zmq.Context()
# We use REQ (Request) because we want to wait for the answer 
# before sending the next frame (flow control).
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")



print(f"Sender: Connected to port 5555. Reading {args.test_list}...")

frame_count = 0

try:
    for video_name in video_list:
        unstable_cap = cv2.VideoCapture(os.path.join(args.prefix,'unstable', video_name))

        fps = unstable_cap.get(cv2.CAP_PROP_FPS)
        output_width = int(unstable_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        output_height = int(unstable_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video_info = {"fps": fps, "output_width": output_width, "output_height": output_height}
        socket.send_pyobj(video_info)
        socket.recv()

        videoWriter = cv2.VideoWriter(os.path.join(args.output_dir, video_name + '.avi'), 
            cv2.VideoWriter_fourcc('M','J','P','G'), fps, (output_width, output_height))
        
        while True:
            ret, frame = unstable_cap.read()
            
            if not ret:
                print("Sender: End of video.")
                # Send a stop signal so the receiver knows to quit
                socket.send_pyobj("NEXT VIDEO")
                socket.recv() # Wait for final ack
                break

            # 3. Send the frame (as a numpy object)
            # resize to speed up transfer if needed: frame = cv2.resize(frame, (640, 480))
            print(f"Sender: Sending frame {frame_count}...")
            socket.send_pyobj(frame)

            # 4. Wait for the processed result (or just an acknowledgement)
            # If your receiver sends back a processed image, read it here.
            response = socket.recv_pyobj()
            
            videoWriter.write(response)
            
            frame_count += 1

except KeyboardInterrupt:
    print("Sender: Interrupted.")
    socket.send_pyobj("STOP")
    socket.recv()

finally:
    print("Sender: End of videos.")
    # Send a stop signal so the receiver knows to quit
    socket.send_pyobj("STOP")
    socket.recv() # Wait for final ack
    unstable_cap.release()
    socket.close()
    context.term()