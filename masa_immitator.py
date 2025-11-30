# process_a_sender.py
import cv2
import zmq
import time
import os


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test-list', nargs='+')
parser.add_argument('--production-dir', nargs='+')
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

        videoWriter = cv2.VideoWriter(os.path.join(production_dir, video_name + '.avi'), 
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
            
            # If the receiver sent back an image, display it
            if isinstance(response, str):
                print(f"Sender: Received message: {response}")
            else:
                cv2.imshow("Processed Result from Other Process", response)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1

except KeyboardInterrupt:
    print("Sender: Interrupted.")
    socket.send_pyobj("STOP")
    socket.recv()

finally:
    print("Sender: End of video.")
    # Send a stop signal so the receiver knows to quit
    socket.send_pyobj("STOP")
    socket.recv() # Wait for final ack
    unstable_cap.release()
    socket.close()
    context.term()