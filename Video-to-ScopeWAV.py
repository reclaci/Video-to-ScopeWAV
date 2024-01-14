import cv2
import numpy as np
from scipy.io.wavfile import write
from scipy.spatial import KDTree
from tkinter import filedialog, Tk, Button, Label
import os

root = Tk()
root.title("Video to Oscilloscope WAV Converter")

video_path = filedialog.askopenfilename(title="Select video file")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError("Cannot open video file")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if frame_height >= frame_width:
    height = 256
    width = round(frame_width / frame_height * height)
else:
    width = 256
    height = round(frame_height / frame_width * width)

sampling_rate = 88200
sound_channels = 2
samp_per_frame = round(sampling_rate / fps)
bits_per_sample = 8

edge_points = []

def sort_edge_points(points):
    if len(points) <= 1:
        return points

    tree = KDTree(points)
    sorted_points = [points[0]]
    points = np.delete(points, 0, axis=0)

    while len(points) > 1:
        last_point = sorted_points[-1]
        dist, idx = tree.query(last_point, k=2)
        next_idx = idx[1] if idx[0] == 0 else idx[0]
        sorted_points.append(points[next_idx])
        points = np.delete(points, next_idx, axis=0)
        tree = KDTree(points)

    sorted_points.append(points[0])

    return np.array(sorted_points)

def process_frame(frame_idx=0):
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        cap.release()
        cv2.destroyAllWindows()
        start_conversion_button.config(state="normal")
        return
    else:
        print("Processing frame: %d" % (frame_idx))
    
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    edges = cv2.Canny(blurred, 50, 150)
    
    y, x = np.where(edges != 0)
    
    points = np.column_stack((x, height - y))
    points = sort_edge_points(points)
    edge_points.append(points)
    
    cv2.imshow('Edges', edges)
    cv2.waitKey(1)
    
    root.after(1, lambda: process_frame(frame_idx + 1))

start_button = Button(root, text="Start Processing", command=lambda: process_frame())
start_button.config(height=2, width=20)
start_button.pack()

def start_conversion():
    wav_data = np.zeros((len(edge_points) * samp_per_frame, sound_channels), dtype=np.uint8)
    
    for frame_idx, points in enumerate(edge_points):
        if len(points) > 0:
            for j in range(samp_per_frame):
                idx = int(j * len(points) / samp_per_frame)
                wav_data[frame_idx * samp_per_frame + j, 0] = int(points[idx, 0] / width * 255)
                wav_data[frame_idx * samp_per_frame + j, 1] = int(points[idx, 1] / height * 255)
        else:
            wav_data[frame_idx * samp_per_frame:(frame_idx + 1) * samp_per_frame, :] = 128
    
    output_path = os.path.splitext(video_path)[0] + '_output.wav'
    write(output_path, sampling_rate, wav_data)
    print(f"WAV file has been saved as '{output_path}'.")
    label.config(text="Conversion Completed!")

start_conversion_button = Button(root, text="Start Conversion", state="disabled", command=start_conversion)
start_conversion_button.config(height=2, width=20)
start_conversion_button.pack()

label = Label(root, text="")
label.pack()

root.mainloop()
