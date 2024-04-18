""" 
Created on Tuesday 02 April 2024
@author: Azad Md Abulkalam
@location: ISB, NTNU

To run the proposed models as examples.
"""

from utils.utils_ import paint_vid, play_video
from utils import viz_utils
import numpy as np
import mediapy as media
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
import torch
from model.net import EchoTracker


# Event handler for mouse clicks
def on_click(event):
    if event.button == 1 and event.inaxes == ax:  # Left mouse button clicked
        x, y = int(np.round(event.xdata)), int(np.round(event.ydata))

        select_points.append(np.array([x, y]))

        color = colormap[len(select_points) - 1]
        color = tuple(np.array(color) / 255.0)
        ax.plot(x, y, 'o', color=color, markersize=5)
        plt.draw()
            

if __name__ == '__main__':
    model = EchoTracker(device_ids=[0])
    model.load(path="model/weights/", eval=True)
    video_path = "data/input.mp4"

    s = 400
    #Show the example video
    frames = play_video(video_path, target_size=(s+40, s))
    sampling_frame = 0 #points will be selected only from the first frame

    #Select Any Points at Any Frame
    # Generate a colormap with 20 points, no need to change unless select more than 20 points
    colormap = viz_utils.get_colors(20)

    fig, ax = plt.subplots(figsize=(8, 5))
    #print(frames[sampling_frame].shape)
    ax.imshow(Image.fromarray(cv.cvtColor(frames[sampling_frame], cv.COLOR_GRAY2RGB)))
    ax.axis('off')
    ax.set_title('You can select more than 1 points. After select enough points, close the window.')

    select_points = []

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

    #print(select_points)

    # processing the input frames
    if len(frames.shape) < 4:
        frames = frames[..., np.newaxis]
    frames = torch.from_numpy(frames)
    frames = frames.unsqueeze(0)
    #print(type(frames), frames.shape)

    # processing the query points
    query_points = torch.tensor(np.array(select_points), dtype=torch.float)
    query_points = query_points.unsqueeze(0)
    query_points[...,0] /= s+40
    query_points[...,1] /= s
    #print(type(query_points), query_points.shape)

     
    trajs_e = model.infer(frames, query_points)
    visibs_e = torch.ones((trajs_e.shape[:-1]))
    
    pd_frames = paint_vid(frames=frames.squeeze().numpy(), points=trajs_e.squeeze().numpy(), visibs=visibs_e.squeeze().numpy(), gray=True)
    media.write_video("results/output.mp4", pd_frames, fps=20)
    print(f'The example video is saved at "results/output.mp4"')
    play_video("results/output.mp4", target_size=(s+40, s))