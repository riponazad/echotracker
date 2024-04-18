from utils import viz_utils
import numpy as np

import io
from PIL import Image
import sys
import imageio
import cv2 
import jax
import tree
import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt
import torch
from skimage.color import gray2rgb
import albumentations as A
from functools import partial
import os



def add_text_to_frames(frames, text, position=(5, 15), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, color=(255, 255, 255), thickness=2):
    frames_with_text = []
    for frame in frames:
        # Convert frame to BGR if it's in grayscale
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        # Add text to the frame
        frame_with_text = frame.copy()
        cv.putText(frame_with_text, text, position, font, font_scale, color, thickness)
        frames_with_text.append(frame_with_text)
    
    return np.stack(frames_with_text)


def create_video_from_images(image_folder, output_video_path, fps=30):
    # Get the list of image files in the folder
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]

    # Sort the image files based on their names (assuming the names contain numbers in sequence)
    image_files.sort()

    # Get the first image to extract its size
    first_image = cv2.imread(os.path.join(image_folder, image_files[0]))
    height, width, _ = first_image.shape

    # Define the video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as per your requirement
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Write each image to the video writer
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    # Release the VideoWriter object
    video_writer.release()


def play_video(video_file, target_size=(256, 256)):
    
    # Open the video file
    cap = cv.VideoCapture(video_file)

    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None, None
    

    # List to store frames for the GIF
    frames = []
    new_frame = True
    text = "Press 'q' on the video player to quit the video player after finishing one cycle."
    print(text)
    while True:
        #Restart the video from the beginning
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        while True:
            #Read a frame from the video
            ret, frame = cap.read()

            # Break the inner loop if we have reached the end of the video
            if not ret:
                new_frame = False
                break
            
            # # Add text to the frame
            # cv.putText(frame, text, (30, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # Resize the grayscale frame to the target size
            frame = cv2.resize(frame, target_size)
            # Display the frame
            cv.imshow('Video Player', frame)

            # Append the frame to the list if it is new
            if new_frame == True:
                frames.append(Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2GRAY)))

    
            #Check for user input to quit (press 'q' key)
            key = cv.waitKey(30)
            if key == ord('q'):
                # Release the video capture object and close the display window
                cap.release()
                cv.destroyAllWindows()

                # # Save the video as a gif file
                # output_gif_file = "results/input_video.gif" 
                # frames[0].save(output_gif_file, save_all=True, append_images=frames[1:], loop=0)
                # print(f"Input video is saved as a GIF file to {output_gif_file}")
                # Convert the list of frames into a NumPy array
                frames = np.array(frames)
                # if len(frames.shape) < 4:
                #     frames = frames[..., np.newaxis]
                
                return frames


def augment_video(augmenter, **kwargs):
    assert isinstance(augmenter, A.ReplayCompose)
    keys = kwargs.keys()
    for i in range(len(next(iter(kwargs.values())))):
        data = augmenter(**{
            key: kwargs[key][i] if key not in ['bboxes', 'keypoints'] else [kwargs[key][i]] for key in keys
        })
        if i == 0:
            augmenter = partial(A.ReplayCompose.replay, data['replay'])
        for key in keys:
            if key == 'bboxes':
                kwargs[key][i] = np.array(data[key]).reshape(4)
            elif key == 'keypoints':
                kwargs[key][i] = np.array(data[key]).reshape(2)
            else:
                kwargs[key][i] = data[key]

def getMetricsDict():
    metrics = {
        'occlusion_accuracy': 0.0,
        'pts_within_1': 0.0,
        'pts_within_2': 0.0,
        'pts_within_4': 0.0,
        'pts_within_8': 0.0,
        'pts_within_16': 0.0,
        'average_pts_within_thresh': 0.0,
        'jaccard_1': 0.0,
        'jaccard_2': 0.0,
        'jaccard_4': 0.0,
        'jaccard_8': 0.0,
        'jaccard_16': 0.0,
        'average_jaccard': 0.0,
        'inference_time': 0.0,
        'survival': 0.0,
        'median_traj_error': 0.0
    }
    return metrics

def get_resized_frames(frames, height, width):
  """Resize frames to the given width and height

  Args:
      frames (ndarray): [S, H, W, C]
      height (int): target height
      width (int): target width

  Returns:
      resized_frames (ndarray): [S, height, width, C]
  """
  resized_frames = []

  for frame in frames:
      # Resize the frame
      resized_frame = cv.resize(frame, (width, height))
      # Append the resized frame to the list
      resized_frames.append(resized_frame)

  # Convert the list of frames to a NumPy array
  resized_frames = np.array(resized_frames)

  if len(resized_frames.shape) < 4:
     resized_frames = resized_frames[..., np.newaxis]

  return resized_frames






def decode(frame):
    byteio = io.BytesIO(frame)
    img = Image.open(byteio)
    return np.array(img)



def resize_frame(frame, target_width):
    # Calculate the ratio to maintain the aspect ratio
    ratio = target_width / frame.shape[1]
    # Calculate the new height
    target_height = int(frame.shape[0] * ratio)
    # Resize the frame
    resized_frame = cv2.resize(frame, (target_width, target_height))
    return resized_frame

def create_gif_from_video(video_file, gif_out_file, fps=20, target_width=400):
    # Read the video file
    cap = cv2.VideoCapture(video_file)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create an imageio VideoWriter object
    writer = imageio.get_writer(gif_out_file, fps=fps, loop=0)

    try:
        while True:
            # Read a frame from the video
            ret, frame = cap.read()

            # Break if no more frames
            if not ret:
                break

            # Resize the frame to the target width
            resized_frame = resize_frame(frame, target_width)

            # Convert frame to RGB (imageio uses RGB)
            frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

            # Append frame to the GIF
            writer.append_data(frame_rgb)
    finally:
        # Release the video capture object
        cap.release()

        # Close the imageio VideoWriter
        writer.close()

def paint_vid(frames, points, visibs, gray = False):
    """This will paint the points into the frames.

    Args:
        frames (ndarray): (n_frames, width, height, channel)
        points (ndarray): (n_points, n_frames, 2)
        occluded (ndarray): (n_points, n_frames)
    
    Return:
        painted_frames (ndarray): (n_frames, width, height, channel)
    """
    
    if gray:
       frames = frames.squeeze()
       frames = gray2rgb(frames)

    scale_factor = np.array(frames.shape[2:0:-1])[np.newaxis, np.newaxis, :]
    painted_frames = viz_utils.paint_point_track(
        frames,
        points  * scale_factor,
        visibs,
    )

    return painted_frames


def get_transform(frames, trajs_g, visibs_g, resize):
  """To apply augmentations/transformations to frames.

  Args:
      frames (ndarray): [S, H, W, C] -> [0 - 255]
      trajs_g (ndarray): [N, S, 2] -> (x, y) -> [0.0, 1.0]
      visibs_g (ndarray): [N, S] -> {True, False}
      resize (tuple): Target size (height, width).

  Returns:
      frames (ndarray): [S, resized_height, resized_width, C] -> [0 - 255]
      trajs_g (ndarray): [N, S, 2] -> (x, y) -> [0.0, 1.0]
      visibs_g (ndarray): [N, S] -> {True, False}
  """  
  frames = get_resized_frames(frames=frames, height=resize[0], width=resize[1])
  return frames, trajs_g, visibs_g


if __name__ == '__main__':
    create_video_from_images(sys.argv[1], sys.argv[2], fps=20)
