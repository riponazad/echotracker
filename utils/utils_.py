from utils import viz_utils
import numpy as np

from PIL import Image
import cv2 as cv
from skimage.color import gray2rgb



def add_text_to_frames(frames, text, position=(5, 15), font=cv.FONT_HERSHEY_SIMPLEX, font_scale=0.5, color=(255, 255, 255), thickness=2):
    frames_with_text = []
    for frame in frames:
        # Convert frame to BGR if it's in grayscale
        if len(frame.shape) == 2:
            frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
        
        # Add text to the frame
        frame_with_text = frame.copy()
        cv.putText(frame_with_text, text, position, font, font_scale, color, thickness)
        frames_with_text.append(frame_with_text)
    
    return np.stack(frames_with_text)

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
            frame = cv.resize(frame, target_size)
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

