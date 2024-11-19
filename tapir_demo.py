""" 
Created on Tuesday 02 April 2024
@author: Azad Md Abulkalam
@location: ISB, NTNU

To run the proposed models as examples.
"""

from utils.utils_ import paint_vid, add_text_to_frames
from utils.trainer import getTrainParams
from utils import evaluate
import numpy as np
import mediapy as media
import pickle
from model.net import TAPIR


if __name__ == '__main__':
    
    # model=TAPIR(pyramid_level=0, device_ids=[0], ft_model=False)
    # model.load(path="model/weights/tapir/tapir_checkpoint_panning.pt")
    ################## for finetuned model #################
    model=TAPIR(pyramid_level=0, device_ids=[0], ft_model=True)
    model.load(path="model/weights/tapir/finetuned")
    getTrainParams(model.model)
    
    # Specify the file path for the pickle file
    data = "data/samples.pkl"

    # Open the pickle file in binary read mode
    with open(data, 'rb') as f:
        # Load the contents of the pickle file into a dictionary
        ds_list = pickle.load(f)

    #print(type(ds_list), len(ds_list['frames']))
    frames, trajs_g, visibs_g = ds_list['frames'][0], ds_list['trajs'][0], ds_list['visibility'][0]
    points_0 = trajs_g[:,:,0] # taking points at frame 0

    frames = frames.repeat(1,1,1,1,3)
    # print(frames.shape, points_0.shape, trajs_g.shape, visibs_g.shape)
    # print(points_0.min(), points_0.max())
    # raise KeyboardInterrupt
    
    gt_frames = paint_vid(frames=frames.squeeze().numpy(), points=trajs_g.squeeze().numpy(), visibs=visibs_g.squeeze().numpy())
    
    trajs_e = model.infer(frames, points_0)

    
    pd_frames = paint_vid(frames=frames.squeeze().numpy(), points=trajs_e.squeeze().numpy(), visibs=visibs_g.squeeze().numpy())
    gt_frames = add_text_to_frames(gt_frames, "Ground-truth", color=(0, 255, 0))
    pd_frames = add_text_to_frames(pd_frames, "Estimation")
    cat_frames = np.concatenate((gt_frames, pd_frames), axis=2)
    media.write_video("results/output.mp4", cat_frames, fps=20)

    print(f'The output video is saved at "results/output.mp4"')

    '''Evaluation'''
    #print(points_0.shape, trajs_g.shape, trajs_e.shape, visibs_g.shape)
    metrics = evaluate.compute_metrics(points_0.numpy(), trajs_g.numpy(), visibs_g.numpy(), trajs_e.numpy(), visibs_g.numpy())
    #print(metrics)

    for k, v in metrics.items():
        print(f"{k}: {v:0.2}")