import torch 
import pickle
import numpy as np
from utils.utils_ import paint_vid, add_text_to_frames
from utils import evaluate
import mediapy as media

#path to the model (torchscript) file
TORCHSCRIPTFILE = "/home/mdaaz/Documents/GitHub/STIRMetrics/src/EchoTracker_TORCHSCRIPT.pth"


if __name__ == '__main__':
    # Specify the file path for the pickle file
    data = "data/samples.pkl"

    # Open the pickle file in binary read mode
    with open(data, 'rb') as f:
        # Load the contents of the pickle file into a dictionary
        ds_list = pickle.load(f)

    #print(type(ds_list), len(ds_list['frames']))
    frames, trajs_g, visibs_g = ds_list['frames'][0].permute(0,1,4,2,3), ds_list['trajs'][0], ds_list['visibility'][0]
    points_0 = trajs_g[:,:,0].long() # taking points at frame 0

    #print(frames.shape, points_0.shape, trajs_g.shape, visibs_g.shape)
    gt_frames = paint_vid(frames=frames.squeeze().numpy(), points=trajs_g.squeeze().numpy(), visibs=visibs_g.squeeze().numpy(), gray=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tracker = torch.jit.load(TORCHSCRIPTFILE, map_location=torch.device('cuda')).to(device)
    
    # print(frames.shape, points_0.shape, trajs_g.shape, visibs_g.shape)
    # print(points_0.min(), points_0.max())
    # raise KeyboardInterrupt

    with torch.no_grad():
        pred, _ = tracker(frames.to(device), points_0.to(device))
    trajs_e = pred[-1].permute(0,2,1,3)#.squeeze(0)#[1]

    # print(trajs_e.shape)
    # raise KeyboardInterrupt


    pd_frames = paint_vid(frames=frames.squeeze().numpy(), points=trajs_e.cpu().squeeze().numpy(), visibs=visibs_g.squeeze().numpy(), gray=True)
    gt_frames = add_text_to_frames(gt_frames, "Ground-truth", color=(0, 255, 0))
    pd_frames = add_text_to_frames(pd_frames, "Estimation")
    cat_frames = np.concatenate((gt_frames, pd_frames), axis=2)
    media.write_video("results/output.mp4", cat_frames, fps=20)

    print(f'The output video is saved at "results/output6.mp4"')

    '''Evaluation'''
    #print(points_0.shape, trajs_g.shape, trajs_e.shape, visibs_g.shape)
    metrics = evaluate.compute_metrics(points_0.numpy(), trajs_g.numpy(), visibs_g.numpy(), trajs_e.cpu().numpy(), visibs_g.numpy())
    #print(metrics)

    for k, v in metrics.items():
        print(f"{k}: {v:0.2}")