""" 
Created on Wednesday Dec 06 2023
@author: Azad Md Abulkalam
@location: ISB, NTNU

Evaluation metrices for point-tracking.
"""
import numpy as np
import torch
from utils import basic
from typing import Mapping



def compute_metrics(query_points, trajs_g, visibs_g, trajs_e, visibs_e):
    """ Compute point tracking evaluation metrics for a given batch of samples

    Args:
        query_points (ndarray): (B, N, 2) -> (x, y) -> [0.0, 1.0], considering all query points are in the first frame
        trajs_g (ndarray): (B, N, S, 2) -> (x, y) -> [0.0, 1.0]
        visibs_g (ndarray): (B, N, S) -> True or False
        trajs_e (ndarray): same as trajs_g
        visibs_e (ndarray): same as visibs_g

    Returns:
        Dict with the following keys:
        
        occlusion_accuracy: Accuracy at predicting occlusion.
        pts_within_{x} for x in [1, 2, 4, 8, 16]: Fraction of points predicted to be within the given pixel threshold, ignoring occlusion
        prediction.
        jaccard_{x} for x in [1, 2, 4, 8, 16]: Jaccard metric for the given threshold 
        average_pts_within_thresh: average across pts_within_{x}
        average_jaccard: average across jaccard_{x}
        median_traj_error: Median Trajectory Error (same as in PIPs++)
        survival: Survival rate (same as in PIPs++)
    """   
    #scale factor to be compatible with TAP-Vid metrics computation
    s = 256.0 # both the height and width are 256.0
    
    # from (x, y) to (y, x)
    query_points = query_points[:,:,[1, 0]] * s # (B, N, 2)
    # preparing the time dimension to be concatenated
    time_dim = np.zeros((query_points.shape[0], query_points.shape[1], 1))
    # prepending a column to be -> (B, N, 3)
    query_points = np.concatenate((time_dim, query_points), axis=-1)
    

    #retrieving ground-truth occlusion from visibility and adding a batch dimension
    gt_occluded = np.logical_not(visibs_g)
    pd_occluded = np.logical_not(visibs_e)
    
    #scaling to conform with TAP-Vid 256x256
    gt_tracks = trajs_g * s
    pd_tracks = trajs_e * s

    outputs = compute_tapvid_metrics(query_points, gt_occluded, gt_tracks, pd_occluded, pd_tracks, 'first')

    # taking the mean across videos
    metrics = {}
    for key, value in outputs.items():
      metrics[key] = value.mean()

    return metrics
  



""" This method is completely adopted from TAP-Vid repo """
def compute_tapvid_metrics(
    query_points: np.ndarray,
    gt_occluded: np.ndarray,
    gt_tracks: np.ndarray,
    pred_occluded: np.ndarray,
    pred_tracks: np.ndarray,
    query_mode: str,
) -> Mapping[str, np.ndarray]:
  """Computes TAP-Vid metrics (Jaccard, Pts.

  Within Thresh, Occ.

  Acc.)

  See the TAP-Vid paper for details on the metric computation.  All inputs are
  given in raster coordinates.  The first three arguments should be the direct
  outputs of the reader: the 'query_points', 'occluded', and 'target_points'.
  The paper metrics assume these are scaled relative to 256x256 images.
  pred_occluded and pred_tracks are your algorithm's predictions.

  This function takes a batch of inputs, and computes metrics separately for
  each video.  The metrics for the full benchmark are a simple mean of the
  metrics across the full set of videos.  These numbers are between 0 and 1,
  but the paper multiplies them by 100 to ease reading.

  Args:
     query_points: The query points, an in the format [t, y, x].  Its size is
       [b, n, 3], where b is the batch size and n is the number of queries
     gt_occluded: A boolean array of shape [b, n, t], where t is the number of
       frames.  True indicates that the point is occluded.
     gt_tracks: The target points, of shape [b, n, t, 2].  Each point is in the
       format [x, y]
     pred_occluded: A boolean array of predicted occlusions, in the same format
       as gt_occluded.
     pred_tracks: An array of track predictions from your algorithm, in the same
       format as gt_tracks.
     query_mode: Either 'first' or 'strided', depending on how queries are
       sampled.  If 'first', we assume the prior knowledge that all points
       before the query point are occluded, and these are removed from the
       evaluation.

  Returns:
      A dict with the following keys:

      occlusion_accuracy: Accuracy at predicting occlusion.
      pts_within_{x} for x in [1, 2, 4, 8, 16]: Fraction of points
        predicted to be within the given pixel threshold, ignoring occlusion
        prediction.
      jaccard_{x} for x in [1, 2, 4, 8, 16]: Jaccard metric for the given
        threshold
      average_pts_within_thresh: average across pts_within_{x}
      average_jaccard: average across jaccard_{x}
  """

  metrics = {}

  eye = np.eye(gt_tracks.shape[2], dtype=np.int32)
  if query_mode == 'first':
    # evaluate frames after the query frame
    query_frame_to_eval_frames = np.cumsum(eye, axis=1) - eye
  elif query_mode == 'strided':
    # evaluate all frames except the query frame
    query_frame_to_eval_frames = 1 - eye
  else:
    raise ValueError('Unknown query mode ' + query_mode)

  query_frame = query_points[..., 0]
  query_frame = np.round(query_frame).astype(np.int32)
  evaluation_points = query_frame_to_eval_frames[query_frame] > 0

  # Occlusion accuracy is simply how often the predicted occlusion equals the
  # ground truth.
  occ_acc = np.sum(
      np.equal(pred_occluded, gt_occluded) & evaluation_points,
      axis=(1, 2),
  ) / np.sum(evaluation_points)
  metrics['occlusion_accuracy'] = occ_acc
  


  '''Computing MTE between estimated and ground truth trajectories and survival 
  rate of the estimated trajectory when the points are not queried and visible 
  in the ground truth'''
  valid_points = np.logical_and(evaluation_points, ~gt_occluded)
  l2_dists = np.linalg.norm(pred_tracks - gt_tracks, axis=-1)
  valid_dists = np.ma.masked_array(l2_dists, ~valid_points)
  thrs = 50 # threshold pixels for tracking failure
  dist_ok = 1 - (valid_dists > thrs) 
  survival = np.ma.cumprod(dist_ok, axis=2)
  survival = np.mean(np.mean(survival, axis=2), axis=1) * 100
  metrics['survival'] = survival
  # print(np.median(valid_dists, axis=2)[0][18], gt_tracks[0][18])
  # raise KeyboardInterrupt
  mte = np.ma.mean(np.ma.median(valid_dists, axis=2), axis=1)
  metrics['median_traj_error'] = mte

  # Next, convert the predictions and ground truth positions into pixel
  # coordinates.
  visible = np.logical_not(gt_occluded)
  pred_visible = np.logical_not(pred_occluded)
  all_frac_within = []
  all_jaccard = []
  for thresh in [1, 2, 4, 8, 16]:
    # True positives are points that are within the threshold and where both
    # the prediction and the ground truth are listed as visible.
    within_dist = np.sum(
        np.square(pred_tracks - gt_tracks),
        axis=-1,
    ) < np.square(thresh)
    is_correct = np.logical_and(within_dist, visible)

    # Compute the frac_within_threshold, which is the fraction of points
    # within the threshold among points that are visible in the ground truth,
    # ignoring whether they're predicted to be visible.
    count_correct = np.sum(
        is_correct & evaluation_points,
        axis=(1, 2),
    )
    count_visible_points = np.sum(visible & evaluation_points, axis=(1, 2))
    frac_correct = count_correct / count_visible_points
    metrics['pts_within_' + str(thresh)] = frac_correct
    all_frac_within.append(frac_correct)

    true_positives = np.sum(
        is_correct & pred_visible & evaluation_points, axis=(1, 2)
    )

    # The denominator of the jaccard metric is the true positives plus
    # false positives plus false negatives.  However, note that true positives
    # plus false negatives is simply the number of points in the ground truth
    # which is easier to compute than trying to compute all three quantities.
    # Thus we just add the number of points in the ground truth to the number
    # of false positives.
    #
    # False positives are simply points that are predicted to be visible,
    # but the ground truth is not visible or too far from the prediction.
    gt_positives = np.sum(visible & evaluation_points, axis=(1, 2))
    false_positives = (~visible) & pred_visible
    false_positives = false_positives | ((~within_dist) & pred_visible)
    false_positives = np.sum(false_positives & evaluation_points, axis=(1, 2))
    jaccard = true_positives / (gt_positives + false_positives)
    metrics['jaccard_' + str(thresh)] = jaccard
    all_jaccard.append(jaccard)
  metrics['average_jaccard'] = np.mean(
      np.stack(all_jaccard, axis=1),
      axis=1,
  )
  metrics['average_pts_within_thresh'] = np.mean(
      np.stack(all_frac_within, axis=1),
      axis=1,
  )
  return metrics