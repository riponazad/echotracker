""" 
Created on Thursday Jan 18 2024
@author: Azad Md Abulkalam
@location: ISB, NTNU

The proposed EchoTracker model for ultrasound medical imaging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


from model.blocks import (
    CorrBlock, BasicEncoder, DeltaBlock, DeltaBlock_temp
)
from utils.samp import bilinear_sample2d
from utils.trainer import sequence_loss
import configs
from utils.utils_ import get_transform
import random
torch.manual_seed(0)





#EchoTracker: Advancing Myocardial Point Tracking in Echocardiography
class EchoPIPs(nn.Module):
    def __init__(
            self,
            stride=8, #spatial stride
            hidden_dim=128,
            latent_dim=64,
            corr_radius=3,
            corr_levels=4,
            scale = 4, # 
    ):
        super(EchoPIPs, self).__init__()
        self.stride = stride
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.scale = scale
        
        self.fnet = BasicEncoder(
            input_dim=1, output_dim=self.latent_dim, norm_fn="instance",
            dropout=0, stride=self.stride
        )
        self.fnet_fine = BasicEncoder(
            input_dim=1, output_dim=self.latent_dim, norm_fn="instance",
            dropout=0, stride=int(self.stride/self.scale)
        )
        self.delta_block_spatial = DeltaBlock(
            hidden_dim=self.hidden_dim, latent_dim=self.latent_dim,
            corr_levels=self.corr_levels, corr_radius=self.corr_radius
        )
        self.delta_block_temporal = DeltaBlock_temp(
            hidden_dim=self.hidden_dim, latent_dim=self.latent_dim,
            corr_levels=self.corr_levels, corr_radius=self.corr_radius
        )
        self.linear = nn.Linear(588, 196+196)
        

    def initialize_(self, rgbs, points_0):
        B, S, C, H, W = rgbs.shape
        B,N,D = points_0.shape
        #creating the coarse feature maps for all the frames
        H_ = H//self.stride
        W_ = W//self.stride
        rgbs_ = rgbs.reshape(B*S, C, H, W)
        fmaps_ = self.fnet(rgbs_)
        fmaps = fmaps_.reshape(B, S, self.latent_dim, H_, W_) # the coarst fmaps

        coords_0 = points_0.clone()/float(self.stride)
        coords = coords_0.unsqueeze(1).repeat(1, S, 1, 1) # B,S,N,2

        #Features for all the points from the first frame
        f_vecs0 = bilinear_sample2d(fmaps[:,0],  coords[:,0,:,0], coords[:,0,:,1]).permute(0, 2, 1) # B,N,C  feature vectors in frame 0
        feats = f_vecs0.unsqueeze(1).repeat(1, S, 1, 1) # B,S,N,C
        
        #computing cost volume at multi-scale using pyramids
        fcorr_fn = CorrBlock(fmaps, num_levels=self.corr_levels, radius=self.corr_radius)
        
        coords_bak = coords.clone()
        coords[:,0] = coords_bak[:,0] # lock coord0 for target
        
        rgbs_ = rgbs_.squeeze()
        frame_flow = torch.cat([rgbs_[-1:], (rgbs_[1:] - rgbs_[:-1])], dim=0).unsqueeze(1)
        frame_flow = F.interpolate(frame_flow, (50, 60), mode='bilinear').permute(1, 0, 2, 3).reshape(1, S, 50*60).repeat(N, 1, 1)
        #taking help from the first frame
        xys0 = coords_bak[:,0,:,:].permute(1, 0, 2).repeat(1, S, 1)
        
        coords = coords.detach()
        fcorr_fn.corr(feats) #computing cross-correlations
        
        #print(fcor.shape)
        fcorrs = []
        for c_v in fcorr_fn.corrs_pyramid:
            B, S, N, CH, CW = c_v.shape
            c_v = c_v.reshape(B*S, N, CH, CW)
            c_v = F.interpolate(c_v, (14, 14), mode='bilinear', align_corners=True)
            fcorrs.append(c_v.reshape(B, S, N, 14*14))
        fcorrs = torch.stack(fcorrs).mean(dim=0)
        
        
        LRR = fcorrs.shape[3]
        # we want everything in the format B*N, S, C
        fcorrs_ = fcorrs.permute(0, 2, 1, 3).reshape(B*N, S, LRR)
        #print(fcorrs_.shape)
        flows_ = (coords[:,1:] - coords[:,:-1]).permute(0,2,1,3).reshape(B*N, S-1, 2)
        flows_ = torch.cat([flows_, flows_[:,-1:]], dim=1) # B*N,S,2
        
        delta_coords_ = self.delta_block_spatial(fcorrs_, flows_, frame_flow, xys0) # B*N,S,2
        delta_coords_ = delta_coords_.reshape(B, N, S, 2).permute(0,2,1,3)
        coords = coords + delta_coords_

        trajs_e = coords * self.stride

        return trajs_e

        
    def forward(self, rgbs, points_0, trajs_g=None, vis_g=None, valids=None, iters=4):
        """ Forward pass of the model

        Args:
            rgbs (tensor): a sequence of frames of shape [B, S, C, H, W]. S must be divisible by 8. range: [0 - 255]
            points_0 (tensor): a list of points on the first frame of the given video [B, N, 2]. range: [0 - H/W]
            
        Returns:
            _type_: _description_
        """  
        B,N,D = points_0.shape
        assert(D==2)  
        B, S, C, H, W = rgbs.shape
        assert (C==1), "Number of channel should be 1 for echo frames"
        #normalizing in [-1.0, 1.0]
        rgbs = 2 * (rgbs / 255.0) - 1.0 #normalization within -1.0 to 1.0

        coords_init = self.initialize_(rgbs, points_0)
        
        coord_predictions1 = [] # for loss
        coord_predictions1.append(coords_init)
        
        #creating the fine feature maps for all the frames
        H_ = H//int(self.stride/self.scale)
        W_ = W//int(self.stride/self.scale)
        rgbs_ = rgbs.reshape(B*S, C, H, W)
        fmaps_ = self.fnet_fine(rgbs_)
        fmaps = fmaps_.reshape(B, S, self.latent_dim, H_, W_) # the fine fmaps


        coords = coords_init.clone()/(self.stride/self.scale) # B,S,N,2

        # #computing cost volume at multi-scale using pyramids
        fcorr_fn1 = CorrBlock(fmaps, num_levels=self.corr_levels, radius=self.corr_radius)
        fcorr_fn2 = CorrBlock(fmaps, num_levels=self.corr_levels, radius=self.corr_radius)
        fcorr_fn4 = CorrBlock(fmaps, num_levels=self.corr_levels, radius=self.corr_radius)

        #Features for all the points from the first frame
        f_vecs0 = bilinear_sample2d(fmaps[:,0],  coords[:,0,:,0], coords[:,0,:,1]).permute(0, 2, 1) # B,N,C  feature vectors in frame 0
        feats1 = f_vecs0.unsqueeze(1).repeat(1, S, 1, 1) # B,S,N,C
        
        fcorr_fn1.corr(feats1) #computing cross-correlations

        coords_bak = coords.clone()
        coords[:,0] = coords_bak[:,0] # lock coord0 for target


        rgbs_ = rgbs_.squeeze()
        frame_flow = torch.cat([rgbs_[-1:], (rgbs_[1:] - rgbs_[:-1])], dim=0).unsqueeze(1)
        frame_flow = F.interpolate(frame_flow, (50, 60), mode='bilinear').permute(1, 0, 2, 3).reshape(1, S, 50*60).repeat(N, 1, 1)
        #taking help from the first frame
        xys0 = coords_bak[:,0,:,:].permute(1, 0, 2).repeat(1, S, 1)


        for itr in range(iters):
            coords = coords.detach()

            # timestep indices
            inds2 = (torch.arange(S)-2).clip(min=0)
            inds4 = (torch.arange(S)-4).clip(min=0)
            # coordinates at these timesteps
            coords2_ = coords[:,inds2].reshape(B*S,N,2)
            coords4_ = coords[:,inds4].reshape(B*S,N,2)
            # featuremaps at these timesteps
            fmaps2_ = fmaps[:,inds2].reshape(B*S,self.latent_dim,H_,W_)
            fmaps4_ = fmaps[:,inds4].reshape(B*S,self.latent_dim,H_,W_)
            # features at these coords/times
            feats2_ = bilinear_sample2d(fmaps2_, coords2_[:,:,0], coords2_[:,:,1]).permute(0, 2, 1) # B*S, N, C
            feats2 = feats2_.reshape(B,S,N,self.latent_dim)
            feats4_ = bilinear_sample2d(fmaps4_, coords4_[:,:,0], coords4_[:,:,1]).permute(0, 2, 1) # B*S, N, C
            feats4 = feats4_.reshape(B,S,N,self.latent_dim)

            fcorr_fn2.corr(feats2)
            fcorr_fn4.corr(feats4)

            # now we want costs at the current locations
            fcorrs1 = fcorr_fn1.sample(coords) # B,S,N,LRR
            fcorrs2 = fcorr_fn2.sample(coords) # B,S,N,LRR
            fcorrs4 = fcorr_fn4.sample(coords) # B,S,N,LRR
            LRR = fcorrs1.shape[3]

            # we want everything in the format B*N, S, C
            fcorrs1_ = fcorrs1.permute(0, 2, 1, 3).reshape(B*N, S, LRR)
            fcorrs2_ = fcorrs2.permute(0, 2, 1, 3).reshape(B*N, S, LRR)
            fcorrs4_ = fcorrs4.permute(0, 2, 1, 3).reshape(B*N, S, LRR)
            fcorrs_ = torch.cat([fcorrs1_, fcorrs2_, fcorrs4_], dim=2)
            fcorrs_ = self.linear(fcorrs_)
            flows_ = (coords[:,1:] - coords[:,:-1]).permute(0,2,1,3).reshape(B*N, S-1, 2)
            flows_ = torch.cat([flows_, flows_[:,-1:]], dim=1) # B*N,S,2

            delta_coords_ = self.delta_block_temporal(fcorrs_, flows_, frame_flow, xys0) # B*N,S,2

            
            coords = coords + delta_coords_.reshape(B, N, S, 2).permute(0,2,1,3)
            coord_predictions1.append(coords * (self.stride/self.scale))

    
        if trajs_g is not None:
            loss = sequence_loss(coord_predictions1, trajs_g, vis_g, valids, 0.8)
        else:
            loss = None
            
        return coord_predictions1, loss

