""" 
Created on Tuesday Jan 23 2024
@author: Azad Md Abulkalam
@location: ISB, NTNU

To load/train/finetune/test the EchoTracker model.
"""
import torch.nn.functional as F
import torch
from utils.trainer import requires_grad, fetch_optimizer
from utils.utils_ import getMetricsDict
from utils import saverloader
from tqdm import tqdm
from utils import evaluate
import os
from torch.utils.tensorboard import SummaryWriter

from model.echopips import EchoPIPs


class EchoTracker():
    def __init__(self, stride=8, device_ids=list(range(torch.cuda.device_count()))) -> None:
        """ initializing the EchoPIPs architecture

        Args:
            stride (int, optional): spatial stride of the model. Defaults to 8.
            device_ids : a list of device ids, Defaults to [0]
        """
        self.device = 'cuda:%d' % device_ids[0]
        self.device_ids = device_ids
        self.model = EchoPIPs(stride=stride).to(self.device)

    def load(self, path=None, eval=True):
        """ load EchoPIPs for training/evaluation

        Args:
            path (str, optional): path to the weights. Defaults to None
            eval (bool, optional): False if the model is loaded for training/finetuing. Defaults to True.
        """
        if path is not None:
            _ = saverloader.load(path, self.model)
            print(f"EchoTracker is loaded from {path}.")
        
        if eval:
            requires_grad(self.model.parameters(), False)
            print("EchoTracker is loaded for evaluation.")
            return self.model.eval()
        else:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)
            requires_grad(self.model.parameters(), True)
            print("EchoTracker is loaded for training.")
            return self.model.train()
        
    def infer(self, video, points, resize=None):
        """Run the model on the video for the given points in 1st frame and return the results.

        Args:
            video (tensor): a sequence of frames of shape [B, S, H, W, C]. S must be divisible by 8. range: [0 - 255]
            points (tensor): a list of points on first frame of the given video [B, N, 2]. range: [0.0 - 1.0]
            resize (tuple): (H, W)
        Returns:
            trajs_e (tensor): Estimated trajectory of points through the video [B, N, S, 2]. range: [0.0 - 1.0]
        """
        points = points.to(self.device)
        rgbs = video.permute(0, 1, 4, 2, 3).float().to(self.device) # B, S, C, H, W
        B, S, C, H, W = rgbs.shape
        if resize is not None:
            rgbs_ = rgbs.reshape(B*S, C, H, W)
            H_, W_ = resize[0], resize[1]
            rgbs_ = F.interpolate(rgbs_, (H_, W_), mode='bilinear')
            H, W = H_, W_
            rgbs = rgbs_.reshape(B, S, C, H, W)
            _, S, C, H, W = rgbs.shape
        # converting to pixel level values
        points[...,0] *= W - 1
        points[...,1] *= H - 1
        _, N, _ = points.shape

        #print(points.shape)
        trajs_e = points.clone().unsqueeze(1).repeat(1, S, 1, 1)
        #print(trajs_e.min(), trajs_e.max(), trajs_e.sum())
        #print(trajs_e.shape)

        with torch.no_grad():
            preds, _ = self.model(rgbs, points, iters=4)
            #print(torch.cuda.memory_allocated()/ (1024 ** 2))
        
        trajs_e = preds[-1]
        
        trajs_e[...,0] /= W - 1
        trajs_e[...,1] /= H - 1
        points[...,0] /= W - 1
        points[...,1] /= H - 1
        trajs_e = trajs_e.cpu().permute(0, 2, 1, 3)
        return trajs_e
    
    def train(self, dataloaders, dataset_size, log_dir, ckpt_path, epochs=10):        
        """ Train the model on the provided dataset.
        Args:
            dataloaders (Dict): {'train':torch train dataloader, 'val':torch train dataloader}
            dataset_size (Dict): {'train': int, 'val':int}
            epoch (int, optional): number of epochs to train. Defaults to 10.
        """
        lr = 5e-4
        weight_decay = 1e-6
        steps_per_epoch = dataset_size['train']
        optimizer, scheduler = fetch_optimizer(lr, weight_decay, 1e-8, epochs, steps_per_epoch, self.model.parameters())
        
        metrics = getMetricsDict()
        metrics['loss'] = 0.0
        best_loss = 100000.0
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir=log_dir)
        for epoch in tqdm(range(epochs)):
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                for rgbs_list, trajs_g_list, visibs_g_list in dataloaders[phase]:
                    metrics_b = getMetricsDict()
                    metrics_b['loss'] = 0.0
                    batch_loss = []
                    batch_size = 0
                    for i in range(len(rgbs_list)):
                        batch_size += 1
                        rgbs, trajs_g, visibs_g = rgbs_list[i], trajs_g_list[i], visibs_g_list[i]
                        rgbs, trajs_g, visibs_g = rgbs.unsqueeze(0), trajs_g.unsqueeze(0), visibs_g.unsqueeze(0)
                        rgbs = rgbs.permute(0, 1, 4, 2, 3).float().to(self.device) # B, S, C, H, W
                        trajs_g = trajs_g.permute(0, 2, 1, 3).to(self.device)
                        valids = visibs_g.permute(0, 2, 1).to(self.device) #B, S, N
                        B, S, C, H, W = rgbs.shape
                        trajs_g[...,0] *= W - 1
                        trajs_g[...,1] *= H - 1
                        _, N, _, _ = trajs_g.shape
                        points_0 = trajs_g[:,0,:,:] # taking all points from frame 0
                        #trajs_e = points_0.unsqueeze(1).repeat(1,S,1,1).to(self.device)# B, S, N, 2
                        if phase == 'train':
                            self.model.train()  # Set model to training mode
                            preds, loss = self.model(rgbs, points_0, trajs_g=trajs_g, vis_g=valids, valids=valids)
                            
                            batch_loss.append(loss.mean()) # mean() is for parallel GPU computing
                         
                        else:
                            self.model.eval()   # Set model to evaluate mode
                            with torch.no_grad():
                                torch.cuda.empty_cache()
                                preds, loss = self.model(rgbs, points_0, trajs_g=trajs_g, vis_g=valids, valids=valids)
                        
                        trajs_e = preds[-1]

                        trajs_e[...,0] /= W - 1
                        trajs_e[...,1] /= H - 1
                        trajs_g[...,0] /= W - 1
                        trajs_g[...,1] /= H - 1
                        points_0[...,0] /= W - 1
                        points_0[...,1] /= H - 1
                        #print(points_0.shape, trajs_g.shape, trajs_e.shape, visibs_g.shape)
                        outputs = evaluate.compute_metrics(points_0.cpu().numpy(), trajs_g.permute(0, 2, 1, 3).cpu().numpy(), 
                                visibs_g.cpu().numpy(), trajs_e.detach().permute(0, 2, 1, 3).cpu().numpy(), visibs_g.cpu().numpy())
                        for key, value in outputs.items():
                            metrics_b[key] += value
                        
                        metrics_b['loss'] += loss.mean().item()
                             
                    
                    if phase == 'train':
                        torch.stack(batch_loss).mean().backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                    
                    for key, value in metrics_b.items():
                        metrics[key] += value/batch_size
                
                for key in metrics.keys():
                    metrics[key] /= dataset_size[phase]
                if phase == 'val':
                    #saving last epochs
                    saverloader.save(os.path.join(ckpt_path, 'lasts') , optimizer, self.model.module, epoch, scheduler=scheduler)
                    if metrics['loss'] < best_loss:
                        #saving best model based on validation loss
                        saverloader.save(ckpt_path , optimizer, self.model.module, epoch, scheduler=scheduler)
                        best_loss = metrics['loss']
                        
                print(f"Epoch:{epoch}=> {phase}-> loss:{metrics['loss']:0.2f}, d_avg:{metrics['average_pts_within_thresh']:0.2f}")
                writer.add_scalars(f"{phase}", metrics, epoch)
                writer.flush()
                for key in metrics.keys():
                    metrics[key] = 0.0
                
        writer.close()
