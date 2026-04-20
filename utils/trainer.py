import numpy as np
import torch
import torch.nn.functional as F
from utils import basic



def sequence_loss(flow_preds, flow_gt, vis, valids, gamma=0.8):
    """ Loss function defined over sequence of flow predictions """
    B, S, N, D = flow_gt.shape
    assert(D==2)
    B, S1, N = vis.shape
    B, S2, N = valids.shape
    assert(S==S1)
    assert(S==S2)
    n_predictions = len(flow_preds)    
    flow_loss = 0.0
    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        flow_pred = flow_preds[i].permute(0, 2, 1, 3)
        i_loss = (flow_pred - flow_gt).abs() # B,S,N,2
        i_loss = torch.mean(i_loss, dim=3) # B,S,N
        #flow_loss += i_weight * basic.reduce_masked_mean(i_loss, valids)
        flow_loss += i_weight * basic.reduce_masked_mean(i_loss, valids, dim=(1, 2), keepdim=True).squeeze(1)
    flow_loss = flow_loss/n_predictions
    return flow_loss




def huber_loss(heat_maps_g, heat_maps_e, delta=0.5):
    heat_maps_g = heat_maps_g.float()
    # print(heat_maps_e.dtype, heat_maps_g.dtype)
    # raise KeyboardInterrupt
    loss = 0.0
    for i in range(heat_maps_g.shape[0]): #for batch
        for s in range(heat_maps_g.shape[1]): #for seq
            loss += F.huber_loss(heat_maps_g[i,s], heat_maps_e[i,s], delta=delta)

    return loss * 10






def getTrainParams(model):
  """Print the number weights is active to be trained

  Args:
      model (pytorch model): _description_
  """    
  parameters = filter(lambda p: p.requires_grad, model.parameters())
  parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
  print('Trainable Parameters: %.3fM' % parameters)

def requires_grad(parameters, flag=True):
  for p in parameters:
    p.requires_grad = flag

def fetch_optimizer(lr, wdecay, epsilon, epochs, steps_per_epoch, params):
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wdecay, eps=epsilon)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
       optimizer, lr, epochs=epochs, steps_per_epoch=steps_per_epoch, pct_start=0.1, cycle_momentum=False, anneal_strategy='linear')
    return optimizer, scheduler
