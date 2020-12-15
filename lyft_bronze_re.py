import os
import random
import time
import math
from typing import Dict

import numpy as np
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter
from prettytable import PrettyTable
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings("ignore")

from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset, AgentDataset
from l5kit.rasterization import build_rasterizer
from l5kit.configs import load_config_data
from l5kit.visualization import PREDICTED_POINTS_COLOR, TARGET_POINTS_COLOR, draw_trajectory
from l5kit.geometry import transform_points
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, create_chopped_dataset
from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace


def self_mkdir(cfg) -> object:
    # save parameters to yaml
    Path(cfg["save_path"]).mkdir(parents=True, exist_ok=True)
    ff = open(os.path.join(cfg["save_path"],'cfg.yaml'), 'w')
    yaml.dump(cfg, ff, allow_unicode=True, default_flow_style=False)

    log_path = os.path.join(cfg["save_path"], 'log/')
    Path(log_path).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_path)
    return writer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    print(f'set seed to {seed} successfully')
    

def pytorch_neg_multi_log_likelihood_batch(
    gt: Tensor, pred: Tensor, confidences: Tensor, avails: Tensor
) -> Tensor:
    gt = torch.unsqueeze(gt, 1) 
    avails = avails[:, None, :, None]  

   
    error = torch.sum(((gt - pred) * avails) ** 2, dim=-1)  
   

    with np.errstate(divide="ignore"):  
        error = torch.log(confidences) - 0.5 * torch.sum(error, dim=-1)  # reduce time

    max_value, _ = error.max(dim=1, keepdim=True)  # error are negative at this point, so max() gives the minimum one
    error = -torch.log(torch.sum(torch.exp(error - max_value), dim=-1, keepdim=True)) - max_value  # reduce modes
    
    return torch.mean(error)



class LyftMultiModel(nn.Module):
    def __init__(self, cfg):
        super(LyftMultiModel, self).__init__()
        cfg['bms'] = cfg['bms'] if cfg['bms'] else [None for _d in cfg['ds']]
        cfg['gws'] = cfg['gws'] if cfg['gws'] else [None for _d in cfg['ds']]
       
        stage_params = list(zip(cfg['ds'], cfg['ws'], cfg['ss'], cfg['bms'], cfg['gws']))

        stem_fun = get_stem_fun(cfg['stem_type'])
        self.stem = stem_fun(25, cfg['stem_w'])
        block_fun = get_block_fun(cfg['block_type'])
        prev_w = cfg['stem_w']
        for i, (d, w, s, bm, gw) in enumerate(stage_params):
            name = "s{}".format(i + 1)
            self.add_module(name, AnyStage(prev_w, w, s, d, block_fun, bm, gw, cfg['se_r']))
            prev_w = w
        self.head = fcnet(prev_w)


    def forward(self, x):
        for module in self.children():
            x = module(x)
    
        return x

class fcnet(nn.Module):
    def __init__(self, in_features):
        super(fcnet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))#1512 -> 4096 ->303
        self.fc = nn.Sequential(
            nn.Linear(in_features, 4096),
            nn.Linear(4096, 303))
    
    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1) 
        x = self.fc(x) #303
        
        bs, _ = x.shape
        pred, score = torch.split(x, 300, dim=1)#3 , 300
        pred = pred.view(bs, 3, 50, 2) 
        assert score.shape == (bs, 3)
        score = torch.softmax(score, dim=1)
        return pred, score
        
        
        

def get_stem_fun(stem_type):
    """Retrieves the stem function by name."""
    stem_funs = {
        "simple_stem_in": SimpleStemIN,
    }
    err_str = "Stem type '{}' not supported"
    assert stem_type in stem_funs.keys(), err_str.format(stem_type)
    return stem_funs[stem_type]


def get_block_fun(block_type):
    """Retrieves the block function by name."""
    block_funs = {
        "vanilla_block": VanillaBlock,
        "res_basic_block": ResBasicBlock,
        "res_bottleneck_block": ResBottleneckBlock,
    }
    err_str = "Block type '{}' not supported"
    assert block_type in block_funs.keys(), err_str.format(block_type)
    return block_funs[block_type]


class AnyStage(nn.Module):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(self, w_in, w_out, stride, d, block_fun, bm, gw, se_r):
        super(AnyStage, self).__init__()
        for i in range(d):
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            name = "b{}".format(i + 1)
            self.add_module(name, block_fun(b_w_in, w_out, b_stride, bm, gw, se_r))

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class VanillaBlock(nn.Module):
    """Vanilla block: [3x3 conv, BN, Relu] x2."""

    def __init__(self, w_in, w_out, stride, bm=None, gw=None, se_r=None):
        err_str = "Vanilla block does not support bm, gw, and se_r options"
        assert bm is None and gw is None and se_r is None, err_str
        super(VanillaBlock, self).__init__()
        self.a = nn.Conv2d(w_in, w_out, 3, stride=stride, padding=1, bias=False)
        self.a_bn = nn.BatchNorm2d(w_out, eps=1e-5, momentum=0.1)
        self.a_relu = nn.ReLU(inplace=True)
        self.b = nn.Conv2d(w_out, w_out, 3, stride=1, padding=1, bias=False)
        self.b_bn = nn.BatchNorm2d(w_out, eps=1e-5, momentum=0.1)
        self.b_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x



class BasicTransform(nn.Module):
    """Basic transformation: [3x3 conv, BN, Relu] x2."""

    def __init__(self, w_in, w_out, stride):
        super(BasicTransform, self).__init__()
        self.a = nn.Conv2d(w_in, w_out, 3, stride=stride, padding=1, bias=False)
        self.a_bn = nn.BatchNorm2d(w_out, eps=1e-5, momentum=0.1)
        self.a_relu = nn.ReLU(inplace=True)
        self.b = nn.Conv2d(w_out, w_out, 3, stride=1, padding=1, bias=False)
        self.b_bn = nn.BatchNorm2d(w_out, eps=1e-5, momentum=0.1)
        self.b_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x



class ResBasicBlock(nn.Module):
    """Residual basic block: x + F(x), F = basic transform."""

    def __init__(self, w_in, w_out, stride, bm=None, gw=None, se_r=None):
        err_str = "Basic transform does not support bm, gw, and se_r options"
        assert bm is None and gw is None and se_r is None, err_str
        super(ResBasicBlock, self).__init__()
        self.proj_block = (w_in != w_out) or (stride != 1)
        if self.proj_block:
            self.proj = nn.Conv2d(w_in, w_out, 1, stride=stride, padding=0, bias=False)
            self.bn = nn.BatchNorm2d(w_out, eps=1e-5, momentum=0.1)
        self.f = BasicTransform(w_in, w_out, stride)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        x = self.relu(x)
        return x



class SE(nn.Module):
    """Squeeze-and-Excitation (SE) block: AvgPool, FC, ReLU, FC, Sigmoid."""

    def __init__(self, w_in, w_se):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.f_ex = nn.Sequential(
            nn.Conv2d(w_in, w_se, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(w_se, w_in, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.f_ex(self.avg_pool(x))



class BottleneckTransform(nn.Module):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

    def __init__(self, w_in, w_out, stride, bm, gw, se_r):
        super(BottleneckTransform, self).__init__()
        w_b = int(round(w_out * bm))
        g = w_b // gw
        self.a = nn.Conv2d(w_in, w_b, 1, stride=1, padding=0, bias=False)
        self.a_bn = nn.BatchNorm2d(w_b, eps=1e-5, momentum=0.1)
        self.a_relu = nn.ReLU(inplace=True)
        self.b = nn.Conv2d(w_b, w_b, 3, stride=stride, padding=1, groups=g, bias=False)
        self.b_bn = nn.BatchNorm2d(w_b, eps=1e-5, momentum=0.1)
        self.b_relu = nn.ReLU(inplace=True)
        if se_r:
            w_se = int(round(w_in * se_r))
            self.se = SE(w_b, w_se)
        self.c = nn.Conv2d(w_b, w_out, 1, stride=1, padding=0, bias=False)
        self.c_bn = nn.BatchNorm2d(w_out, eps=1e-5, momentum=0.1)
        self.c_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x



class ResBottleneckBlock(nn.Module):
    """Residual bottleneck block: x + F(x), F = bottleneck transform."""

    def __init__(self, w_in, w_out, stride, bm=1.0, gw=1, se_r=None):
        super(ResBottleneckBlock, self).__init__()
        # Use skip connection with projection if shape changes
        self.proj_block = (w_in != w_out) or (stride != 1)
        if self.proj_block:
            self.proj = nn.Conv2d(w_in, w_out, 1, stride=stride, padding=0, bias=False)
            self.bn = nn.BatchNorm2d(w_out, eps=1e-5, momentum=0.1)
        self.f = BottleneckTransform(w_in, w_out, stride, bm, gw, se_r)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        x = self.relu(x)
        return x


class SimpleStemIN(nn.Module):
    """Simple stem for ImageNet: 3x3, BN, ReLU."""

    def __init__(self, w_in, w_out):
        super(SimpleStemIN, self).__init__()
        self.conv = nn.Conv2d(w_in, w_out, 3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(w_out, eps=1e-5, momentum=0.1)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x




def forward(data, model, criterion = pytorch_neg_multi_log_likelihood_batch):
    inputs = data["image"].cuda()
    target_availabilities = data["target_availabilities"].cuda()
    targets = data["target_positions"].cuda()

    # matrix = data["world_to_image"].cuda()
    # centroid = data["centroid"][:,None,:].to(torch.float).cuda()
    # targets = targets + centroid
    # targets = torch.cat([targets,torch.ones((targets.shape[0],50,1)).cuda()], dim=2)
    # targets = torch.matmul(matrix.to(torch.float), targets.transpose(1,2))
    # targets = targets.transpose(1,2)[:,:,:2]
    # bias = torch.tensor([56.25, 112.5])[None,None,:].cuda()
    # targets = targets - bias

    # Forward pass
    preds, confidences = model(inputs)
    loss = criterion(targets, preds, confidences, target_availabilities)

    # matrix_inv = torch.inverse(matrix)
    # preds = preds + bias[:,None,:,:]
    # preds = torch.cat([preds,torch.ones((preds.shape[0],3,50,1)).cuda()], dim=3)
    # preds = torch.stack([torch.matmul(matrix_inv.to(torch.float), preds[:,i].transpose(1,2)) 
    #                     for i in range(3)], dim=1)
    # preds = preds.transpose(2,3)[:,:,:,:2]
    # preds = preds - centroid[:,None,:,:]
    return loss, preds, confidences


def evaluate(cfg, model, dm, rasterizer, first_time, iters, eval_dataloader, eval_gt_path):
    if first_time:
        num_frames_to_chop = 100
        print("min_future_steps: ",MIN_FUTURE_STEPS)
        eval_cfg = cfg["val_data_loader"]
        eval_base_path = create_chopped_dataset(dm.require(eval_cfg["key"]), cfg["raster_params"]["filter_agents_threshold"], 
                                    num_frames_to_chop, cfg["model_params"]["future_num_frames"], MIN_FUTURE_STEPS)
        eval_zarr_path = str(Path(eval_base_path) / Path(dm.require(eval_cfg["key"])).name)
        eval_mask_path = str(Path(eval_base_path) / "mask.npz")
        eval_gt_path = str(Path(eval_base_path) / "gt.csv")
        eval_zarr = ChunkedDataset(eval_zarr_path).open()
        eval_mask = np.load(eval_mask_path)["arr_0"]
        eval_dataset = AgentDataset(cfg, eval_zarr, rasterizer, agents_mask=eval_mask)
        eval_dataloader = DataLoader(eval_dataset, shuffle=eval_cfg["shuffle"], batch_size=eval_cfg["batch_size"], 
                                    num_workers=eval_cfg["num_workers"])
        print(eval_dataset)
        first_time = False

    model.eval()
    torch.set_grad_enabled(False)

    future_coords_offsets_pd = []
    timestamps = []
    confidences_list = []
    agent_ids = []
    progress_bar = tqdm(eval_dataloader)
    for data in progress_bar:
        _, preds, confidences = forward(data, model)
        
        # convert agent coordinates into world offsets
        preds = preds.cpu().numpy()
        world_from_agents = data["world_from_agent"].numpy()
        centroids = data["centroid"].numpy()
        coords_offset = []
        
        for idx in range(len(preds)):
            for mode in range(3):
                preds[idx, mode, :, :] = transform_points(preds[idx, mode, :, :], world_from_agents[idx]) - centroids[idx][:2]
        
        future_coords_offsets_pd.append(preds.copy())
        confidences_list.append(confidences.cpu().numpy().copy())
        timestamps.append(data["timestamp"].numpy().copy())
        agent_ids.append(data["track_id"].numpy().copy())
    
    model.train()
    torch.set_grad_enabled(True)

    pred_path = os.path.join(cfg["save_path"],f"pred_{iters}.csv")

    write_pred_csv(pred_path,
        timestamps=np.concatenate(timestamps),
        track_ids=np.concatenate(agent_ids),
        coords=np.concatenate(future_coords_offsets_pd),
        confs = np.concatenate(confidences_list)
        )

        
    metrics = compute_metrics_csv(eval_gt_path, pred_path, [neg_multi_log_likelihood, time_displace])
    for metric_name, metric_mean in metrics.items():
        print(metric_name, metric_mean)

    return first_time, eval_dataloader, eval_gt_path


def main():

    cfg = {
        'save_path': "./regnetx/",
        'seed': 39,
        'data_path': "kagglepath",
        'stem_type':'simple_stem_in',
        'stem_w':32,
        'block_type':'res_bottleneck_block',
        'ds':[2,5,13,1],
        'ws':[72,216,576,1512],
        'ss':[2,2,2,2],
        'bms':[1.0,1.0,1.0,1.0],
        'gws':[24,24,24,24],
        'se_r':0.25,
        'model_params': {
            'history_num_frames': 10, #3+20+2=25 25*h*w
            'history_step_size': 1,
            'history_delta_time': 0.1,
            'future_num_frames': 50, # 1512 -> 50*2*3+3=303 nn.linear
            'future_step_size': 1,
            'future_delta_time': 0.1,
            'opt_type' : 'adam',
            'lr': 3e-4,
            'w_decay': 0,
            'reduce_type':'stone',
0            'r_factor': 0.5,
            'r_step' : [200_000, 300_000, 360_000, 420_000, 480_000, 540_000],
            'weight_path': './050.pth',
        },

        'raster_params': {
            'raster_size': [224, 224],
            'pixel_size': [0.5, 0.5],
            'ego_center': [0.25, 0.5],
            'map_type': 'py_semantic',
            'satellite_map_key': 'aerial_map/aerial_map.png',
            'semantic_map_key': 'semantic_map/semantic_map.pb',
            'dataset_meta_key': 'meta.json',
            'filter_agents_threshold': 0.5
        },

        'train_data_loader': {
            'key': 'scenes/train.zarr',
            'batch_size': 16,
            'shuffle': True,
            'num_workers': 20
        },

        'val_data_loader': {
            'key': "scenes/validate.zarr",
            'batch_size': 16,
            'shuffle': False,
            'num_workers': 20
        },

        'test_data_loader': {
            'key': 'scenes/test.zarr',
            'batch_size': 32,
            'shuffle': False,
            'num_workers': 4
        },

        'train_params': {
            'max_num_steps': 600_000,
            'checkpoint_every_n_steps': 100_000,
            'eval_every_n_steps' : 100_000,
        }
    }
    
    writer = self_mkdir(cfg)
    set_seed(cfg["seed"])
    os.environ["L5KIT_DATA_FOLDER"] = cfg["data_path"]
    dm = LocalDataManager(None)

    model = LyftMultiModel(cfg)#1
    weight_path = cfg["model_params"]["weight_path"]
    if weight_path:
        model.load_state_dict(torch.load(weight_path))
    else: print('no check points')
    model.cuda()
    
    m_params = cfg["model_params"]
    if m_params['opt_type'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=m_params["lr"], weight_decay=m_params['w_decay'],)
    elif m_params['opt_type'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=m_params["lr"], weight_decay=m_params['w_decay'],)
    else:
        assert False, 'cfg opt_type error'

    if m_params['reduce_type'] == 'stone':
        lr_sche = optim.lr_scheduler.MultiStepLR(optimizer, m_params['r_step'], 
            gamma=m_params['r_factor'], last_epoch=-1)
    else:
        assert False, 'cfg reduce_type error'

    Training = False
    if Training:
        train_cfg = cfg["train_data_loader"]
        rasterizer = build_rasterizer(cfg, dm)
        train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
        train_dataset = AgentDataset(cfg, train_zarr, rasterizer)
        train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"], 
                                    num_workers=train_cfg["num_workers"])
        print(train_dataset)
        
        tr_it = iter(train_dataloader)
        progress_bar = tqdm(range(cfg["train_params"]["max_num_steps"]))
        
        
        model_name = cfg["model_params"]["model_name"]
        first_time = True
        eval_dataloader = None
        eval_gt_path = None
        model.train()
        torch.set_grad_enabled(True)
        loss_ten = 0
        for i in progress_bar:
            try:
                data = next(tr_it)
            except StopIteration:
                tr_it = iter(train_dataloader)
                data = next(tr_it)
            
            loss, _, _ = forward(data, model)#2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_sche.step()
            
            writer.add_scalar('train_loss', loss.item(), i)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], i)
            if i == 10:
                loss_ten = loss.item()
            progress_bar.set_description(f"loss: {loss.item()} and {loss_ten}")

            if (i+1) % cfg['train_params']['checkpoint_every_n_steps'] == 0:
                torch.save(model.state_dict(), os.path.join(cfg['save_path'],f'{model_name}_{i}.pth'))
            
            if (i+1) % cfg['train_params']['eval_every_n_steps'] == 0: #3
                first_time, eval_dataloader, eval_gt_path = evaluate(cfg, model, dm, rasterizer, first_time, i+1, eval_dataloader, eval_gt_path)


    test_cfg = cfg["test_data_loader"]
    rasterizer = build_rasterizer(cfg, dm)
    test_zarr = ChunkedDataset(dm.require(test_cfg["key"])).open()
    test_mask = np.load(os.path.join(cfg["data_path"],'scenes/mask.npz'))["arr_0"]
    test_dataset = AgentDataset(cfg, test_zarr, rasterizer, agents_mask=test_mask)
    test_dataloader = DataLoader(test_dataset,shuffle=test_cfg["shuffle"],batch_size=test_cfg["batch_size"],
                                num_workers=test_cfg["num_workers"])
    print(test_dataset)

    model.eval()
    torch.set_grad_enabled(False)

    # store information for evaluation
    future_coords_offsets_pd = []
    timestamps = []
    confidences_list = []
    agent_ids = []

    progress_bar = tqdm(test_dataloader)
    
    for data in progress_bar:
        
        _, preds, confidences = forward(data, model)
    
        #fix for the new environment
        preds = preds.cpu().numpy()
        world_from_agents = data["world_from_agent"].numpy()
        centroids = data["centroid"].numpy()
        coords_offset = []
        
        # convert into world coordinates and compute offsets
        for idx in range(len(preds)):
            for mode in range(3):
                preds[idx, mode, :, :] = transform_points(preds[idx, mode, :, :], world_from_agents[idx]) - centroids[idx][:2]
    
        future_coords_offsets_pd.append(preds.copy())
        confidences_list.append(confidences.cpu().numpy().copy())
        timestamps.append(data["timestamp"].numpy().copy())
        agent_ids.append(data["track_id"].numpy().copy())

    pred_path = 'submission.csv'
    write_pred_csv(pred_path,
            timestamps=np.concatenate(timestamps),
            track_ids=np.concatenate(agent_ids),
            coords=np.concatenate(future_coords_offsets_pd),
            confs = np.concatenate(confidences_list)
            )


if __name__ == "__main__":
    main()

    