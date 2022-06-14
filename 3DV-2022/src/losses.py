import torch
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
import numpy as np

# define losses
def voxel_loss(voxel_src,voxel_tgt):
    diff = (voxel_src - voxel_tgt)**2
    loss_voxel = diff.mean()
    return loss_voxel

def chamfer_loss(point_cloud_src, point_cloud_tgt):
    loss_chamfer, _ = chamfer_distance(point_cloud_src, point_cloud_tgt)
    # # Weighted sum of the losses
    # loss = loss_chamfer * w_chamfer + loss_edge * w_edge + loss_normal * w_normal + loss_laplacian * w_laplacian
    return loss_chamfer

def edge_loss(new_src_mesh):
    loss_edge = mesh_edge_loss(new_src_mesh)
    return loss_edge

def normal_loss(new_src_mesh):
    loss_normal = mesh_normal_consistency(new_src_mesh)
    return loss_normal

def smoothness_loss(mesh_src):
	loss_laplacian = mesh_laplacian_smoothing(mesh_src, method="uniform")
	# implement laplacian smoothening loss
	return loss_laplacian

def calculate_loss(ground_truth, predictions, cfg):
    if cfg['dtype'] == 'voxel':
        assert isinstance(ground_truth, torch.Tensor) and isinstance(predictions, torch.Tensor)
        loss = voxel_loss(ground_truth, predictions)
    elif cfg['dtype'] == 'point':
        assert isinstance(ground_truth, torch.Tensor) and isinstance(predictions, torch.Tensor)
        loss = chamfer_loss(ground_truth, predictions)
    elif cfg['dtype'] == 'mesh':
        assert isinstance(ground_truth, list) and \
            isinstance(predictions, list) and \
            len(ground_truth) == len(predictions) and\
            isinstance(ground_truth[0], Meshes) and isinstance(predictions[0], Meshes)

        loss = 0.0
        for i in range(len(ground_truth)):
            sample_src = sample_points_from_meshes(ground_truth[i], 5000)
            sample_trg = sample_points_from_meshes(predictions[i], 5000)
            loss_chamfer = chamfer_loss(sample_src, sample_trg)
            loss_edge = edge_loss(predictions[i])
            loss_smooth = smoothness_loss(predictions[i])
            loss_normal = normal_loss(predictions[i])
            loss += cfg['w_chamfer'] * loss_chamfer + cfg['w_edge'] * loss_edge + \
                    cfg['w_smooth'] * loss_smooth  + cfg['w_normal'] * loss_normal   
        loss /= len(ground_truth)
    return loss

class ChamferDistanceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, points1: torch.Tensor, points2: torch.Tensor, w1=1.0, w2=1.0, each_batch=False):
        self.check_parameters(points1)
        self.check_parameters(points2)

        diff = points1[:, :, None, :] - points2[:, None, :, :]
        dist = torch.sum(diff * diff, dim=3)
        dist1 = dist
        dist2 = torch.transpose(dist, 1, 2)

        dist1 = torch.sqrt(dist1)**2
        dist2 = torch.sqrt(dist2)**2

        dist_min1, indices1 = torch.min(dist1, dim=2)
        dist_min2, indices2 = torch.min(dist2, dim=2)

        loss1 = dist_min1.mean(1)
        loss2 = dist_min2.mean(1)
        
        loss = w1 * loss1 + w2 * loss2

        if not each_batch:
            loss = loss.mean()

        return loss

    @staticmethod
    def check_parameters(points: torch.Tensor):
        assert points.ndimension() == 3  # (B, N, 3)
        assert points.size(-1) == 3