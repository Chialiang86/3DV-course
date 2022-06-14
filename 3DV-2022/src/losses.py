import torch
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)

# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# loss = 
	# implement some loss for binary voxel grids
	return prob_loss

def chamfer_loss(point_cloud_src, point_cloud_tgt, n_points):
    sample_trg = sample_points_from_meshes(point_cloud_src, n_points)
    sample_pred = sample_points_from_meshes(point_cloud_tgt, n_points)
    loss_chamfer = chamfer_distance(sample_trg, sample_pred)
    # loss_edge = mesh_edge_loss(new_src_mesh)
    # loss_normal = mesh_normal_consistency(new_src_mesh)
    # loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")
    # # Weighted sum of the losses
    # loss = loss_chamfer * w_chamfer + loss_edge * w_edge + loss_normal * w_normal + loss_laplacian * w_laplacian
	# implement chamfer loss from scratch
    return loss_chamfer

def smoothness_loss(mesh_src):
	loss_laplacian = mesh_laplacian_smoothing(mesh_src, method="uniform")
	# implement laplacian smoothening loss
	return loss_laplacian

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