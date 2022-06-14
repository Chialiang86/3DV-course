import argparse
import time
import os
import torch
import src.losses as losses
from src.dataset import ShapeNetDB
from src.model import SingleViewto3D
from src.losses import calculate_loss
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm

# for rendering
# from mpl_toolkits.mplot3d import Axes3D
from renderer import Scene
from pytorch3d.structures import Pointclouds, Meshes, Volumes
from pytorch3d.datasets import (
    collate_batched_meshes,
)
from pytorch3d.renderer import (
    TexturesVertex,
)


def plot_pointcloud(gt, pred, fname):
    plt.clf()
    fig = plt.figure(figsize=(10, 5))
    plt.suptitle(f'Inference Result')

    # for gt
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    # ax = Axes3D(fig)
    x, y, z = gt.detach().clone().cpu().squeeze().unbind(1)  
    ax.scatter3D(-x, z, -y)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title('Ground Truth')
    ax.view_init(190, 30)

    # for pred
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    # ax = Axes3D(fig)
    x, y, z = pred.detach().clone().cpu().squeeze().unbind(1)  
    ax.scatter3D(-x, z, -y)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title('Prediction')
    ax.view_init(190, 30)

    plt.savefig(fname)
    print(f'{fname} saved')

def plot_voxel(gt, pred, fname):
    voxel_gt = gt.detach().clone().cpu().squeeze()
    voxel_pred = pred.detach().clone().cpu().squeeze()
    voxel_gt = torch.where(voxel_gt > 0.5, 1, 0)
    voxel_pred = torch.where(voxel_pred > 0.5, 1, 0)
    voxel_gt = torch.permute(voxel_gt, (0, 2, 1))
    voxel_pred = torch.permute(voxel_pred, (0, 2, 1))

    plt.clf()
    fig = plt.figure(figsize=(10, 5))
    plt.suptitle(f'Inference Result')

    # for gt
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.voxels(voxel_gt)
    ax.set_xlabel('x')
    ax.invert_xaxis()
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.invert_zaxis()
    ax.set_title('Ground Truth')
    ax.view_init(190, 30)

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.voxels(voxel_pred)
    ax.set_xlabel('x')
    ax.invert_xaxis()
    ax.set_ylabel('z')
    ax.invert_zaxis()
    ax.set_zlabel('y')
    ax.set_title('Prediction')
    ax.view_init(190, 30)

    plt.savefig(fname)
    print(f'{fname} saved')

def plot_render(scene, gt, pred, fname, cfg):
    if cfg['dtype'] == 'point':
        scene.set_point_rasterizer(image_size=256)
        scene.set_point_renderer()

        features_gt = torch.ones_like(gt.detach())
        point_cloud_gt = Pointclouds(points=[gt.detach()], features=[features_gt])
        images_gt = scene.renderer(point_cloud_gt)
        
        features_pred = torch.ones_like(pred.detach())
        point_cloud_pred = Pointclouds(points=[pred.detach()], features=[features_pred])
        images_pred = scene.renderer(point_cloud_pred)
    elif cfg['dtype'] == 'mesh':
        scene.set_mesh_rasterizer(image_size=256)
        scene.set_mesh_renderer()

        images_gt = scene.renderer(gt.detach())
        images_pred = scene.renderer(pred.detach())

    plt.clf()
    plt.figure(figsize=(20, 10))
    plt.suptitle(f'Inference Result')
    plt.subplot(1, 2, 1)
    plt.imshow(images_gt[0, ..., :3].cpu().numpy())
    plt.axis("off")
    plt.title('Ground Truth')
    plt.subplot(1, 2, 2)
    plt.imshow(images_pred[0, ..., :3].cpu().numpy())
    plt.axis("off")
    plt.title('Prediction')
    plt.savefig(fname)
    print(f'{fname} saved')


def val_point_or_voxel(loader_val : torch.utils.data.DataLoader,
                        model : SingleViewto3D, 
                        scene : Scene,
                        cfg, out_dir):
    dtype = cfg['dtype']

    print(type(loader_val))
    loss_list = []
    cnt = 0
    val_loader = iter(loader_val)
    for images_gt, ground_truth_3d, object_id in tqdm(val_loader):

        cnt += 1
        if cnt == 10:
            break

        # prediction
        images_gt, ground_truth_3d = images_gt.cuda(), ground_truth_3d.cuda()
        prediction_3d = model(images_gt, cfg)
        loss = calculate_loss(ground_truth_3d, prediction_3d, cfg).cpu().item()
        
        loss_vis = loss.cpu().item()
        loss_list.append(loss_vis)

        # torch.save(prediction_3d.detach().cpu(), f'{out_dir}/pre_point_cloud_{cnt}.pt')

        if cfg['dtype'] == "voxel":
            print('rendering voxels ...')
            plot_voxel(ground_truth_3d[0], 
                        prediction_3d[0], 
                        fname=f'{out_dir}/{dtype}_{cnt}.png')
        
        # render points and meshes
        if cfg['dtype'] != "voxel":
            print(f'rendering {dtype} ...')
            plot_pointcloud(ground_truth_3d[0], 
                            prediction_3d[0], 
                            fname=f'{out_dir}/{dtype}_{cnt}.png')
            plot_render(scene,
                        ground_truth_3d[0], 
                        prediction_3d[0],
                        fname=f'{out_dir}/render_{cnt}.png',
                        cfg=cfg)

    plt.clf()
    plt.title(f'Loss Curve (type = {dtype})')
    plt.plot(loss_list)
    plt.savefig(f'{out_dir}/loss_curve.png')

    print('Done!')

def val_mesh(loader_val : torch.utils.data.DataLoader,
                model : SingleViewto3D, 
                scene : Scene,
                cfg, out_dir):
    dtype = cfg['dtype']

    loss_list = []
    cnt = 0

    val_loader = iter(loader_val)
    for data in val_loader:

        cnt += 1
        if cnt == 10:
            break

        images_gt = data['img']
        mesh_verts_gt = data['verts']
        mesh_faces_gt = data['faces']

        # set to cuda
        images_gt = torch.stack(images_gt).cuda()

        verts_pred = model(images_gt, cfg) # just vertices
        
        meshes_gt = []
        meshes_pred = []
        for i in range(cfg['batch_size']):
            verts_rgb = torch.ones_like(mesh_verts_gt[i].cuda())[None]  # (1, V, 3)
            textures = TexturesVertex(verts_features=verts_rgb.cuda())
            mesh_gt = Meshes(
                verts=[mesh_verts_gt[i].cuda()],
                faces=[mesh_faces_gt[i].cuda()],
                textures=textures
            )
            meshes_gt.append(mesh_gt)

            verts_rgb = torch.ones_like(verts_pred[i].cuda())[None]  # (1, V, 3)
            textures = TexturesVertex(verts_features=verts_rgb.cuda())
            mesh_pred = Meshes(
                verts=[verts_pred[i]],
                faces=[model.faces],
                textures=textures
            )
            meshes_pred.append(mesh_pred)

        loss = calculate_loss(meshes_gt, meshes_pred, cfg)
        
        loss_vis = loss.cpu().item()
        loss_list.append(loss_vis)
        print(f'cnt  : {cnt}, loss = {float(loss)}')

        plot_render(scene,
                    meshes_gt[0], 
                    meshes_pred[0],
                    fname=f'{out_dir}/render_{cnt}.png',
                    cfg=cfg)

    plt.clf()
    plt.title(f'Loss Curve (type = {dtype})')
    plt.plot(loss_list)
    plt.savefig(f'{out_dir}/loss_curve.png')

    print('Done!')

# @hydra.main(config_path="configs/", config_name="config.yaml")
def test_model(args, cfg):

    # config working dir
    cwd = os.getcwd()
    dtype = cfg['dtype']

    # config args : pth dir id
    out_dir = f'{cwd}/checkpoint/{args.id}_{dtype}/pred'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        print(f'{out_dir} created')

    print(cfg['data_dir'])
    shapenetdb_val = ShapeNetDB(cfg['data_dir'], cfg['val_data'], dtype)
    
    loader_val = None
    if dtype == 'point' or dtype == 'voxel':
        loader_val = torch.utils.data.DataLoader(
            shapenetdb_val,
            batch_size=cfg['batch_size'],
            num_workers=cfg['num_workers'],
            pin_memory=True,
            drop_last=True)
    elif dtype == 'mesh':
        loader_val = torch.utils.data.DataLoader(
            shapenetdb_val,
            batch_size=cfg['batch_size'],
            num_workers=cfg['num_workers'],
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_batched_meshes)

    model =  SingleViewto3D(cfg)
    model.cuda()
    model.eval()

    pth_path = f'{cwd}/checkpoint/{args.id}_{dtype}/checkpoint_{dtype}.pth'
    if cfg['load_eval_checkpoint']:
        checkpoint = torch.load(pth_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    # for rendering
    device = "cuda:0"
    scene = Scene(device=device)
    scene.set_cam(2, 30, 90)
    scene.set_light(location=[[0.0, 1.0, 0.0]])

    if dtype == 'point' or dtype == 'voxel':
        val_point_or_voxel(loader_val, model, scene, cfg, out_dir)
    elif dtype == 'mesh':
        val_mesh(loader_val, model, scene, cfg, out_dir) 


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--id', '-i', type=str, default='lab0')
    args = parser.parse_args()

    with open("configs/config.yaml", "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    test_model(args, cfg)