import argparse
from cProfile import label
import os
import time
import torch
import src.losses as losses
from tqdm import tqdm
from src.dataset import ShapeNetDB
from src.model import SingleViewto3D
from src.losses import calculate_loss
import yaml
import matplotlib.pyplot as plt

from pytorch3d.structures import Meshes
from pytorch3d.datasets import (
    collate_batched_meshes,
)

# def collate_batched_meshes(batch: List[Dict]):
#     """
#     Take a list of objects in the form of dictionaries and merge them
#     into a single dictionary. This function can be used with a Dataset
#     object to create a torch.utils.data.Dataloader which directly
#     returns Meshes objects.
#     TODO: Add support for textures.

#     Args:
#         batch: List of dictionaries containing information about objects
#             in the dataset.

#     Returns:
#         collated_dict: Dictionary of collated lists. If batch contains both
#             verts and faces, a collated mesh batch is also returned.
#     """
#     if batch is None or len(batch) == 0:
#         return None
#     collated_dict = {}
#     for k in batch[0].keys():
#         collated_dict[k] = [d[k] for d in batch]

#     collated_dict["mesh"] = None
#     if {"verts", "faces"}.issubset(collated_dict.keys()):

#         textures = None
#         if "textures" in collated_dict:
#             textures = TexturesAtlas(atlas=collated_dict["textures"])

#         collated_dict["mesh"] = Meshes(
#             verts=collated_dict["verts"],
#             faces=collated_dict["faces"],
#             textures=textures,
#         )

#     return collated_dict


def train_point_or_voxel(loader_train : torch.utils.data.DataLoader,
                         loader_val : torch.utils.data.DataLoader,
                         model : SingleViewto3D, 
                         optimizer, cfg, start_iter, out_dir):
    dtype = cfg['dtype']
    print("Starting training !")

    loss_list_train = []
    loss_list_val = []
    for step in range(start_iter, cfg['max_iter']):

        iter_start_time = time.time()

        model.train()
        train_loader = iter(loader_train)
        for images_gt, ground_truth_3d, object_id in tqdm(train_loader):

            images_gt, ground_truth_3d = images_gt.cuda(), ground_truth_3d.cuda()

            prediction_3d = model(images_gt, cfg)
            loss = calculate_loss(ground_truth_3d, prediction_3d, cfg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()        

            loss_vis = loss.cpu().item()
            loss_list_train.append(loss_vis)

            iter_time = time.time() - iter_start_time
            print("[%4d/%4d]; ttime: %.0f ; loss: %.5f" % (step, cfg['max_iter'], iter_time, loss_vis))

        model.eval()
        val_loader = iter(loader_val)
        for images_gt, ground_truth_3d, object_id in tqdm(val_loader):

            # prediction
            images_gt, ground_truth_3d = images_gt.cuda(), ground_truth_3d.cuda()
            prediction_3d = model(images_gt, cfg)
            loss = calculate_loss(ground_truth_3d, prediction_3d, cfg)
            
            loss_vis = loss.cpu().item()
            loss_list_val.append(loss_vis)

        
        if (step % cfg['save_freq']) == 0:
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, f'{out_dir}/checkpoint_{dtype}.pth')

    plt.title(f'Loss Curve (type = {dtype})')
    plt.plot(loss_list_train, label='train')
    plt.plot(loss_list_val, label='val')
    plt.legend()
    plt.savefig(f'{out_dir}/loss_curve.png')

    print('Done!')

def train_mesh(loader_train : torch.utils.data.DataLoader,
                loader_val : torch.utils.data.DataLoader,
                model : SingleViewto3D, 
                optimizer, cfg, start_iter, out_dir):
    dtype = cfg['dtype']
    print("Starting training !")

    loss_list_train = []
    loss_list_val = []
    for step in range(start_iter, cfg['max_iter']):

        iter_start_time = time.time()

        model.train()
        train_loader = iter(loader_train)
        for data in tqdm(train_loader):

            images_gt = data['img']
            mesh_verts_gt = data['verts']
            mesh_faces_gt = data['faces']

            # set to cuda
            images_gt = torch.stack(images_gt).cuda()

            verts_pred = model(images_gt, cfg) # just vertices
            
            meshes_gt = []
            meshes_pred = []
            for i in range(cfg['batch_size']):
                mesh_gt = Meshes(
                    verts=[mesh_verts_gt[i].cuda()],
                    faces=[mesh_faces_gt[i].cuda()],
                    textures=None
                )
                meshes_gt.append(mesh_gt)

                mesh_pred = Meshes(
                    verts=[verts_pred[i]],
                    faces=[model.faces],
                    textures=None
                )
                meshes_pred.append(mesh_pred)

            loss = calculate_loss(meshes_gt, meshes_pred, cfg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()        

            loss_vis = loss.cpu().item()
            loss_list_train.append(loss_vis)

            iter_time = time.time() - iter_start_time
            print("[%4d/%4d]; ttime: %.0f ; loss: %.5f" % (step, cfg['max_iter'], iter_time, loss_vis))

        model.eval()
        val_loader = iter(loader_val)
        for data in tqdm(val_loader):

            images_gt = data['img']
            mesh_verts_gt = data['verts']
            mesh_faces_gt = data['faces']

            # set to cuda
            images_gt = torch.stack(images_gt).cuda()

            verts_pred = model(images_gt, cfg) # just vertices
            
            meshes_gt = []
            meshes_pred = []
            for i in range(cfg['batch_size']):
                mesh_gt = Meshes(
                    verts=[mesh_verts_gt[i].cuda()],
                    faces=[mesh_faces_gt[i].cuda()],
                    textures=None
                )
                meshes_gt.append(mesh_gt)

                mesh_pred = Meshes(
                    verts=[verts_pred[i]],
                    faces=[model.faces],
                    textures=None
                )
                meshes_pred.append(mesh_pred)

            loss = calculate_loss(meshes_gt, meshes_pred, cfg)
            
            loss_vis = loss.cpu().item()
            loss_list_train.append(loss_vis)

        if (step % cfg['save_freq']) == 0:
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, f'{out_dir}/checkpoint_{dtype}.pth')

    plt.clf()
    plt.title(f'Loss Curve (type = {dtype})')
    plt.plot(loss_list_train, label='train')
    plt.plot(loss_list_val, label='val')
    plt.legend()
    plt.savefig(f'{out_dir}/loss_curve.png')

    print('Done!')

# @hydra.main(config_path="configs/", config_name="config.yaml")
def train_model(args, cfg):

    # config working dir
    cwd = os.getcwd()
    dtype = cfg['dtype']

    # config args : pth dir id
    out_dir = f'{cwd}/checkpoint/{args.id}_{dtype}'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        print(f'{out_dir} created.')

    shapenetdb_train = ShapeNetDB(cfg['data_dir'], cfg['train_data'], dtype)
    shapenetdb_val = ShapeNetDB(cfg['data_dir'], cfg['val_data'], dtype)

    loader_train, loader_val = None, None
    if dtype == 'point' or dtype == 'voxel':
        loader_train = torch.utils.data.DataLoader(
            shapenetdb_train,
            batch_size=cfg['batch_size'],
            num_workers=cfg['num_workers'],
            pin_memory=True,
            drop_last=True)
        loader_val = torch.utils.data.DataLoader(
            shapenetdb_val,
            batch_size=cfg['batch_size'],
            num_workers=cfg['num_workers'],
            pin_memory=True,
            drop_last=True)
    elif dtype == 'mesh':
        loader_train = torch.utils.data.DataLoader(
            shapenetdb_train,
            batch_size=cfg['batch_size'],
            num_workers=cfg['num_workers'],
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_batched_meshes)
        loader_val = torch.utils.data.DataLoader(
            shapenetdb_val,
            batch_size=cfg['batch_size'],
            num_workers=cfg['num_workers'],
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_batched_meshes)

    model =  SingleViewto3D(cfg)
    model.cuda()
    model.train()

    # ============ preparing optimizer ... ============
    optimizer = torch.optim.Adam(model.parameters(), lr = cfg['lr'])  # to use with ViTs

    start_iter = 0

    if cfg['load_checkpoint']:
        checkpoint = torch.load(f'{out_dir}/checkpoint_{dtype}.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iter = checkpoint['step']
        print(f"Succesfully loaded iter {start_iter}")
    
    if dtype == 'point' or dtype == 'voxel':
        train_point_or_voxel(loader_train, loader_val, model, optimizer, cfg, start_iter, out_dir)
    elif dtype == 'mesh':
        train_mesh(loader_train, loader_val, model, optimizer, cfg, start_iter, out_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--id', '-i', type=str, default='lab0')
    args = parser.parse_args()

    with open("configs/config.yaml", "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    train_model(args, cfg)