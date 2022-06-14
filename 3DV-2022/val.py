import argparse
import time
import os
import torch
import src.losses as losses
from src.dataset import ShapeNetDB
from src.model import SingleViewto3D
from src.losses import ChamferDistanceLoss
import matplotlib.pyplot as plt
import yaml

# for rendering
from renderer import Scene
from pytorch3d.structures import Pointclouds, Meshes, Volumes

import hydra
from omegaconf import DictConfig

cd_loss = ChamferDistanceLoss()


def calculate_loss(predictions, ground_truth, cfg):
    if cfg['dtype'] == 'voxel':
        loss = losses.voxel_loss(predictions,ground_truth)
    elif cfg['dtype'] == 'point':
        loss = cd_loss(predictions, ground_truth)
    elif cfg['dtype'] == 'mesh':

        loss_reg = losses.chamfer_loss(predictions, ground_truth, cfg)
        loss_smooth = losses.smoothness_loss(predictions)

        loss = cfg['w_chamfer'] * loss_reg + cfg['w_smooth'] * loss_smooth        
    return loss

def render_pred(scene, gt, pred, cfg):
    if cfg['dtype'] == 'voxel':
        images_gt = None
        images_pred = None
    elif cfg['dtype'] == 'point':
        scene.set_point_rasterizer(image_size=256)
        scene.set_point_renderer()

        features_gt = torch.ones_like(gt)
        point_cloud_gt = Pointclouds(points=[gt], features=[features_gt])
        images_gt = scene.renderer(point_cloud_gt)
        
        features_pred = torch.ones_like(pred)
        point_cloud_pred = Pointclouds(points=[pred], features=[features_pred])
        images_pred = scene.renderer(point_cloud_pred)
    elif cfg['dtype'] == 'mesh':
        images_gt = None
        images_pred = None   

    return images_gt[0, ..., :3].cpu().numpy(), images_pred[0, ..., :3].cpu().numpy()

# @hydra.main(config_path="configs/", config_name="config.yaml")
def test_model(args, cfg):

    # config working dir
    cwd = os.getcwd()

    # config args : pth dir id
    out_dir = f'{cwd}/checkpoint/{args.id}/pred'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        print(f'{out_dir} created')

    print(cfg['data_dir'])
    shapenetdb = ShapeNetDB(cfg['data_dir'], cfg['dtype'])

    loader = torch.utils.data.DataLoader(
        shapenetdb,
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        pin_memory=True,
        drop_last=True)
    test_loader = iter(loader)

    model =  SingleViewto3D(cfg)
    model.cuda()
    model.eval()

    pth_path = f'{cwd}/checkpoint/{args.id}/checkpoint_point.pth'
    if cfg['load_eval_checkpoint']:
        checkpoint = torch.load(pth_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    # for rendering
    device = "cuda:0"
    scene = Scene(device=device)
    scene.set_cam(1, 30, 90)
    scene.set_light(location=[[0.0, 1.0, 0.0]])

    # ============ preparing optimizer ... ============
    
    loss_list = []
    print("Starting testing !")
    
    cnt = 0
    for images_gt, ground_truth_3d, object_id in test_loader:

        cnt += 1

        # prediction
        images_gt, ground_truth_3d = images_gt.cuda(), ground_truth_3d.cuda()
        prediction_3d = model(images_gt, cfg)
        loss = calculate_loss(prediction_3d, ground_truth_3d, cfg).cpu().item()
        loss_vis = loss
        loss_list.append(loss_vis)
        print(f'cnt  : {cnt}, loss = {loss}')

        torch.save(prediction_3d.detach().cpu(), f'{out_dir}/pre_point_cloud_{cnt}.pt')

        # render
        print('rendering...')
        images_gt, images_pred = render_pred(scene,
                                            ground_truth_3d[0].detach(), 
                                            prediction_3d[0].detach(),
                                            cfg)

        plt.figure(figsize=(20, 10))
        plt.suptitle(f'Inference Result, loss = {loss}')
        plt.subplot(1, 2, 1)
        plt.imshow(images_gt)
        plt.axis("off")
        plt.title('Ground Truth')
        plt.subplot(1, 2, 2)
        plt.imshow(images_pred)
        plt.axis("off")
        plt.title('Prediction')
        plt.savefig(f'{out_dir}/render_{cnt}.png')

        if cnt == 10:
            break
    

    print('Done!')


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