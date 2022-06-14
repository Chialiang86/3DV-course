import argparse
import os
import time
import torch
import src.losses as losses
from tqdm import tqdm
from src.dataset import ShapeNetDB
from src.model import SingleViewto3D
from src.losses import ChamferDistanceLoss
import yaml
import matplotlib.pyplot as plt

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

# @hydra.main(config_path="configs/", config_name="config.yaml")
def train_model(args, cfg):

    # config working dir
    cwd = os.getcwd()

    # config args : pth dir id
    out_dir = f'{cwd}/checkpoint/{args.id}'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        print(f'{out_dir} created.')

    print(cfg['data_dir'])
    dtype = cfg['dtype']
    shapenetdb = ShapeNetDB(cfg['data_dir'], cfg['dtype'])

    loader = torch.utils.data.DataLoader(
        shapenetdb,
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        pin_memory=True,
        drop_last=True)

    model =  SingleViewto3D(cfg)
    model.cuda()
    model.train()

    # ============ preparing optimizer ... ============
    optimizer = torch.optim.Adam(model.parameters(), lr = cfg['lr'])  # to use with ViTs
    start_iter = 0
    start_time = time.time()

    if cfg['load_checkpoint']:
        checkpoint = torch.load(f'{out_dir}/checkpoint_{dtype}.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iter = checkpoint['step']
        print(f"Succesfully loaded iter {start_iter}")
    
    loss_list = []
    print("Starting training !")
    for step in range(start_iter, cfg['max_iter']):

        iter_start_time = time.time()

        train_loader = iter(loader)
        for images_gt, ground_truth_3d, object_id in tqdm(train_loader):

            images_gt, ground_truth_3d = images_gt.cuda(), ground_truth_3d.cuda()

            prediction_3d = model(images_gt, cfg)
            loss = calculate_loss(prediction_3d, ground_truth_3d, cfg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()        

            loss_vis = loss.cpu().item()
            loss_list.append(loss_vis)

        iter_time = time.time() - iter_start_time

        if (step % cfg['save_freq']) == 0:
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, f'{out_dir}/checkpoint_{dtype}.pth')
        print("[%4d/%4d]; ttime: %.0f ; loss: %.5f" % (step, cfg['max_iter'], iter_time, loss_vis))

    dtype = cfg['dtype']
    plt.title(f'Loss Curve (type = {dtype})')
    plt.plot(loss_list)
    plt.savefig(f'{out_dir}/loss_curve.png')

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
    
    train_model(args, cfg)