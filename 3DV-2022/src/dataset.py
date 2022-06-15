import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from torchvision.io import read_image
from PIL import Image

import os
import glob
import numpy as np

from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from pytorch3d.renderer import TexturesVertex

class ShapeNetDB(Dataset):
    def __init__(self, data_dir, data_txt, data_type, img_transform=False):
        super(ShapeNetDB).__init__()
        self.data_dir = data_dir
        self.data_txt = data_txt
        self.data_type = data_type
        self.db = self.__load_db()
        self.img_transform = img_transform

        self.__get_index()


    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        if self.data_type == 'point':
            """
            Return shapes:
            img: (B, 256, 256, 3)
            pc: (B, 2048, 3)
            object_id: (B,)
            """
            img, img_id = self.load_img(idx)
            pc, object_id = self.load_point(idx)

            assert img_id == object_id

            return img, pc, object_id
        
        elif self.data_type == 'voxel':
            """
            Return shapes:
            img: (B, 256, 256, 3)
            voxel: (B, 33, 33, 33)
            object_id: (B,)
            """
            img, img_id = self.load_img(idx)
            voxel, object_id = self.load_voxel(idx)

            assert img_id == object_id

            return img, voxel, object_id

        elif self.data_type == 'mesh':
            img, img_id = self.load_img(idx)
            verts, faces_idx, object_id = self.load_mesh(idx)

            model = {
                'img': img,
                'verts': verts,
                'faces': faces_idx,
                'object_id': object_id
            }

            assert img_id == object_id

            return model

    def __load_db(self):
        # ex : data/chair_img_pc_voxel_mesh/train_data.txt
        f = open(os.path.join(self.data_dir, self.data_txt), 'r')
        f_dirs = f.readlines()
        db_list = []
        for f_dir in f_dirs:
            db_list.append(f_dir.split('\n')[0])

        return db_list
    
    def __get_index(self):
        # ex : 1a6f615e8b1b5ae4dbbc9440457e303e
        self.id_index = self.data_dir.split('/').index("data") + 2

    def load_img(self, idx):
        path = os.path.join(self.db[idx], 'view.png')
        img = read_image(path) / 255.0
        img = img.permute(1,2,0)

        object_id = self.db[idx].split('/')[self.id_index]

        return img, object_id
    
    def load_mesh(self, idx):
        path = os.path.join(self.db[idx], 'model.obj')
        verts, faces, _ = load_obj(path, load_textures=False)
        faces_idx = faces.verts_idx

        # normalize
        center = verts.mean(0)
        verts = verts - center
        scale = max(verts.abs().max(0)[0])
        verts = verts / scale

        object_id = self.db[idx].split('/')[self.id_index]

        return verts, faces_idx, object_id

    def load_point(self, idx):
        path = os.path.join(self.db[idx], 'point_cloud.npy')
        points = np.load(path)

        # resample
        # n_points = 2048
        # choice = np.random.choice(points.shape[0], n_points, replace=True)
        # points = points[choice, :3]

        # normalize
        points = points - np.expand_dims(np.mean(points, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(points ** 2, axis = 1)),0)
        points = points / dist #scale

        object_id = self.db[idx].split('/')[self.id_index]

        return torch.from_numpy(points), object_id
    
    def load_voxel(self, idx):
        path = os.path.join(self.db[idx], 'voxel.npy')
        voxel = np.load(path).astype(float)

        object_id = self.db[idx].split('/')[self.id_index]

        return torch.from_numpy(voxel), object_id


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    # from pytorch3d.datasets import collate_batched_meshes

    db = ShapeNetDB('/home/chialiang86/Desktop/3DCV/hw/3DV-2022/data/chair_img_pc_voxel_mesh', 'point')
    dataloader = DataLoader(db, batch_size=10, shuffle=True)

    for img, point, object_id in dataloader:
        print(img.shape)
        print(point.shape)   
        print(object_id)
        break
