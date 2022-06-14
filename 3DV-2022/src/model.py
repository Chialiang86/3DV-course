from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch
import torch.nn.functional as F
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Pointclouds, Meshes, Volumes
from pytorch3d.utils import ico_sphere

class SingleViewto3D(nn.Module):
    def __init__(self, cfg):
        super(SingleViewto3D, self).__init__()
        self.device = "cuda"
        vision_model = torchvision_models.__dict__[cfg['arch']](pretrained=True)
        self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

        # define decoder
        if cfg['dtype'] == "voxel":
            self.decoder =  VolumeDecoder(cfg['voxel_length'], cfg['voxel_width'], cfg['voxel_height'], 512)           
        elif cfg['dtype'] == "point":
            self.n_point = cfg['n_points']
            self.decoder = PointDecoder(cfg['n_points'], 512)
        elif cfg['dtype'] == "mesh":
            self.decoder = MeshVertsDecoder(cfg['n_verts'], 512)
            
            # make white texturre
            mesh_pred = ico_sphere(4,'cuda')
            
            self.verts = mesh_pred.verts_list()[0].cuda()
            self.faces = mesh_pred.faces_list()[0].cuda()
            
            verts_rgb = torch.ones_like(self.verts)[None]  # (1, V, 3)
            self.textures = TexturesVertex(verts_features=verts_rgb)

            # self.mesh_pred = Meshes(
            #     mesh_pred.verts_list()*cfg['batch_size'], 
            #     mesh_pred.faces_list()*cfg['batch_size'])
            # We will learn to deform the source mesh by offsetting its vertices
            # The shape of the deform parameters is equal to the total number of vertices in src_mesh
            # self.deform_verts = torch.full(self.mesh_pred.verts_packed().shape, 0.0, device=self.device, requires_grad=True)

    def forward(self, images, cfg):
        # results = dict()

        # total_loss = 0.0
        # start_time = time.time()
        # B = images.shape[0]

        images_normalize = self.normalize(images.permute(0,3,1,2))
        encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1)

        # call decoder
        if cfg['dtype'] == "voxel":
            voxels_pred = self.decoder(encoded_feat)
            return voxels_pred

        elif cfg['dtype'] == "point":
            # TODO:
            pointclouds_pred = self.decoder(encoded_feat)
            return pointclouds_pred

        elif cfg['dtype'] == "mesh":
            # deform_vertices_pred = None
            verts_pred = self.decoder(encoded_feat)
            return  verts_pred         

# class PointDecoder(nn.Module):
#     def __init__(self, num_points, latent_size):
#         super(PointDecoder, self).__init__()
#         self.num_points = num_points
#         self.fc0 = nn.Linear(latent_size, 100)
#         self.fc1 = nn.Linear(100, 128)
#         self.fc2 = nn.Linear(128, 256)
#         self.fc3 = nn.Linear(256, 512)
#         self.fc4 = nn.Linear(512, 1024)
#         self.fc5 = nn.Linear(1024, self.num_points * 3)
#         self.th = nn.Tanh()

#     def forward(self, x):
#         batchsize = x.size()[0]
#         x = F.relu(self.fc0(x))
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = F.relu(self.fc4(x))
#         x = self.th(self.fc5(x))
#         x = x.view(batchsize, self.num_points, 3)
#         return x

class PointDecoder(nn.Module):
    def __init__(self, num_points, latent_size):
        super(PointDecoder, self).__init__()
        self.num_points = num_points
        self.fc0 = nn.Linear(latent_size, 100)
        self.fc1 = nn.Linear(100, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 1024)
        self.fc5 = nn.Linear(1024, self.num_points * 3)
        self.th = nn.Tanh()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.th(self.fc5(x))
        x = x.view(batchsize, self.num_points, 3)
        return x

class VolumeDecoder(nn.Module):
    def __init__(self, length, width, height, latent_size):
        super(VolumeDecoder, self).__init__()
        self.length = length
        self.width = width
        self.height = height
        self.fc0 = nn.Linear(latent_size, 1024)
        self.fc1 = nn.Linear(1024, 2048)
        self.fc2 = nn.Linear(2048, 4096)
        self.fc3 = nn.Linear(4096, 8192)
        self.fc4 = nn.Linear(8192, self.length * self.width * self.height)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        x = x.view(batchsize, self.length, self.width, self.height)
        return x

class MeshVertsDecoder(nn.Module):
    def __init__(self, num_verts, latent_size):
        super(MeshVertsDecoder, self).__init__()
        self.num_verts = num_verts
        self.fc0 = nn.Linear(latent_size, 1024)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, self.num_verts * 3)
        self.th = nn.Tanh()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.th(self.fc3(x))
        x = x.view(batchsize, self.num_verts, 3)
        return x