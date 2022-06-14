import glob
import os
import numpy as np

def main():
    data_root = 'data/chair_img_pc_voxel_mesh'
    data_dirs = glob.glob(f'{data_root}/*')

    n = len(data_dirs)
    ratio = 0.9
    all_inds = np.random.randint(n, size=n)
    train_inds = all_inds[:int(ratio * n)]
    val_inds = all_inds[int(ratio * n):]

    train_path = 'data/chair_img_pc_voxel_mesh/train_data.txt'
    with open(train_path, 'w') as f:
        for ind in train_inds:
            f.write(f'{data_dirs[ind]}\n')
    val_path = 'data/chair_img_pc_voxel_mesh/val_data.txt'
    with open(val_path, 'w') as f:
        for ind in val_inds:
            f.write(f'{data_dirs[ind]}\n')

if __name__=="__main__":
    main()