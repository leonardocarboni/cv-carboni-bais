import glm
import random
import numpy as np
import cv2 as cv
from time import time

block_size = 1
block_size2 = 115
n_frame = 0
visible_voxels = []
lookup_table = []

with np.load('./data/cam1/masks.npz') as file:
    mask1 = file['masks']
with np.load('./data/cam2/masks.npz') as file:
    mask2 = file['masks']
with np.load('./data/cam3/masks.npz') as file:
    mask3 = file['masks']
with np.load('./data/cam4/masks.npz') as file:
    mask4 = file['masks']


def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data = []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -
                        block_size, z*block_size - depth/2])
    return data


def set_voxel_positions(width, height, depth):
    global n_frame
    global visible_voxels
    global lookup_table
    start_lookup = time()
    # Generates random voxel locations
    # TODO: You need to calculate proper voxel arrays instead of random ones.
    # Reading the stored lookup table

    # COMMENT THIS AND UNCOMMENT ROW 53 IF YOU WANT TO BUILD LOOK UP TABLE THE FIRST FRAME RATHER THAN LOAD IT FROM FILE
    with np.load(f'./data/lookup_table3.npz') as file:
        lookup_table = file['lookup_table']
    voxel_positions = np.array(create_cube(750, 1500, 1500), dtype=np.float32)
    print(f"time to load/create lookup table: {time()-start_lookup}")
    
    # Last frame to be reconstructed
    if n_frame == 50:
        n_frame = 0 # reset frame count
        visible_voxels = [] # reset silhouette
        return visible_voxels

    # first frame encountered
    if n_frame == 0:
        # voxel_positions, lookup_table = create_lookup_table(750, 1500, 1500) # UNCOMMENT THIS 
        start_reconstruction = time()
        for vox in range(voxel_positions.shape[0]): # for each voxel id
            flag = True # the voxel is foreground for all cameras (flag)
            for i in range(4): # for each camera
                x_voxels, y_voxels, z_voxels, x, y = lookup_table[vox, :, i] # extract voxel 3D and 2D coordinates for that camera
                x = int(x)
                y = int(y)
                # check if the pixel is foreground for all cameras
                if i == 0 and mask1[0][y, x] == 0:
                    flag = False
                if i == 1 and mask2[0][y, x] == 0:
                    flag = False
                if i == 2 and mask3[0][y, x] == 0:
                    flag = False
                if i == 3 and mask4[0][y, x] == 0:
                    flag = False
            if flag: # if it is foreground for all cameras
                visible_voxels.append([x_voxels/75, -z_voxels/75, y_voxels/75]) # adapt to glm format, scale and add to reconstruction 
        print(f"time to reconstruct all: {time()-start_reconstruction}")
        n_frame += 1 # next frame
        return visible_voxels
    else: # a frame other than the first, we perform optimization by only looking at changed pixels
        
        # load last and current frames' masks for each camera
        current_mask1 = np.array(mask1[n_frame], dtype=np.int8)
        last_mask1 = np.array(mask1[n_frame-1], dtype=np.int8)
        current_mask2 = np.array(mask2[n_frame], dtype=np.int8)
        last_mask2 = np.array(mask2[n_frame-1], dtype=np.int8)
        current_mask3 = np.array(mask3[n_frame], dtype=np.int8)
        last_mask3 = np.array(mask3[n_frame-1], dtype=np.int8)
        current_mask4 = np.array(mask4[n_frame], dtype=np.int8)
        last_mask4 = np.array(mask4[n_frame-1], dtype=np.int8)

        # differences on the masks between the frames for each camera
        differences = [current_mask1 - last_mask1, current_mask2 -
                       last_mask2, current_mask3 - last_mask3, current_mask4 - last_mask4]
        
        start_reconstruction_opt = time()
        removed = 0
        added = 0
        for n_cam in range(len(differences)):
            d = differences[n_cam] # differences in mask N
            xs, ys = np.where((d == 1) | (d == -1)) # voxels that have changed
            coords = np.stack((xs, ys), axis=1)
            lookup = lookup_table[:, :, n_cam] # lookup table for the current camera
            for coord in coords: # for each 2D changed pixel
                x, y = coord[1], coord[0]
                all_pos = np.where((lookup[:, 3] == x) & (lookup[:, 4] == y))[0] # get all corresponding voxels ID
                for pos in all_pos: # for each voxel ID
                    # same as frame 1 but with the current mask
                    flag = True
                    for i in range(4):
                        x_voxels, y_voxels, z_voxels, x, y = lookup_table[pos, :, i]
                        x = int(x)
                        y = int(y)
                        vox = [x_voxels/75, -z_voxels/75, y_voxels/75]
                        if i == 0 and mask1[n_frame][y, x] == 0:
                            flag = False
                        if i == 1 and mask2[n_frame][y, x] == 0:
                            flag = False
                        if i == 2 and mask3[n_frame][y, x] == 0:
                            flag = False
                        if i == 3 and mask4[n_frame][y, x] == 0:
                            flag = False
                    if not flag and vox in visible_voxels: # if at least a camera said background
                        visible_voxels.remove(vox)
                        removed +=1
                        # print("Removed voxel: ", vox)
                    if flag: # if it's foreground
                        visible_voxels.append([x_voxels/75, -z_voxels/75, y_voxels/75])
                        # print("Appended voxel: ", [x_voxels/75, -z_voxels/75, y_voxels/75])
                        added += 1
        print(f"time reconstruct optimaly: {time()-start_reconstruction_opt}")
        print(f"voxels removed this frame: {removed}, voxels added this frame: {added}")
        print(len(visible_voxels))
        n_frame += 1
        return visible_voxels


def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room
    # TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.
    positions = []
    for camera_i in range(1, 5):
        s = cv.FileStorage(
            f"data/cam{camera_i}/config.xml", cv.FileStorage_READ)
        tvec_extr = s.getNode('tvec_extr').mat()
        R = s.getNode('R_MAT').mat()
        positions.append(np.dot(-R.T, tvec_extr))
        s.release()
    positions = np.stack(positions)
    # positions = normalize(positions.squeeze(), axis=0)
    positions = positions.squeeze() / 115
    # converting from opencv space to glm space
    # swap y and z
    positions[:, [1, 2]] = positions[:, [2, 1]]
    # abs y positions
    positions[:, 1] = np.abs(positions[:, 1])

    return positions


def get_cam_rotation_matrices():
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
    cam_rotations = []
    for camera_i in range(1, 5):
        s = cv.FileStorage(
            f"data/cam{camera_i}/config.xml", cv.FileStorage_READ)
        tvec_extr = s.getNode('tvec_extr').mat()
        R = s.getNode('R_MAT').mat()

        t1 = np.hstack((R, tvec_extr))
        t1 = np.vstack((t1, [0, 0, 0, 1]))
        # swap y and z
        t1[:, [1, 2]] = t1[:, [2, 1]]
        # invert x rotation of the camears
        t1[:, 1] = -t1[:, 1]
        # transform to mat4
        rot = glm.mat4(t1.T)
        # rotate cameras by 90 degrees because they point on the wrong side
        rot = glm.rotate(rot, -90 * np.pi / 180, [0, 1, 0])

        cam_rotations.append(rot)
    return cam_rotations


def create_cube(width, height, depth):
    "creates a solid with resolution 100x100x100"
    cube = []
    for x in np.arange(0, width, 8):
        for y in np.arange(-depth//2, depth//2, 15):
            for z in np.arange(-height, height, 30):
                cube.append([x, y, z])
    return cube


def create_lookup_table(width, height, depth):
    "Returns the lookup table mapping each voxels to its correspongin 3D and 2D coordinates for each camera"
    "Inputs are for the placement of the solid for the reconstruction"
    # create solid
    voxel_positions = np.array(create_cube(
        width, height, depth), dtype=np.float32) 
    lookup_table = np.zeros((voxel_positions.shape[0], 5, 4)) # initialize look up table
    for camera_i in range(1, 5): # for each camera
        # load parameters
        s = cv.FileStorage(
            f"data/cam{camera_i}/config.xml", cv.FileStorage_READ)
        camera_matrix = s.getNode('camera_matrix').mat()
        dist_coeffs = s.getNode('dist_coeffs').mat()
        tvec_extr = s.getNode('tvec_extr').mat()
        rvec_extr = s.getNode('rvec_extr').mat()
        s.release()
        # for each 3D point of the voxel solid
        for i, pos in enumerate(voxel_positions):
            # project the point in the 2D image plane for this camera
            imgpoint, _ = cv.projectPoints(
                pos, rvec_extr, tvec_extr, camera_matrix, dist_coeffs)
            imgpoint = imgpoint.ravel()
            x = imgpoint[0]
            y = imgpoint[1]
            if x >= 0 and x < 644 and y >= 0 and y < 486: # if the 2D pixel is in range of the frame
                # store 2D and 3D coordinates
                lookup_table[i, :, camera_i -
                             1] = [int(pos[0]), int(pos[1]), int(pos[2]), int(x), int(y)] 

    np.savez('data/lookup_table3', lookup_table=lookup_table) # comment in the actual execution
    return voxel_positions, lookup_table

# create_lookup_table(750, 1500, 1500)
# print(set_voxel_positions(1,2,3))
