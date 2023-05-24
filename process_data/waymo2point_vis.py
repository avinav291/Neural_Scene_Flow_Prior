import os
import tensorflow.compat.v1 as tf
import numpy as np
import multiprocessing
import argparse
tf.enable_eager_execution()
import cv2
import math
from typing import Optional, Tuple
from itertools import accumulate
from collections import namedtuple

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset import dataset_pb2
import matplotlib.cm
cmap = matplotlib.cm.get_cmap("viridis")

def make_colorwheel(transitions: tuple=(15, 6, 4, 11, 13, 6)) -> np.ndarray:
    """Creates a colorwheel (borrowed/modified from flowpy).
    A colorwheel defines the transitions between the six primary hues:
    Red(255, 0, 0), Yellow(255, 255, 0), Green(0, 255, 0), Cyan(0, 255, 255), Blue(0, 0, 255) and Magenta(255, 0, 255).
    Args:
        transitions: Contains the length of the six transitions, based on human color perception.
    Returns:
        colorwheel: The RGB values of the transitions in the color space.
    Notes:
        For more information, see:
        https://web.archive.org/web/20051107102013/http://members.shaw.ca/quadibloc/other/colint.htm
        http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    """
    colorwheel_length = sum(transitions)
    # The red hue is repeated to make the colorwheel cyclic
    base_hues = map(
        np.array, ([255, 0, 0], [255, 255, 0], [0, 255, 0], [0, 255, 255], [0, 0, 255], [255, 0, 255], [255, 0, 0])
    )
    colorwheel = np.zeros((colorwheel_length, 3), dtype="uint8")
    hue_from = next(base_hues)
    start_index = 0
    for hue_to, end_index in zip(base_hues, accumulate(transitions)):
        transition_length = end_index - start_index
        colorwheel[start_index:end_index] = np.linspace(hue_from, hue_to, transition_length, endpoint=False)
        hue_from = hue_to
        start_index = end_index
    return colorwheel


def flow_to_rgb(
    flow: np.ndarray,
    flow_max_radius: Optional[float]=None,
    background: Optional[str]="bright",
) -> np.ndarray:
    """Creates a RGB representation of an optical flow (borrowed/modified from flowpy).
    Args:
        flow: scene flow.
            flow[..., 0] should be the x-displacement
            flow[..., 1] should be the y-displacement
            flow[..., 2] should be the z-displacement
        flow_max_radius: Set the radius that gives the maximum color intensity, useful for comparing different flows.
            Default: The normalization is based on the input flow maximum radius.
        background: States if zero-valued flow should look 'bright' or 'dark'.
    Returns: An array of RGB colors.
    """
    valid_backgrounds = ("bright", "dark")
    if background not in valid_backgrounds:
        raise ValueError(f"background should be one the following: {valid_backgrounds}, not {background}.")
    wheel = make_colorwheel()
    # For scene flow, it's reasonable to assume displacements in x and y directions only for visualization pursposes.
    complex_flow = flow[..., 0] + 1j * flow[..., 1]
    radius, angle = np.abs(complex_flow), np.angle(complex_flow)
    if flow_max_radius is None:
        flow_max_radius = np.max(radius)
    if flow_max_radius > 0:
        radius /= flow_max_radius
    ncols = len(wheel)
    # Map the angles from (-pi, pi] to [0, 2pi) to [0, ncols - 1)
    angle[angle < 0] += 2 * np.pi
    angle = angle * ((ncols - 1) / (2 * np.pi))
    # Make the wheel cyclic for interpolation
    wheel = np.vstack((wheel, wheel[0]))
    # Interpolate the hues
    (angle_fractional, angle_floor), angle_ceil = np.modf(angle), np.ceil(angle)
    angle_fractional = angle_fractional.reshape((angle_fractional.shape) + (1,))
    float_hue = (
        wheel[angle_floor.astype(np.int)] * (1 - angle_fractional) + wheel[angle_ceil.astype(np.int)] * angle_fractional
    )
    ColorizationArgs = namedtuple(
        'ColorizationArgs', ['move_hue_valid_radius', 'move_hue_oversized_radius', 'invalid_color']
    )
    def move_hue_on_V_axis(hues, factors):
        return hues * np.expand_dims(factors, -1)
    def move_hue_on_S_axis(hues, factors):
        return 255. - np.expand_dims(factors, -1) * (255. - hues)
    if background == "dark":
        parameters = ColorizationArgs(
            move_hue_on_V_axis, move_hue_on_S_axis, np.array([255, 255, 255], dtype=np.float)
        )
    else:
        parameters = ColorizationArgs(move_hue_on_S_axis, move_hue_on_V_axis, np.array([0, 0, 0], dtype=np.float))
    colors = parameters.move_hue_valid_radius(float_hue, radius)
    oversized_radius_mask = radius > 1
    colors[oversized_radius_mask] = parameters.move_hue_oversized_radius(
        float_hue[oversized_radius_mask],
        1 / radius[oversized_radius_mask]
    )
    return colors.astype(np.uint8)



def display_laser_on_image(img, pcl, pcl_attr, pcl_flow, vehicle_to_image):
    # Convert the pointcloud to homogeneous coordinates.
    pcl1 = np.concatenate((pcl,np.ones_like(pcl[:,0:1])),axis=1)

    # Transform the point cloud to image space.
    proj_pcl = np.einsum('ij,bj->bi', vehicle_to_image, pcl1) 

    # Filter LIDAR points which are behind the camera.
    mask = proj_pcl[:,2] > 0
    proj_pcl = proj_pcl[mask]
    proj_pcl_attr = pcl_attr[mask]
    proj_pcl_flow = pcl_flow[mask]

    # Project the point cloud onto the image.
    proj_pcl = proj_pcl[:,:2]/proj_pcl[:,2:3]

    # Filter points which are outside the image.
    mask = np.logical_and(
        np.logical_and(proj_pcl[:,0] > 0, proj_pcl[:,0] < img.shape[1]),
        np.logical_and(proj_pcl[:,1] > 0, proj_pcl[:,1] < img.shape[1]))

    proj_pcl = proj_pcl[mask]
    proj_pcl_attr = proj_pcl_attr[mask]
    proj_pcl_flow = proj_pcl_flow[mask]

    # # Colour code the points based on distance.
    # coloured_intensity = 255*cmap(proj_pcl_attr[:,0]/30)

    # Colour code the points based on flow.
    coloured_intensity = flow_to_rgb(proj_pcl_flow, flow_max_radius=2.5, background='dark').tolist()

    # Draw a circle for each point.
    for i in range(proj_pcl.shape[0]):
        cv2.circle(img, (int(proj_pcl[i,0]),int(proj_pcl[i,1])), 1, coloured_intensity[i])

def get_camera_matrix(camera_calibration, frame):
    # extrinsic = np.array(camera_calibration.extrinsic.transform).reshape(4,4)
    intrinsic = camera_calibration.intrinsic
    extrinsic = np.array(frame.pose.transform).reshape((4,4))
    # extrinsic = extrinsic-frame_pose
    # Camera model:
    # | fx  0 cx 0 |
    # |  0 fy cy 0 |
    # |  0  0  1 0 |
    camera_model = np.array([
        [intrinsic[0], 0, intrinsic[2], 0],
        [0, intrinsic[1], intrinsic[3], 0],
        [0, 0,                       1, 0]])
    camera_mat = np.concatenate([extrinsic, camera_model], axis=0)
    return camera_mat

def get_image_transform(camera_calibration):
    """ For a given camera calibration, compute the transformation matrix
        from the vehicle reference frame to the image space.
    """

    # TODO: Handle the camera distortions
    extrinsic = np.array(camera_calibration.extrinsic.transform).reshape(4,4)
    intrinsic = camera_calibration.intrinsic

    # Camera model:
    # | fx  0 cx 0 |
    # |  0 fy cy 0 |
    # |  0  0  1 0 |
    camera_model = np.array([
        [intrinsic[0], 0, intrinsic[2], 0],
        [0, intrinsic[1], intrinsic[3], 0],
        [0, 0,                       1, 0]])

    # Swap the axes around
    axes_transformation = np.array([
        [0,-1,0,0],
        [0,0,-1,0],
        [1,0,0,0],
        [0,0,0,1]])

    # Compute the projection matrix from the vehicle space to image space.
    vehicle_to_image = np.matmul(camera_model, np.matmul(axes_transformation, np.linalg.inv(extrinsic)))
    return vehicle_to_image

def visualize(frame, range_images, frame_no, pcl, pcl_flow, out_path):
    # ri = np.array(range_images[1][0].data).reshape(64,2650,-1)
    ri = range_images
    # pcl = points[:,:3]
    mask = ri[:,:,0] > 0
    pcl_attr = ri[mask]

    camera_calibration = frame.context.camera_calibrations[0]

    # camera = utils.get(frame.images, camera_name)
    img = tf.image.decode_jpeg(frame.images[0].image)
    img = np.array(img)

    # Get the transformation matrix for the camera.
    vehicle_to_image = get_image_transform(camera_calibration)

    # BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the LIDAR points on the image.
    display_laser_on_image(img, pcl, pcl_attr, pcl_flow, vehicle_to_image)

    # Display the image
    cv2.imwrite(f"{out_path}/laser_{frame_no}.jpg", img)

# def filter_pc(frame,range_images,camera_projections,range_image_top_pose,range_images_flow):
def filter_pc(frame,proposal_mask,range_images,camera_projections,range_image_top_pose,range_images_flow):
    top_range = np.array(range_images[1][0].data).reshape(64,2650,-1)
    top_range[...,0] = top_range[...,0]*proposal_mask

    # convert flow
    range_image_flow = range_images_flow[1][0]
    range_image_flow_tensor = tf.reshape(tf.convert_to_tensor(value=range_image_flow.data), range_image_flow.shape.dims)

    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    c = calibrations[0]

    points = []
    cp_points = []
    cartesian_range_images = frame_utils.convert_range_image_to_cartesian(frame, range_images, range_image_top_pose, 0, False)
    range_image = range_images[c.name][0]
    range_image_tensor = tf.convert_to_tensor(top_range)
    # range_image_tensor = tf.convert_to_tensor(range_image)
    range_image_mask = range_image_tensor[..., 0] > 0
    range_image_cartesian = cartesian_range_images[c.name]

    # add flow
    result = tf.concat([range_image_cartesian,range_image_flow_tensor],axis=2)
    range_image_cartesian = result

    points_tensor = tf.gather_nd(range_image_cartesian,tf.compat.v1.where(range_image_mask))

    cp = camera_projections[c.name][0]
    cp_tensor = tf.reshape(tf.convert_to_tensor(value=cp.data), cp.shape.dims)
    cp_points_tensor = tf.gather_nd(cp_tensor, tf.compat.v1.where(range_image_mask))
    points.append(points_tensor.numpy())
    
    cp_points.append(cp_points_tensor.numpy())
    return cp_points[0],points[0], c, top_range

def transform_global(points_all,frame):
    new_pcs = np.concatenate((points_all,np.ones(points_all.shape[0])[:,np.newaxis]),axis=1)
    T_1 = np.array(frame.pose.transform).reshape((4,4))
    global_pc1 = T_1 @ new_pcs.T
    global_pc1 = global_pc1.T[:,:3]
    return global_pc1

def transform_frame(global_pc,frame):
    new_pcs = np.concatenate((global_pc,np.ones(global_pc.shape[0])[:,np.newaxis]),axis=1)
    T_1 = np.array(frame.pose.transform).reshape((4,4))
    points = np.matmul(new_pcs, np.linalg.inv(T_1).T)
    points = points[:,:3]
    return points

def transform(points,pose):
    new_pcs = np.concatenate((points,np.ones(points.shape[0])[:,np.newaxis]),axis=1)
    point_T = pose @ new_pcs.T
    point_T = point_T.T[:,:3]
    return point_T

def read_root(root): 
    file_list = sorted(os.listdir(root))
    # file_list = [k for k in file_list if 'segment-9' in k]
    return file_list

def extract_flow(frame): 
    range_images_flow = {}
    for laser in frame.lasers:
        if len(laser.ri_return1.range_image_flow_compressed) > 0:
            flow_tensor = tf.io.decode_compressed(frame.lasers[0].ri_return1.range_image_flow_compressed, 'ZLIB')
            flow_tensor2 = tf.io.decode_compressed(frame.lasers[0].ri_return2.range_image_flow_compressed, 'ZLIB')
            ri = dataset_pb2.MatrixFloat()
            ri.ParseFromString(bytearray(flow_tensor.numpy()))
            ri2 = dataset_pb2.MatrixFloat()
            ri2.ParseFromString(bytearray(flow_tensor2.numpy()))
        range_images_flow[laser.name] = [ri,ri2]
    return range_images_flow

def main(waymo_root,split,output_dir,token, process_num, debug, vis):
    train_root = os.path.join(waymo_root,split)
    file_list = read_root(train_root)
    if debug:
        file_list = file_list[0:5]

    for s in range(len(file_list)):
        if s % process_num != token:
            continue
        filename = file_list[s]
        FILENAME = os.path.join(train_root,filename)
        segment_dir = os.path.join(output_dir,filename.split('.')[0])
        if not os.path.exists(segment_dir): continue
        segmenr_pc_dir = os.path.join(segment_dir,'PC')
        os.makedirs(segmenr_pc_dir,exist_ok=True)
        if vis is True:
            segmenr_vis_dir = os.path.join(segment_dir,'PC_VIS_FLOW')
            os.makedirs(segmenr_vis_dir,exist_ok=True)

        proposal_root = os.path.join(output_dir,segment_dir,'proposal')
        proposals = sorted(os.listdir(proposal_root))

        dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
        frame_list = ["%06d" % (x) for x in range(199)]
        new_file = ['frame_'+t+'.npz' for t in frame_list]
        i = 0
        print('Processing:',filename)
        # pc_prev, ri_prev = None, None
        frame_prev = None
        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            range_images, camera_projections, range_image_top_pose = frame_utils.parse_range_image_and_camera_projection(frame)
            range_images_flow = extract_flow(frame)
            proposal_path = os.path.join(proposal_root,proposals[i])
            proposal = np.load(proposal_path)
            proposal_mask = np.where(proposal[:,:]!=0,1,0)
            cp_points_all,points_all, calib, range_images = filter_pc(frame,proposal_mask,range_images,camera_projections,range_image_top_pose,range_images_flow)
            
            point_xyz=points_all[:,:3]

            #To eliminate no flow frames
            if np.amax(points_all[:, 3:6])<0.2: continue
            # get global coordinates
            global_pc1 = transform_global(point_xyz,frame)

            name = os.path.join(segmenr_pc_dir,new_file[i])
            if vis is True:
                if frame_prev is not None:
                    pc_car2 = points_all[:, :3] - points_all[:, 3:6]*0.1
                    T_world_car1 = np.array(frame_prev.pose.transform).reshape((4,4))
                    T_world_car2 = np.array(frame.pose.transform).reshape((4,4))
                    T_car1_car2 = np.linalg.inv(T_world_car1) @ T_world_car2
                    pc_car1_car2 = transform(pc_car2, T_car1_car2)
                    visualize(frame_prev, range_images, i, pc_car1_car2, points_all[:, 3:6]*0.1, segmenr_vis_dir)
                frame_prev = frame

            # filter front camera image
            front_mask = np.where((cp_points_all[:,0]==1)|(cp_points_all[:,3]==1),1,0)
            global_pc1 = global_pc1[front_mask==1]
            points_all = points_all[front_mask==1]
            cp_points_1 = cp_points_all[front_mask==1]

            camera_matrix = get_camera_matrix(frame.context.camera_calibrations[0], frame)

            np.savez(name,global_pc1=global_pc1,cp_point =cp_points_1, point = points_all, camera=camera_matrix)

            i = i+1
            if i>198:
                break
        print('finish:',filename)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--waymo_root',default='/vast/work/public/ml-datasets/waymo_open_dataset_scene_flow',help='path to the waymo open dataset')
    parser.add_argument('--split',default='valid',choices=['train','valid'])
    parser.add_argument('--output_dir',default='/scratch/ag7644/waymo_sf_debug11/',help='path to save the data')
    parser.add_argument('--process', type=int, default=1, help = 'num workers to use')
    parser.add_argument('--debug',type=bool,default=False,help='test 5 segments for debug')
    parser.add_argument('--visualize',type=bool,default=False,help='Dump Lidar on image')
    
    args = parser.parse_args()

    if args.process>1:
        pool = multiprocessing.Pool(args.process)
        for token in range(args.process):
            result = pool.apply_async(main, args=(args.waymo_root,args.split,args.output_dir, token, args.process,args.debug, args.visualize))
        pool.close()
        pool.join()
    else:
        main(args.waymo_root,args.split,args.output_dir, 0, args.process, args.debug, args.visualize)
