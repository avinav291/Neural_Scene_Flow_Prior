import numpy as np
import os
import argparse
import multiprocessing
import open3d as o3d

def transform_frame(global_pc,pose):
    new_pcs = np.concatenate((global_pc,np.ones(global_pc.shape[0])[:,np.newaxis]),axis=1)
    T_1 = pose
    points = np.matmul(new_pcs, np.linalg.inv(T_1).T)
    points = points[:,:3]
    return points

def transform_world(points_all,pose):
    new_pcs = np.concatenate((points_all,np.ones(points_all.shape[0])[:,np.newaxis]),axis=1)
    T_1 = pose
    global_pc1 = T_1 @ new_pcs.T
    global_pc1 = global_pc1.T[:,:3]
    return global_pc1

def transform(points,pose):
    new_pcs = np.concatenate((points,np.ones(points.shape[0])[:,np.newaxis]),axis=1)
    point_T = pose @ new_pcs.T
    point_T = point_T.T[:,:3]
    return point_T

def main(waymo_root, token, process_num):
    segs = sorted(os.listdir(waymo_root))

    for idx, seg in enumerate(segs):
        if idx % process_num != token:
            continue
        path = os.path.join(waymo_root,seg,'PC')
        files = sorted(os.listdir(path))
        output_dir = os.path.join(waymo_root,seg,'point')
        os.makedirs(output_dir,exist_ok=True)
        print('Processing:',seg)
        for i in range(5,len(files)-1):
            data_pc1 = np.load(os.path.join(path,files[i]))
            data_pc2 = np.load(os.path.join(path,files[i+1]))

            p1_car1 = data_pc1['point'][:,:3]
            p2_car2 = data_pc2['point'][:,:3]

            flow_car1 = data_pc1['point'][:,3:6]
            flow_car2 = data_pc2['point'][:,3:6]
            car1_camera_matrix = data_pc1['camera']
            T_world_car1 = car1_camera_matrix[:4]
            car2_camera_matrix = data_pc2['camera']
            T_world_car2 = car2_camera_matrix[:4]
            T_car2_car1 = np.linalg.inv(T_world_car2) @ T_world_car1
            p1_car2 = transform(p1_car1, T_car2_car1)

            if np.amax(flow_car2)<0.2 or abs(np.amin(flow_car2))<0.2: continue
            flow_car2 = -flow_car2*0.1

            name = 'frame_'+ '%06d' % i + '.npz'
            np.savez(os.path.join(output_dir,name),p1=p2_car2,p2=p1_car2,flow=flow_car2, transform = T_world_car2)
        print('finish:',seg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--waymo_root',default='/scratch/ag7644/waymo_sf/valid/',help='path to the waymo open dataset')
    parser.add_argument('--waymo_root',default='/scratch/ag7644/waymo_sf_debug11/',help='path to the waymo open dataset')
    parser.add_argument('--process', type=int, default=1, help = 'num workers to use')
    args = parser.parse_args()

    if args.process>1:
        pool = multiprocessing.Pool(args.process)
        for token in range(args.process):
            result = pool.apply_async(main, args=(args.waymo_root, token, args.process))
        pool.close()
        pool.join()
    else:
        main(args.waymo_root, 0, args.process)

    # main(args.waymo_root)