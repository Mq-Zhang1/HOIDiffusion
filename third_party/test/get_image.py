import sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import os
import argparse
import csv
import mano
import random
from psbody.mesh import MeshViewers, Mesh
from grabnet.tools.meshviewer import Mesh as M
from grabnet.tools.vis_tools import points_to_spheres
from grabnet.tools.utils import euler
from grabnet.tools.cfg_parser import Config
from grabnet.tests.tester import Tester

from bps_torch.bps import bps_torch

from psbody.mesh.colors import name_to_rgb
from grabnet.tools.train_tools import point2point_signed
from grabnet.tools.utils import aa2rotmat
from grabnet.tools.utils import makepath
from grabnet.tools.utils import to_cpu
from grabnet.tools.utils import showHandJoints, project_3D_points
import open3d
from PIL import Image
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
import shutil
import copy


jointsMapManoToSimple = [0,
                         13, 14, 15, 16,
                         1, 2, 3, 17,
                         4, 5, 6, 18,
                         10, 11, 12, 19,
                         7, 8, 9, 20]

def sample_points_in_ball(radius, num_points):
    theta = 2 * np.pi * np.random.rand(num_points)  # Azimuthal angle
    phi = np.arccos(2 * np.random.rand(num_points) - 1)  # Polar angle

    # Convert spherical to Cartesian coordinates
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    points = np.stack((x, y, z), axis=-1)
    return points

def getObjPath(obj_path):
    if not isinstance(obj_path, list):
        obj_path = [obj_path]
    input_obj_path = []
    for path in obj_path:
        all_objs = os.listdir(path)
        for obj in all_objs:
            '''===This is used to obtain object model==='''
            obj_model_path = os.path.join(path, obj, "align")
            if not os.path.exists(obj_model_path): continue
            obj_files = os.listdir(obj_model_path)
            for file in obj_files:
                if file[-3:] == "obj" or file[-3:] == "ply":
                    input_obj_path.append(os.path.join(obj_model_path, file))
    return input_obj_path

def getObjCat(obj_name:str):
    if "contactpose_" in obj_name:
        return obj_name.split("contactpose_")[-1]
    elif "knife" in obj_name:
        return "knife"
    elif "dispenser" in obj_name:
        return "dispenser"
    elif "bottle" in obj_name:
        return "bottle"
    elif "mug" in obj_name:
        return "mug"
    elif "binoculars" in obj_name:
        return "binoculars"
    elif "bowl" in obj_name:
        return "bowl"
    elif "cameras" in obj_name:
        return "camera"
    elif "can" in obj_name:
        return "can"
    elif "eyeglasses" in obj_name:
        return "eyeglasses"
    elif "flashlight" in obj_name:
        return "flashlight"
    elif "frying_pan" in obj_name:
        return "frying pan"
    elif "gamecontroller" in obj_name:
        return "game controller"
    elif "hammers" in obj_name:
        return "hammer"
    elif "headphones" in obj_name:
        return "headphone"
    elif "lightbulb" in obj_name:
        return "lightbulb"
    elif "lotion_pump" in obj_name:
        return "lotion pump"
    elif "marker_gluestick" in obj_name:
        return "marker gluestick"
    elif "marker_pen1" in obj_name:
        return "marker pen"
    elif "phone" in obj_name:
        return "phone"
    elif "screwdriver" in obj_name:
        return "screwdriver"
    elif "teapot" in obj_name:
        return "teapot"
    elif "toothbrush" in obj_name:
        return "toothbrush"
    elif "wineglass" in obj_name:
        return "wineglass"
    elif "trigger_sprayer" in obj_name:
        return "trigger sprayer"
    elif "_" in obj_name:
        return obj_name.replace("_", " ")
    else: return obj_name

def vis_results(dorig, coarse_net, refine_net, rh_model , rh_model_rot, it, save=False, save_dir = None, obj_name='random', obj_path = ''):
    data = []
    view_mat = np.diag(np.array([1,-1,-1,1]))
    with torch.no_grad():
        imw, imh = 512, 512
        cols = len(dorig['bps_object'])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        vis = open3d.visualization.Visualizer()
        vis.create_window(width=imw, height=imh, left=0, top=0, visible = False)

        drec_cnet = coarse_net.sample_poses(dorig['bps_object'])
        verts_rh_gen_cnet = rh_model(**drec_cnet).vertices

        _, h2o, _ = point2point_signed(verts_rh_gen_cnet, dorig['verts_object'].to(device))

        drec_cnet['trans_rhand_f'] = drec_cnet['transl']
        drec_cnet['global_orient_rhand_rotmat_f'] = aa2rotmat(drec_cnet['global_orient']).view(-1, 3, 3)
        drec_cnet['fpose_rhand_rotmat_f'] = aa2rotmat(drec_cnet['hand_pose']).view(-1, 15, 3, 3)
        drec_cnet['verts_object'] = dorig['verts_object'].to(device)
        drec_cnet['h2o_dist']= h2o.abs()

        drec_rnet = refine_net(**drec_cnet)
        verts_rh_gen_rnet = rh_model(**drec_rnet).vertices
        obj_cat = getObjCat(obj_name=obj_name)
        sentence = f"A hand is grasping a {obj_cat}"
        for cId in tqdm(range(0, len(dorig['bps_object']))):
            try:
                from copy import deepcopy
                meshes = deepcopy(dorig['mesh_object'])
                obj_mesh = meshes[cId]
            except:
                obj_mesh = points_to_spheres(to_cpu(dorig['verts_object'][cId]), radius=0.002, vc=name_to_rgb['green'])
            obj_numVert = np.asarray(obj_mesh.vertices).shape[0]
            # NOTE: Start interpolation
            # Read end pose:
            ori_point = torch.tensor([0,0,0]).float().cuda()
            end_fullpose = torch.cat([drec_rnet['global_orient'][cId],drec_rnet['hand_pose'][cId]])
            end_translate = drec_rnet['transl'][cId] - ori_point
            angle_axis = end_translate - ori_point
            norm_axis = angle_axis/torch.linalg.vector_norm(angle_axis)
            global_trans = np.array([0,0,0]) #dorig['trans'][cId] #

            # Add initial pose
            init_fullpose = end_fullpose.clone()
            init_fullpose[3:] = 0
            init_trans = (ori_point+0.3*norm_axis)
            # Add slerp transformation  
            n_inter = 10
            key_rots = []
            slerp = []
            key_times = [0, 1]
            for k in range(16):
                key_rots.append(Rotation.from_rotvec(np.array([init_fullpose[3*k:3*(k+1)].detach().cpu().numpy(), end_fullpose[3*k:3*(k+1)].detach().cpu().numpy()])))
            for k in range(16):
                slerp.append(Slerp(key_times, key_rots[k]))
            all_frames = []
            start_point=4
            for k in range(start_point,n_inter+2):
                if k == 0:
                    pose = init_fullpose
                    tr = init_trans
                elif k == n_inter + 1:
                    pose = end_fullpose
                    tr = end_translate
                else:
                    tr = init_trans +  (k/(n_inter+1)) * (end_translate - init_trans) #/(torch.linalg.vector_norm(trans-init_trans))
                    rot = []
                    for m in range(16):
                        times = [k/(n_inter+1)]
                        interp_rots = slerp[m](times)
                        rot.append(interp_rots.as_rotvec())
                    pose = torch.tensor(np.array(rot).reshape(-1)).float().to(device)
                pose = pose.unsqueeze(0)
                tr = tr.unsqueeze(0)
                rh_model_output = rh_model_rot(global_orient=pose[:, 0:3], hand_pose=pose[:, 3:], transl=tr, return_tips = True)
                verts = rh_model_output.vertices[0]
                joints = rh_model_output.joints[0]

                hand_mesh = open3d.geometry.TriangleMesh()
                hand_mesh.vertices = open3d.utility.Vector3dVector(np.copy(verts.cpu().numpy()))
                numVert = verts.shape[0]
                hand_mesh.triangles = open3d.utility.Vector3iVector(np.copy(rh_model.faces.reshape((-1, 3))))
                hand_mesh.vertex_colors = open3d.utility.Vector3dVector(np.tile(np.array([[0.5, 0.5, 0.5]]), [numVert, 1]))
                obj_mesh_k = copy.deepcopy(obj_mesh)
                global_trans = dorig["shift"][cId]
                hand_mesh = hand_mesh.translate(global_trans)
                obj_mesh_k = obj_mesh_k.translate(global_trans)
                # Start rendering
                if k == start_point:
                    vis.add_geometry(hand_mesh)
                    vis.add_geometry(obj_mesh_k)
                    camera_param = vis.get_view_control().convert_to_pinhole_camera_parameters()
                    # global_trans = dorig["shift"][cId] #np.copy(camera_param.extrinsic[:3,-1])
                    # global_trans *= np.array([1,-1,-1])
                    camera_param.extrinsic = view_mat
                    ctr = vis.get_view_control()
                    ctr.convert_from_pinhole_camera_parameters(camera_param)
                else:
                    vis.add_geometry(hand_mesh, reset_bounding_box=False)
                    vis.add_geometry(obj_mesh_k, reset_bounding_box=False)
                
            
                # Mask rendering
                mask = vis.capture_screen_float_buffer(do_render=True)
                mask = (np.asarray(mask)*255).astype(np.uint8)

                # Segmentation rendering
                hand_mesh.vertex_colors = open3d.utility.Vector3dVector(np.tile(np.array([[0, 0, 0]]), [numVert, 1]))
                obj_mesh_k.vertex_colors = open3d.utility.Vector3dVector(np.tile(np.array([[0, 0, 0]]), [obj_numVert, 1]))
                vis.update_geometry(hand_mesh)
                vis.update_geometry(obj_mesh_k)
                seg = vis.capture_screen_float_buffer(do_render=True)
                seg = np.asarray(seg)
                seg = seg.sum(axis=-1)
                index = np.where(seg<1.5)
                top = int(max(index[0].min()-50, 0))
                bottom = int(min(index[0].max()+50, 511))
                left = int(max(index[1].min()-50, 0))
                right = int(min(index[1].max()+50, 511))
                seg_image = np.where(seg<1.5, 1, 0)
                mask = mask * seg_image[:,:,None]
                seg_image = (np.asarray(seg_image)*255).astype(np.uint8)

                # object Segmentation rendering
                hand_mesh.vertex_colors = open3d.utility.Vector3dVector(np.tile(np.array([[1, 1, 1]]), [numVert, 1]))
                vis.update_geometry(hand_mesh)
                seg2 = vis.capture_screen_float_buffer(do_render=True)
                seg2 = np.asarray(seg2)
                seg2 = seg2.sum(axis=-1)
                seg_image2 = np.where(seg2<1.5, 1, 0)
                seg_image2 = (np.asarray(seg_image2)*255).astype(np.uint8)

                img_mask_path = os.path.join(mask_path, f"{obj_name}_{it}_{cId}_{k}.jpg")
                mask = Image.fromarray(mask.astype(np.uint8))
                mask.save(img_mask_path)
                img_seg_path = os.path.join(seg_path, f"{obj_name}_{it}_{cId}_{k}.jpg")
                seg_image = Image.fromarray(seg_image)
                seg_image.save(img_seg_path)

                img_seg_path2 = os.path.join(seg_path2, f"{obj_name}_{it}_{cId}_{k}.jpg")
                seg_image2 = Image.fromarray(seg_image2)
                seg_image2.save(img_seg_path2)

                # RGB rendering
                hand_mesh.vertex_colors = open3d.utility.Vector3dVector(np.tile(np.array([[0.5, 0.5, 0.5]]), [numVert, 1]))
                obj_mesh_k.vertex_colors = open3d.utility.Vector3dVector(np.tile(np.array([[0.6, 0.2, 0.2]]), [obj_numVert, 1]))
                hand_mesh.compute_triangle_normals()
                hand_mesh.compute_vertex_normals()
                obj_mesh_k.compute_triangle_normals()
                obj_mesh_k.compute_vertex_normals()
                vis.update_geometry(hand_mesh)
                vis.update_geometry(obj_mesh_k)
                img_rgb_path = os.path.join(rgb_path, f"{obj_name}_{it}_{cId}_{k}.jpg")
                render = vis.capture_screen_float_buffer(do_render=True)
                render = (np.asarray(render)*255).astype(np.uint8)
                all_frames.append(render)

                # Depth rendering
                img_depth_path = os.path.join(depth_path, f"{obj_name}_{it}_{cId}_{k}.png")
                vis.capture_depth_image(filename=img_depth_path, do_render=True)

                # Skeleton rendering
                camera_param = vis.get_view_control().convert_to_pinhole_camera_parameters()
                cam_mat = camera_param.intrinsic.intrinsic_matrix
                cam_extrinsics = camera_param.extrinsic
                handKps = project_3D_points(cam_mat, cam_extrinsics, joints.unsqueeze(0), global_trans, use_ext=False)
                imgAnno = showHandJoints(render, handKps[jointsMapManoToSimple])
                imgAnno = Image.fromarray(imgAnno)
                im_skeleton_path = os.path.join(skeleton_path, f"{obj_name}_{it}_{cId}_{k}.jpg")
                imgAnno.save(im_skeleton_path)

                render = Image.fromarray(render)
                render.save(img_rgb_path)
                vis.remove_geometry(hand_mesh, reset_bounding_box=False)
                vis.remove_geometry(obj_mesh_k, reset_bounding_box=False)

                # Lable saving
                im_label_path = os.path.join(label_path, f"{obj_name}_{it}_{cId}_{k}.npz")
                np.savez(im_label_path, joint3d = joints.cpu().numpy(), handparam = pose.cpu().numpy(), trans = tr.cpu().numpy(), obj_pose = dorig['rotmat'][cId], obj_trans = global_trans, intrinsics = cam_mat, extrinsics=cam_extrinsics)

                tmp = np.asarray(obj_mesh_k.vertices)
                print("====after center====",global_trans-tmp.mean(axis=0))
                entry = [img_rgb_path, im_skeleton_path, top, bottom, left, right, sentence, img_seg_path, img_mask_path, img_depth_path, im_label_path, obj_path]
                data.append(entry)
   
    return data



def grab_new_objs(grabnet, objs_path, rot=True, n_samples=10, it=1, scale=1.):
    grabnet.coarse_net.eval()
    grabnet.refine_net.eval()

    rh_model = mano.load(model_path=grabnet.cfg.rhm_path,
                         model_type='mano',
                         num_pca_comps=45,
                         batch_size=n_samples,
                         flat_hand_mean=True).to(grabnet.device)
    rh_model_rot = mano.load(model_path=grabnet.cfg.rhm_path,
                         model_type='mano',
                         num_pca_comps=45,
                         batch_size=1,
                         flat_hand_mean=True).to(grabnet.device)

    grabnet.refine_net.rhm_train = rh_model

    grabnet.logger(f'################# \n'
                   f'Colors Guide:'
                   f'                   \n'
                   f'Gray  --->  GrabNet generated grasp\n')

    bps = bps_torch(custom_basis = grabnet.bps)
    path_name = objs_path.split("/")[-3]
    if not isinstance(objs_path, list):
        objs_path = [objs_path]

    for new_obj in objs_path:
        radius = 0.05
        points = sample_points_in_ball(radius, n_samples)
        obj_poses = []
        for i in range(n_samples):
            # Here to control object rotation matrix
            camera_distance = random.uniform(0.35, 0.45)
            camera_elevation = random.uniform(5, 60)
            camera_azimuth = random.uniform(0, 0.1)
            # Calculate camera position in spherical coordinates
            camera_x = camera_distance * np.cos(np.radians(camera_elevation)) * np.cos(np.radians(camera_azimuth))
            camera_y = camera_distance * np.cos(np.radians(camera_elevation)) * np.sin(np.radians(camera_azimuth))
            camera_z = camera_distance * np.sin(np.radians(camera_elevation))   
            # Calculate camera orientation to look at the center of the object
            camera_pose = np.eye(4)
            camera_pose[:3, 3] = [camera_x, camera_y, camera_z]
            look_at = points[i]  # Center of the object
            camera_direction = look_at - camera_pose[:3, 3]
            camera_direction /= np.linalg.norm(camera_direction)
            camera_up = np.array([0.0, 0.0, 1.0])
            camera_right = np.cross(camera_direction, camera_up)
            camera_right /= np.linalg.norm(camera_right)
            camera_up = np.cross(camera_right, camera_direction)
            camera_pose[:3, :3] = np.column_stack((camera_right, camera_up, -camera_direction))
            obj_pose = np.linalg.inv(camera_pose)
            obj_poses.append(obj_pose)

        dorig = {'bps_object': [],
                 'verts_object': [],
                 'mesh_object': [],
                 'rotmat':[],
                 'shift':[],
                 }

        for samples in range(n_samples):
            obj_pose_matrix = obj_poses[samples]
            rand_rotmat = obj_pose_matrix[:3,:3]
            rand_shift = obj_pose_matrix[:3,-1]

            verts_obj, mesh_obj, rotmat = load_obj_verts(new_obj, rand_rotmat, rndrotate=rot, scale=scale)
            
            bps_object = bps.encode(torch.from_numpy(verts_obj), feature_type='dists')['dists']

            dorig['bps_object'].append(bps_object.to(grabnet.device))
            dorig['verts_object'].append(torch.from_numpy(verts_obj.astype(np.float32)).unsqueeze(0))
            dorig['mesh_object'].append(mesh_obj)
            dorig['rotmat'].append(rotmat)
            dorig['shift'].append(rand_shift)
            # dorig['trans'].append(rand_trans[samples])
            obj_name = os.path.basename(new_obj)

        dorig['bps_object'] = torch.cat(dorig['bps_object'])
        dorig['verts_object'] = torch.cat(dorig['verts_object'])

        save_dir = os.path.join(grabnet.cfg.work_dir, 'grab_new_objects')
        grabnet.logger(f'#################\n'
                              f'                   \n'
                              f'Showing results for the {obj_name.upper()}'
                              f'                      \n')

        return vis_results(dorig=dorig,
                    coarse_net=grabnet.coarse_net,
                    refine_net=grabnet.refine_net,
                    rh_model=rh_model,
                    rh_model_rot = rh_model_rot,
                    save=False,
                    save_dir=save_dir,
                    obj_name = path_name,
                    it = it,
                    obj_path = new_obj,
                    )

def load_obj_verts(mesh_path, rand_rotmat, rndrotate=True, scale=1., n_sample_verts=10000): #10000

    np.random.seed(100)
    obj_mesh = Mesh(filename=mesh_path, vscale=scale)

    obj_mesh.reset_normals()
    obj_mesh.vc = obj_mesh.colors_like('green')

    ## center and scale the object
    max_length = np.linalg.norm(obj_mesh.v, axis=1).max()
    if  max_length > .3:
        re_scale = max_length/.08
        print(f'The object is very large, down-scaling by {re_scale} factor')
        obj_mesh.v = obj_mesh.v/re_scale

    object_fullpts = obj_mesh.v
    maximum = object_fullpts.max(0, keepdims=True)
    minimum = object_fullpts.min(0, keepdims=True)

    offset = ( maximum + minimum) / 2
    verts_obj = object_fullpts - offset
    obj_mesh.v = verts_obj

    if rndrotate:
        obj_mesh.rotate_vertices(rand_rotmat)
    else:
        rand_rotmat = np.eye(3)

    mesh = M(vertices=obj_mesh.v, faces = obj_mesh.f)
    mesh = mesh.subdivide()
    obj_mesh = Mesh(v=mesh.vertices, f = mesh.faces, vc=name_to_rgb['green'])

    objmesh = open3d.geometry.TriangleMesh()
    objmesh.vertices = open3d.utility.Vector3dVector(np.copy(mesh.vertices))
    numVert = mesh.vertices.shape[0]
    objmesh.triangles = open3d.utility.Vector3iVector(np.copy(mesh.faces))
    objmesh.vertex_colors = open3d.utility.Vector3dVector(np.tile(np.array([[0.6, 0.2, 0.2]]), [numVert, 1]))
    

    while (obj_mesh.v.shape[0] < n_sample_verts):
        mesh = M(vertices=obj_mesh.v, faces = obj_mesh.f)
        mesh = mesh.subdivide()
        obj_mesh = Mesh(v=mesh.vertices, f = mesh.faces, vc=name_to_rgb['green'])

        objmesh = open3d.geometry.TriangleMesh()
        objmesh.vertices = open3d.utility.Vector3dVector(np.copy(mesh.vertices))
        numVert = mesh.vertices.shape[0]
        objmesh.triangles = open3d.utility.Vector3iVector(np.copy(mesh.faces))
        objmesh.vertex_colors = open3d.utility.Vector3dVector(np.tile(np.array([[0.6, 0.2, 0.2]]), [numVert, 1]))

    verts_obj = obj_mesh.v
    verts_sample_id = np.random.choice(verts_obj.shape[0], n_sample_verts, replace=False)
    verts_sampled = verts_obj[verts_sample_id]

    return verts_sampled, objmesh, rand_rotmat

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GrabNet-Testing')

    parser.add_argument('--obj-path', required = True, type=str,
                        help='The path to the 3D object Mesh or Pointcloud')
    parser.add_argument('--outdir', required = True, type=str,
                        help='Output image path')
    parser.add_argument('--rhm-path', required = True, type=str,
                        help='The path to the folder containing MANO_RIHGT model')
    parser.add_argument('--config-path', default= None, type=str,
                        help='The path to the confguration of the trained GrabNet model')
    parser.add_argument('--n_samples', default= 50, type=int,
                        help='Number of samples per object')
    parser.add_argument('--bs', default= 3, type=int,
                        help='batch size')

    args = parser.parse_args()

    cfg_path = args.config_path
    obj_path = args.obj_path # you could also revise as list of object folders [path1, path2]
    rhm_path = args.rhm_path
    n_samples = args.n_samples
    save_path = args.outdir
    
    # Create directory
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    rgb_path = os.path.join(save_path,'rgb')
    if  not os.path.exists(rgb_path):
        os.mkdir(rgb_path)

    seg_path = os.path.join(save_path, 'seg')
    if not os.path.exists(seg_path):
        os.mkdir(seg_path)

    seg_path2 = os.path.join(save_path, 'obj_mask')
    if not os.path.exists(seg_path2):
        os.mkdir(seg_path2)  

    mask_path = os.path.join(save_path, 'mask')
    if not os.path.exists(mask_path):
        os.mkdir(mask_path)   

    skeleton_path = os.path.join(save_path, 'skeleton')
    if not os.path.exists(skeleton_path):
        os.mkdir(skeleton_path)

    depth_path = os.path.join(save_path, 'depth')
    if not os.path.exists(depth_path):
        os.mkdir(depth_path) 

    label_path = os.path.join(save_path, 'label')
    if not os.path.exists(label_path):
        os.mkdir(label_path)

    cwd = os.getcwd()
    work_dir = cwd + '/logs'
    json_file_path_train = os.path.join(save_path, "train.csv")
    json_file_path_test = os.path.join(save_path, "test.csv")
    best_cnet = 'grabnet/models/coarsenet.pt'
    best_rnet = 'grabnet/models/refinenet.pt'
    bps_dir   = 'grabnet/configs/bps.npz'

    if cfg_path is None:
        cfg_path = 'grabnet/configs/grabnet_cfg.yaml'


    config = {
        'work_dir': work_dir,
        'best_cnet': best_cnet,
        'best_rnet': best_rnet,
        'bps_dir': bps_dir,
        'rhm_path': rhm_path
    }

    cfg = Config(default_cfg_path=cfg_path, **config)
    grabnet = Tester(cfg=cfg)
    all_data = [["image","skeleton","top","bottom","left","right","sentence","seg","mask","depth","label","objpath"]]
    paths = getObjPath(obj_path)
    random.shuffle(paths)
    # print(paths)
    num = max(1, int(0.1*len(paths)))
    per_samples = args.n_samples
    iters = max(1,per_samples // args.bs)
    print(f"======There are {len(paths)} objects models======")
    # test_iters = 200
    for it in tqdm(range(iters)):
        for path in paths:
            obj_data = grab_new_objs(grabnet, path, rot=True, n_samples=args.bs, it = it)
            all_data += obj_data
            with open(json_file_path_train, "a", newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerows(all_data)
                outfile.close()
            all_data = []


