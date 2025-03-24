#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.sh_utils import SH2RGB

import tinycudann as tcnn
from utils.gpcc_utils import compress_gpcc, decompress_gpcc, calculate_morton_order, float16_to_uint16, uint16_to_float16
from utils.compress_utils import *
import cupy as cp
from cuml.cluster import KMeans

def init_cdf_mask(importance, thres=1.0):
    importance = importance.flatten()   
    if thres!=1.0:
        percent_sum = thres
        vals,idx = torch.sort(importance+(1e-6))
        cumsum_val = torch.cumsum(vals, dim=0)
        split_index = ((cumsum_val/vals.sum()) > (1-percent_sum)).nonzero().min()
        split_val_nonprune = vals[split_index]

        non_prune_mask = importance>split_val_nonprune 
    else: 
        non_prune_mask = torch.ones_like(importance).bool()
        
    return non_prune_mask

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self.max_sh_rest = (sh_degree+1)**2 - 1
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._features_static = torch.empty(0)
        self._features_view = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        
        self.vq_enabled = False
        self.net_enabled = False

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.active_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        if self.net_enabled:
            l = [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                {'params': [self._features_static], 'lr': training_args.feature_lr, "name": "f_static"},
                {'params': [self._features_view], 'lr': training_args.feature_lr, "name": "f_view"},
            ]
        else:
            l = [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self.net_enabled:
            self._features_static = optimizable_tensors["f_static"]
            self._features_view = optimizable_tensors["f_view"]
        else:
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
            self._opacity = optimizable_tensors["opacity"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_static, new_view):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "f_static": new_static,
        "f_view": new_view}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self.net_enabled:
            self._features_static = optimizable_tensors["f_static"]
            self._features_view = optimizable_tensors["f_view"]
        else:
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
            self._opacity = optimizable_tensors["opacity"]
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        if self.net_enabled:
            new_static = self._features_static[selected_pts_mask].repeat(N,1)
            new_view = self._features_view[selected_pts_mask].repeat(N,1)
            self.densification_postfix(new_xyz, None, None, None, new_scaling, new_rotation, new_static, new_view)
        else:
            new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
            new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
            new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, None, None)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        if self.net_enabled:
            new_static = self._features_static[selected_pts_mask]
            new_view = self._features_view[selected_pts_mask]
            self.densification_postfix(new_xyz, None, None, None, new_scaling, new_rotation, new_static, new_view)
        else:
            new_features_dc = self._features_dc[selected_pts_mask]
            new_features_rest = self._features_rest[selected_pts_mask]
            new_opacities = self._opacity[selected_pts_mask]
            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, None, None)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1


    def densify_and_prune_split(self, max_grad, min_opacity, extent, max_screen_size, mask):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split_mask(grads, max_grad, extent, mask)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()


    def densify_and_split_mask(self, grads, grad_threshold, scene_extent, mask, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        padded_mask = torch.zeros((n_init_points), dtype=torch.bool, device='cuda')
        padded_mask[:grads.shape[0]] = mask
        selected_pts_mask = torch.logical_or(selected_pts_mask, padded_mask)
        

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        if self.net_enabled:
            new_static = self._features_static[selected_pts_mask].repeat(N,1)
            new_view = self._features_view[selected_pts_mask].repeat(N,1)
            self.densification_postfix(new_xyz, None, None, None, new_scaling, new_rotation, new_static, new_view)
        else:
            new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
            new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
            new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, None, None)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def depth_reinit(self, scene, render_depth, iteration, num_depth, args, pipe, background):

        out_pts_list=[]
        gt_list=[]
        views = scene.getTrainCameras()
        for view in views:
            gt = view.original_image[0:3, :, :]

            render_depth_pkg = render_depth(view, self, pipe, background)

            out_pts = render_depth_pkg["out_pts"]
            accum_alpha = render_depth_pkg["accum_alpha"]


            prob=1-accum_alpha

            prob = prob/prob.sum()
            prob = prob.reshape(-1).cpu().numpy()

            factor=1/(gt.shape[1]*gt.shape[2]*len(views)/num_depth)

            N_xyz=prob.shape[0]
            num_sampled=int(N_xyz*factor)

            indices = np.random.choice(N_xyz, size=num_sampled, 
                                        p=prob,replace=False)
            
            out_pts = out_pts.permute(1,2,0).reshape(-1,3)
            gt = gt.permute(1,2,0).reshape(-1,3)

            out_pts_list.append(out_pts[indices])
            gt_list.append(gt[indices])       


        out_pts_merged=torch.cat(out_pts_list)
        gt_merged=torch.cat(gt_list)

        return out_pts_merged, gt_merged
        

    def reinitial_pts(self, pts, rgb):

        fused_point_cloud = pts
        fused_color = RGB2SH(rgb)
        features = torch.zeros((fused_color.shape[0], 3, (self.active_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        # print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        if self.net_enabled:
            self._features_static = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
            self._features_view = nn.Parameter(torch.zeros((fused_point_cloud.shape[0], 3), device="cuda").requires_grad_(True))
        else:
            self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
            self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
            self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")     


    def intersection_sampling(self, scene, render, iteration, args, pipe, background):

        imp_score = torch.zeros(self._xyz.shape[0]).cuda()
        accum_area_max = torch.zeros(self._xyz.shape[0]).cuda()
        views = scene.getTrainCameras()
        for view in views:
            render_pkg = render(view, self, pipe, background)
            
            accum_weights = render_pkg["accum_weights"]
            area_proj = render_pkg["area_proj"]
            area_max = render_pkg["area_max"]

            accum_area_max = accum_area_max+area_max

            if args.imp_metric=='outdoor':
                mask_t=area_max!=0
                temp=imp_score+accum_weights/area_proj
                imp_score[mask_t] = temp[mask_t]
            else:
                imp_score=imp_score+accum_weights
        
        imp_score[accum_area_max==0]=0
        prob = imp_score/imp_score.sum()
        prob = prob.cpu().numpy()


        factor=args.sampling_factor
        N_xyz=self._xyz.shape[0]
        num_sampled=int(N_xyz*factor*((prob!=0).sum()/prob.shape[0]))
        indices = np.random.choice(N_xyz, size=num_sampled, 
                                    p=prob, replace=False)

        mask = np.zeros(N_xyz, dtype=bool)
        mask[indices] = True

        self.prune_points(mask==False)
        if self.net_enabled:
            rgb = SH2RGB(self._features_static+0)
        else:
            rgb = SH2RGB(self._features_dc+0)[:,0]
        return self._xyz, rgb

    def intersection_preserving(self, scene, render, iteration, args, pipe, background):

        imp_score = torch.zeros(self._xyz.shape[0]).cuda()
        accum_area_max = torch.zeros(self._xyz.shape[0]).cuda()
        views = scene.getTrainCameras()
        for view in views:
            render_pkg = render(view, self, pipe, background)
            
            accum_weights = render_pkg["accum_weights"]
            area_proj = render_pkg["area_proj"]
            area_max = render_pkg["area_max"]

            accum_area_max = accum_area_max+area_max

            if args.imp_metric=='outdoor':
                mask_t=area_max!=0 
                temp=imp_score+accum_weights/area_proj
                imp_score[mask_t] = temp[mask_t]
            else:
                imp_score=imp_score+accum_weights
            
        imp_score[accum_area_max==0]=0

        return imp_score

    def ld_scoring(self, imp_score, importance_thresh, lambda_ld):
        order = self.sort_morton()
        order_l = torch.clamp_min(order-1,0)
        order_r = torch.clamp_max(order+1,torch.amax(order))

        if not self.net_enabled:
            ori_color = self._features_dc[:,0]
            ordered_color = self._features_dc[order,0]
        else:
            ori_color = self._features_static
            ordered_color = self._features_static[order]
        res_color = torch.mean(torch.abs(ordered_color[order_l] - ori_color) + torch.abs(ordered_color[order_r] - ori_color),dim=-1)

        imp_score = imp_score * res_color**lambda_ld
        non_prune_mask = init_cdf_mask(importance=imp_score, thres=importance_thresh)
        
        self.prune_points(non_prune_mask==False)
        
    def construct_net(self, train=True):

        self.mlp_cont = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=13,
            encoding_config={
                "otype": "Frequency",
                "n_frequencies": 16,
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )
        self.mlp_view = tcnn.Network(
            n_input_dims=16,
            n_output_dims=3*self.max_sh_rest,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "LeakyReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )
    
        self.mlp_dc = tcnn.Network(
            n_input_dims=16,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "LeakyReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )
        
        self.mlp_opacity = tcnn.Network(
            n_input_dims=16,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "LeakyReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )

        if train:
            self.net_enabled = True
            self._features_static = nn.Parameter(self._features_dc[:, 0].clone().detach())
            self._features_view = nn.Parameter(torch.zeros((self.get_xyz.shape[0], 3), device="cuda").requires_grad_(True))
        
            mlp_params = []
            for params in self.mlp_cont.parameters():
                mlp_params.append(params)
            for params in self.mlp_view.parameters():
                mlp_params.append(params)
            for params in self.mlp_dc.parameters():
                mlp_params.append(params)
            for params in self.mlp_opacity.parameters():
                mlp_params.append(params)
                
            self.optimizer_net = torch.optim.Adam(mlp_params, lr=0.01, eps=1e-15)
            self.scheduler_net = torch.optim.lr_scheduler.ChainedScheduler(
            [
                torch.optim.lr_scheduler.LinearLR(
                self.optimizer_net, start_factor=0.01, total_iters=100
            ),
                torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer_net,
                milestones=[1_000, 3_500, 6_000],
                gamma=0.33,
            ),
            ]
            )
        
    def sort_morton(self):
        with torch.no_grad():
            xyz_q = (
                (2**21 - 1)
                * (self._xyz - self._xyz.min(0).values)
                / (self._xyz.max(0).values - self._xyz.min(0).values)
            ).long()
            order = mortonEncode(xyz_q).sort().indices
        return order
    
    def sort_attribute(self, order, xyz_only=False):
        self._xyz = nn.Parameter(self._xyz[order], requires_grad=True)
        if not xyz_only:
            # self._opacity = nn.Parameter(self._opacity[order], requires_grad=True)
            self._scaling = nn.Parameter(self._scaling[order], requires_grad=True)
            self._rotation = nn.Parameter(self._rotation[order], requires_grad=True)
            # self._features_dc = nn.Parameter(self._features_dc[order], requires_grad=True)
            # self._features_rest = nn.Parameter(self._features_rest[order], requires_grad=True)
            self._features_static = nn.Parameter(self._features_static[order], requires_grad=True)
            self._features_view = nn.Parameter(self._features_view[order], requires_grad=True)
            for i in range(len(self.scale_indices)):
                self.scale_indices[i] = self.scale_indices[i][order]
            for i in range(len(self.rotation_indices)):
                self.rotation_indices[i] = self.rotation_indices[i][order]
            for i in range(len(self.appearance_indices)):
                self.appearance_indices[i] = self.appearance_indices[i][order]
        return
    
    def contract_to_unisphere(self,
        x: torch.Tensor,
        aabb: torch.Tensor,
        ord: int = 2,
        eps: float = 1e-6,
        derivative: bool = False,
    ):
        aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
        x = (x - aabb_min) / (aabb_max - aabb_min)
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1

        if derivative:
            dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (
                1 / mag**3 - (2 * mag - 1) / mag**4
            )
            dev[~mask] = 1.0
            dev = torch.clamp(dev, min=eps)
            return dev
        else:
            x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
            x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
            return x

    def apply_svq(self, args):
        self.scale_codes = []
        self.scale_indices = []
        self.rotation_codes = []
        self.rotation_indices = []
        self.appearance_codes = []
        self.appearance_indices = []
        
        code_params = []
        self.kmeans(self._scaling, self.scale_codes, self.scale_indices, args.slice_scale, args.cluster_scale, code_params)
        self.kmeans(self._rotation, self.rotation_codes, self.rotation_indices, args.slice_rot, args.cluster_rot, code_params)
        self.kmeans(torch.cat([self._features_static, self._features_view],dim=-1), self.appearance_codes, self.appearance_indices, args.slice_app, args.cluster_app, code_params)

        self.optimizer_code = torch.optim.Adam(code_params, lr=1e-8, eps=1e-15)
        self.vq_enabled = True

    @property
    def get_svq_scale(self):
        scale = []
        for i in range(len(self.scale_codes)):
            scale.append(self.scale_codes[i][self.scale_indices[i]])
        return self.scaling_activation(torch.cat(scale, dim=-1))

    @property
    def get_svq_rotation(self):
        rotation = []
        for i in range(len(self.rotation_codes)):
            rotation.append(self.rotation_codes[i][self.rotation_indices[i]])
        return self.rotation_activation(torch.cat(rotation, dim=-1))
    
    @property
    def get_svq_appearance(self):
        appearance = []
        for i in range(len(self.appearance_codes)):
            appearance.append(self.appearance_codes[i][self.appearance_indices[i]])
        return torch.cat(appearance, dim=-1)
    
    def kmeans(self, param_data, code_list, index_list, svq_len, n_clusters, code_params):
        assert param_data.shape[1] % svq_len == 0, "invalid sub-vector length"
        for i in range(param_data.shape[1]//svq_len):
            input_cp = cp.asarray(param_data[:, i*svq_len:(i+1)*svq_len].detach().cpu())
            kmeans = KMeans(n_clusters=n_clusters, max_iter=1000, n_init=1)
            labels = kmeans.fit_predict(input_cp)
            cluster_centers = kmeans.cluster_centers_

            codebook = torch.nn.Parameter(torch.from_dlpack(cluster_centers)).cuda()
            index = torch.from_dlpack(labels).cuda().long()

            code_list.append(codebook)
            index_list.append(index)
            code_params.append(codebook) 

    def encode(self):
        save_dict = dict()
        xyz_uint16 = float16_to_uint16(self.get_xyz.half())
        sorted_indices = calculate_morton_order(xyz_uint16.int())
        self.sort_attribute(sorted_indices, xyz_only=False)
        xyz_uint16 = float16_to_uint16(self.get_xyz.half())
        save_dict['xyz'] = compress_gpcc(xyz_uint16)

        save_dict['scale_code'] = []
        save_dict['scale_index'] = []
        save_dict['scale_htable'] = []
        for i in range(len(self.scale_codes)):
            save_dict['scale_code'].append(self.scale_codes[i].half().cpu().numpy())
            huf_idx, huf_tab = huffman_encode(self.scale_indices[i].cpu().numpy())
            save_dict['scale_index'].append(huf_idx)
            save_dict['scale_htable'].append(huf_tab)

        save_dict['rotation_code'] = []
        save_dict['rotation_index'] = []
        save_dict['rotation_htable'] = []
        for i in range(len(self.rotation_codes)):
            save_dict['rotation_code'].append(self.rotation_codes[i].half().cpu().numpy())
            huf_idx, huf_tab = huffman_encode(self.rotation_indices[i].cpu().numpy())
            save_dict['rotation_index'].append(huf_idx)
            save_dict['rotation_htable'].append(huf_tab)

        save_dict['app_code'] = []
        save_dict['app_index'] = []
        save_dict['app_htable'] = []
        for i in range(len(self.appearance_codes)):
            save_dict['app_code'].append(self.appearance_codes[i].half().cpu().numpy())
            huf_idx, huf_tab = huffman_encode(self.appearance_indices[i].cpu().numpy())
            save_dict['app_index'].append(huf_idx)
            save_dict['app_htable'].append(huf_tab)
                                                                               
        save_dict['MLP_cont'] = self.mlp_cont.params.half().cpu().numpy()
        save_dict['MLP_dc'] = self.mlp_dc.params.half().cpu().numpy()
        save_dict['MLP_sh'] = self.mlp_view.params.half().cpu().numpy()
        save_dict['MLP_opacity'] = self.mlp_opacity.params.half().cpu().numpy()     
                
        return save_dict

    def decode(self, save_dict, decompress=True):
        self.vq_enabled = False
        self.net_enabled = False
        
        means_strings = save_dict['xyz']
        xyz_uint16 = decompress_gpcc(means_strings).to('cuda')
        sorted_indices = calculate_morton_order(xyz_uint16.int())
        self._xyz = uint16_to_float16(xyz_uint16).float()
        self.sort_attribute(sorted_indices, xyz_only=True)

        scale = []
        rotation = []
        appearance = []

        if decompress:
            for i in range(len(save_dict['scale_code'])):
                labels = huffman_decode(save_dict['scale_index'][i], save_dict['scale_htable'][i])
                cluster_centers = save_dict['scale_code'][i]
                scale.append(torch.tensor(cluster_centers[labels]).cuda())
            self._scaling = torch.cat(scale, dim=-1).float()
            
            for i in range(len(save_dict['rotation_code'])):
                labels = huffman_decode(save_dict['rotation_index'][i], save_dict['rotation_htable'][i])
                cluster_centers = save_dict['rotation_code'][i]
                rotation.append(torch.tensor(cluster_centers[labels]).cuda())
            self._rotation = torch.cat(rotation, dim=-1).float()
            
            for i in range(len(save_dict['app_code'])):
                labels = huffman_decode(save_dict['app_index'][i], save_dict['app_htable'][i])
                cluster_centers = save_dict['app_code'][i]
                appearance.append(torch.tensor(cluster_centers[labels]).cuda())
            app_feature = torch.cat(appearance, dim=-1).float()

            self.mlp_cont.params = torch.nn.Parameter(torch.tensor(save_dict['MLP_cont']).cuda().half().requires_grad_(True))
            self.mlp_dc.params = torch.nn.Parameter(torch.tensor(save_dict['MLP_dc']).cuda().half().requires_grad_(True))
            self.mlp_view.params = torch.nn.Parameter(torch.tensor(save_dict['MLP_sh']).cuda().half().requires_grad_(True))
            self.mlp_opacity.params = torch.nn.Parameter(torch.tensor(save_dict['MLP_opacity']).cuda().half().requires_grad_(True))
        
        else:
            for i in range(len(self.scale_codes)):
                scale.append(self.scale_codes[i][self.scale_indices[i]])
            self._scaling = torch.cat(scale, dim=-1).float()
            
            for i in range(len(self.rotation_codes)):
                rotation.append(self.rotation_codes[i][self.rotation_indices[i]])
            self._rotation = torch.cat(rotation, dim=-1).float()
            
            for i in range(len(self.appearance_codes)):
                appearance.append(self.appearance_codes[i][self.appearance_indices[i]])
            app_feature = torch.cat(appearance, dim=-1).float()
            
        cont_feature = self.mlp_cont(self.contract_to_unisphere(self.get_xyz.clone().detach(), torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device='cuda')))
        space_feature = torch.cat([cont_feature, app_feature[:,0:3]],dim=-1)
        view_feature = torch.cat([cont_feature, app_feature[:,3:6]],dim=-1)

        self._features_rest = self.mlp_view(view_feature).reshape(-1,self.max_sh_rest,3).float()
        self._features_dc = self.mlp_dc(space_feature).reshape(-1,1,3).float()
        self._opacity = self.mlp_opacity(space_feature).float()

        del self._features_static
        del self._features_view


