# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ProDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from ProDepth.layers import BackprojectDepth, Project3D, disp_to_depth, depth_to_disp
from ProDepth.networks.depth_encoder import LiteMono, LiteCVEncoder

from collections import OrderedDict
from ProDepth.layers import ConvBlock, Conv3x3, upsample
from ProDepth import datasets, networks

import math

class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class ResnetEncoderMatching(nn.Module):
    """Resnet encoder adapted to include a cost volume after the 2nd block.

    Setting adaptive_bins=True will recompute the depth bins used for matching upon each
    forward pass - this is required for training from monocular video as there is an unknown scale.
    """

    def __init__(self, num_layers, pretrained, encoder, input_height, input_width,
                 min_depth_bin=0.1, max_depth_bin=20.0, num_depth_bins=96,
                 adaptive_bins=False, depth_binning='linear'):

        super(ResnetEncoderMatching, self).__init__()

        self.adaptive_bins = adaptive_bins
        self.depth_binning = depth_binning
        self.set_missing_to_max = True
        self.encoder = encoder
        
        if self.encoder == 'resnet':
            self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        elif self.encoder == 'lite':
            self.num_ch_enc = np.array([64, 64, 128, 224])
        elif self.encoder == 'lvt':
            self.num_ch_enc = np.array([64, 64, 160, 256])
            

        self.num_depth_bins = num_depth_bins
        # we build the cost volume at 1/4 resolution
        self.matching_height, self.matching_width = input_height // 4, input_width // 4

        self.is_cuda = False
        self.warp_depths = None
        self.depth_bins = None
        if self.encoder == 'resnet':
            resnets = {18: models.resnet18,
                    34: models.resnet34,
                    50: models.resnet50,
                    101: models.resnet101,
                    152: models.resnet152}

            if num_layers not in resnets:
                raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

            encoder = resnets[num_layers](pretrained)
            self.layer0 = nn.Sequential(encoder.conv1,  encoder.bn1, encoder.relu)
            self.layer1 = nn.Sequential(encoder.maxpool,  encoder.layer1)
            self.layer2 = encoder.layer2
            self.layer3 = encoder.layer3
            self.layer4 = encoder.layer4

            if num_layers > 34:
                self.num_ch_enc[1:] *= 4

        self.backprojector = BackprojectDepth(batch_size=self.num_depth_bins,
                                            height=self.matching_height,
                                            width=self.matching_width)
        self.projector = Project3D(batch_size=self.num_depth_bins,
                                height=self.matching_height,
                                width=self.matching_width)

        self.compute_depth_bins(min_depth_bin, max_depth_bin)

        self.prematching_conv = nn.Sequential(nn.Conv2d(64, out_channels=16,
                                                        kernel_size=1, stride=1, padding=0),
                                            nn.ReLU(inplace=True)
                                            )

        self.reduce_conv = nn.Sequential(nn.Conv2d(self.num_ch_enc[1] + self.num_depth_bins,
                                                   out_channels=self.num_ch_enc[1],
                                                   kernel_size=3, stride=1, padding=1),
                                         nn.ReLU(inplace=True)
                                         )
        


        self.cv_encoder = LiteCVEncoder(model='lite-mono-8m', drop_path_rate=0.4)
        self.load_cv_pretrain()
        self.cv_decoder = networks.DepthDecoder(self.cv_encoder.num_ch_enc, [0, 1, 2])

        if self.encoder == 'lite':
            self.lite_encoder = LiteMono(model='lite-mono-8m', drop_path_rate=0.4)
            self.load_pretrain()


    def compute_depth_bins(self, min_depth_bin, max_depth_bin):
        """Compute the depths bins used to build the cost volume. Bins will depend upon
        self.depth_binning, to either be linear in depth (linear) or linear in inverse depth
        (inverse)"""

        if self.depth_binning == 'inverse':
            self.depth_bins = 1 / np.linspace(1 / max_depth_bin,
                                              1 / min_depth_bin,
                                              self.num_depth_bins)[::-1]  # maintain depth order

        elif self.depth_binning == 'linear':
            self.depth_bins = np.linspace(min_depth_bin, max_depth_bin, self.num_depth_bins)

        elif self.depth_binning == 'sid':
            self.depth_bins = np.array(
                [np.exp(np.log(min_depth_bin) + np.log(max_depth_bin / min_depth_bin) * i / (self.num_depth_bins - 1))
                for i in range(self.num_depth_bins)])
        else:
            raise NotImplementedError
        
        self.depth_bins = torch.from_numpy(self.depth_bins).float()

        self.warp_depths = []
        for depth in self.depth_bins:
            depth = torch.ones((1, self.matching_height, self.matching_width)) * depth
            self.warp_depths.append(depth)
        self.warp_depths = torch.stack(self.warp_depths, 0).float()
        if self.is_cuda:
            self.warp_depths = self.warp_depths.cuda()

    def match_features(self, current_feats, lookup_feats, relative_poses, K, invK):
        """Compute a cost volume based on L1 difference between current_feats and lookup_feats.

        We backwards warp the lookup_feats into the current frame using the estimated relative
        pose, known intrinsics and using hypothesised depths self.warp_depths (which are either
        linear in depth or linear in inverse depth).

        If relative_pose == 0 then this indicates that the lookup frame is missing (i.e. we are
        at the start of a sequence), and so we skip it"""

        batch_cost_volume = []  # store all cost volumes of the batch
        cost_volume_masks = []  # store locations of '0's in cost volume for confidence
        cv_features = []
        
        for batch_idx in range(len(current_feats)):

            volume_shape = (self.num_depth_bins, self.matching_height, self.matching_width)
            cost_volume = torch.zeros(volume_shape, dtype=torch.float, device=current_feats.device)
            counts = torch.zeros(volume_shape, dtype=torch.float, device=current_feats.device)
            
            # select an item from batch of ref feats
            _lookup_feats = lookup_feats[batch_idx:batch_idx + 1]
            _lookup_poses = relative_poses[batch_idx:batch_idx + 1]

            _K = K[batch_idx:batch_idx + 1]
            _invK = invK[batch_idx:batch_idx + 1]
            world_points = self.backprojector(self.warp_depths, _invK)

            # loop through ref images adding to the current cost volume
            for lookup_idx in range(_lookup_feats.shape[1]):
                lookup_feat = _lookup_feats[:, lookup_idx]  # 1 x C x H x W
                lookup_pose = _lookup_poses[:, lookup_idx]

                # ignore missing images
                if lookup_pose.sum() == 0:
                    continue

                lookup_feat = lookup_feat.repeat([self.num_depth_bins, 1, 1, 1])
                pix_locs = self.projector(world_points, _K, lookup_pose)
                warped = F.grid_sample(lookup_feat, pix_locs, padding_mode='zeros', mode='bilinear',
                                       align_corners=True)

                
                # mask values landing outside the image (and near the border)
                # we want to ignore edge pixels of the lookup images and the current image
                # because of zero padding in ResNet
                # Masking of ref image border
                x_vals = (pix_locs[..., 0].detach() / 2 + 0.5) * (
                    self.matching_width - 1)  # convert from (-1, 1) to pixel values
                y_vals = (pix_locs[..., 1].detach() / 2 + 0.5) * (self.matching_height - 1)

                edge_mask = (x_vals >= 2.0) * (x_vals <= self.matching_width - 2) * \
                            (y_vals >= 2.0) * (y_vals <= self.matching_height - 2)
                edge_mask = edge_mask.float()

                # masking of current image
                current_mask = torch.zeros_like(edge_mask)
                current_mask[:, 2:-2, 2:-2] = 1.0
                edge_mask = edge_mask * current_mask

                diffs = torch.abs(warped - current_feats[batch_idx:batch_idx + 1]).mean(
                    1) * edge_mask

                # integrate into cost volume
                cost_volume = cost_volume + diffs
                counts = counts + (diffs > 0).float()
                

                
            # average over lookup images
            cost_volume = cost_volume / (counts + 1e-7)
            cv_feature = cost_volume.clone().detach()

            # if some missing values for a pixel location (i.e. some depths landed outside) then
            # set to max of existing values
            missing_val_mask = (cost_volume == 0).float()
            if self.set_missing_to_max:
                cost_volume = cost_volume * (1 - missing_val_mask) + \
                    cost_volume.max(0)[0].unsqueeze(0) * missing_val_mask
            batch_cost_volume.append(cost_volume)
            cost_volume_masks.append(missing_val_mask)
            
            cv_features.append(cv_feature)
            
        batch_cost_volume = torch.stack(batch_cost_volume, 0)
        cost_volume_masks = torch.stack(cost_volume_masks, 0)
        
        cv_features = torch.stack(cv_features,0)
        

        return batch_cost_volume, cost_volume_masks, cv_features

    def feature_extraction(self, image, return_all_feats=False):
        """ Run feature extraction on an image - first 2 blocks of ResNet"""

        image = (image - 0.45) / 0.225  # imagenet normalisation
        feats_0 = self.layer0(image)
        feats_1 = self.layer1(feats_0)

        if return_all_feats:
            return [feats_0, feats_1]
        else:
            return feats_1

    def indices_to_disparity(self, indices):
        """Convert cost volume indices to 1/depth for visualisation"""

        batch, height, width = indices.shape
        depth = self.depth_bins[indices.reshape(-1).cpu()]
        disp = 1 / depth.reshape((batch, height, width))
        return disp

    def compute_confidence_mask(self, cost_volume, num_bins_threshold=None):
        """ Returns a 'confidence' mask based on how many times a depth bin was observed"""

        if num_bins_threshold is None:
            num_bins_threshold = self.num_depth_bins
        confidence_mask = ((cost_volume > 0).sum(1) == num_bins_threshold).float()

        return confidence_mask

    def forward(self, current_image, lookup_images, poses, K, invK,
                min_depth_bin=None, max_depth_bin=None, mono_disp=None, var = None,
                ):
        #+++++++++++++++++++++++++++++++++++++++++++++++++#
        if self.encoder == 'resnet':
            # feature extraction
            self.features = self.feature_extraction(current_image, return_all_feats=True)
            current_feats = self.features[-1]

            # feature extraction on lookup images - disable gradients to save memory
            with torch.no_grad():
                if self.adaptive_bins:
                    self.compute_depth_bins(min_depth_bin, max_depth_bin)

                batch_size, num_frames, chns, height, width = lookup_images.shape
                lookup_images = lookup_images.reshape(batch_size * num_frames, chns, height, width)
                lookup_feats = self.feature_extraction(lookup_images,
                                                    return_all_feats=False)
                _, chns, height, width = lookup_feats.shape
                lookup_feats = lookup_feats.reshape(batch_size, num_frames, chns, height, width)

                # warp features to find cost volume
                cost_volume, missing_mask, cv_feature = \
                    self.match_features(current_feats, lookup_feats, poses, K, invK)
                confidence_mask = self.compute_confidence_mask(cost_volume.detach() *
                                                            (1 - missing_mask.detach()))
            #+++++++++++++++++++++++++++++++++++++++++++++++++#
            # CV Deocder
            cv_tmp, cv_x, mono_features = self.cv_encoder.forward_features(current_image)
            self.cv_features = self.cv_encoder(cv_tmp, cv_x, mono_features ,cv_feature) 
            self.cv_features = self.cv_decoder(self.cv_features)[('disp',0)].squeeze(1)
            
        #+++++++++++++++++++++++++++++++++++++++++++++++++#
        elif self.encoder == 'lite':
            tmp, x, self.features = self.lite_encoder.forward_features(current_image)

            lite_current_feats = self.features[-1]  # [B, C, H, W]

            # feature extraction on lookup images - disable gradients to save memory
            with torch.no_grad():
                if self.adaptive_bins:
                    self.compute_depth_bins(min_depth_bin, max_depth_bin)

                batch_size, num_frames, chns, height, width = lookup_images.shape
                lookup_images = lookup_images.reshape(batch_size * num_frames, chns, height, width)
                _, _, feats = self.lite_encoder.forward_features(lookup_images) # [B, C, H, W]
                lite_lookup_feats = feats[-1]
                _, chns, height, width = lite_lookup_feats.shape
                lite_lookup_feats = lite_lookup_feats.reshape(batch_size, num_frames, chns, height, width)
                
                # warp features to find cost volume
                cost_volume, missing_mask, cv_feature = \
                    self.match_features(lite_current_feats, lite_lookup_feats, poses, K, invK)
                confidence_mask = self.compute_confidence_mask(cost_volume.detach() *
                                                            (1 - missing_mask.detach()))
            #+++++++++++++++++++++++++++++++++++++++++++++++++#
            # CV Decoder
            cv_tmp = [t.clone().detach() for t in tmp]
            cv_x = [t.clone().detach() for t in x]
            mono_features = [t.clone().detach() for t in self.features]

            self.cv_features = self.cv_encoder(cv_tmp, cv_x, mono_features ,cv_feature) 
            self.cv_features = self.cv_decoder(self.cv_features)[('disp',0)].squeeze(1)    
        
        #+++++++++++++++++++++++++++++++++++++++++++++++++#
        # CV Refine
        cost_volume_distribution = cost_volume.clone().detach()

        cv_max,_ = torch.max(cost_volume_distribution,1)
        cv_min,_ = torch.min(cost_volume_distribution,1)

        cost_volume_distribution = (cost_volume_distribution - cv_min.unsqueeze(1)) / (cv_max.unsqueeze(1) - cv_min.unsqueeze(1) + 1e-7)
        
        mono_output = F.interpolate(mono_disp.detach(), [mono_disp.shape[-2] // 4, mono_disp.shape[-1] // 4], mode="bilinear")
        _mono_disp, _mono_depth = disp_to_depth(mono_output,0.1, 100)
        
        d_i = self.depth_bins.view(1, -1, 1, 1).repeat(_mono_depth.shape[0],1,_mono_depth.shape[-2],_mono_depth.shape[-1]).cuda()

        sigma = F.interpolate(var.detach(), [mono_disp.shape[-2] // 4, mono_disp.shape[-1] // 4], mode="bilinear") 
        gaussian_mono_distribution = (1 / (sigma * math.sqrt(2 * math.pi))) * torch.exp(-((d_i - _mono_depth) ** 2) / (2 * sigma ** 2))

        gmd_max,_ = torch.max(gaussian_mono_distribution,1)
        gmd_min,_ = torch.min(gaussian_mono_distribution,1)
        gaussian_mono_distribution = (gaussian_mono_distribution - gmd_min.unsqueeze(1)) / (gmd_max.unsqueeze(1) - gmd_min.unsqueeze(1) + 1e-7)
        
        cv_output = F.interpolate(self.cv_features.unsqueeze(1).detach(), [mono_disp.shape[-2] // 4, mono_disp.shape[-1] // 4], mode="bilinear")
        matching_disp, matching_depth = disp_to_depth(cv_output.detach(),0.1, 100)

        depth_mask = torch.exp(-torch.abs(matching_depth - _mono_depth)*(2/3))
        disp_mask = torch.exp(-torch.abs(matching_disp - _mono_disp)*(2/3))
        weighted_mask = disp_mask * depth_mask  # 0: Moving, 1: Static
        binary_mask = (weighted_mask > 0.4).float()

        softmax_cv_distribution = torch.softmax(-cost_volume_distribution,1)

        sfm_cvd_max,_ = torch.max(softmax_cv_distribution,1)
        sfm_cvd_min,_ = torch.min(softmax_cv_distribution,1)
        softmax_cv_distribution = (softmax_cv_distribution - sfm_cvd_min.unsqueeze(1)) / (sfm_cvd_max.unsqueeze(1) - sfm_cvd_min.unsqueeze(1) + 1e-7)

        refined_cost_volume_distribution = (softmax_cv_distribution**(weighted_mask) * gaussian_mono_distribution**(1-weighted_mask))

        refined_cost_volume_distribution = binary_mask * softmax_cv_distribution + (1-binary_mask) * refined_cost_volume_distribution

        refined_max, _ = torch.max(refined_cost_volume_distribution,1)
        refined_min, _ = torch.min(refined_cost_volume_distribution,1)
        

        refined_cost_volumes = ((refined_max.unsqueeze(1) - refined_cost_volume_distribution) / (refined_max.unsqueeze(1) - refined_min.unsqueeze(1) + 1e-7)) * (cv_max.unsqueeze(1)-cv_min.unsqueeze(1)) + cv_min.unsqueeze(1)
        
        maxs, argmax = torch.max(refined_cost_volumes, 1)
        refined_lowest_cost = self.indices_to_disparity(argmax)
        
        maxs,argmax = torch.max(gaussian_mono_distribution,1)
        gaussian_cost = self.indices_to_disparity(argmax)
        #+++++++++++++++++++++++++++++++++++++++++++++++++#
            

        # for visualisation - ignore 0s in cost volume for minimum
        viz_cost_vol = cost_volume.clone().detach()
        viz_cost_vol[viz_cost_vol == 0] = 100
        mins, argmin = torch.min(viz_cost_vol, 1)
        lowest_cost = self.indices_to_disparity(argmin)

        # mask the cost volume based on the confidence
        cost_volume *= confidence_mask.unsqueeze(1)

        post_matching_feats = self.reduce_conv(torch.cat([self.features[-1], refined_cost_volumes], 1))

        if self.encoder == 'resnet':
            self.features.append(self.layer2(post_matching_feats))
            self.features.append(self.layer3(self.features[-1]))
            self.features.append(self.layer4(self.features[-1]))
        elif self.encoder == "lite":
            tmp[-1] = post_matching_feats
            self.features = self.lite_encoder.forward_features2(tmp, x, self.features)
            
        return self.features, lowest_cost, confidence_mask, self.cv_features, binary_mask, refined_lowest_cost, gaussian_cost
            
    def cuda(self):
        super().cuda()
        self.backprojector.cuda()
        self.projector.cuda()
        self.is_cuda = True
        if self.warp_depths is not None:
            self.warp_depths = self.warp_depths.cuda()

    def cpu(self):
        super().cpu()
        self.backprojector.cpu()
        self.projector.cpu()
        self.is_cuda = False
        if self.warp_depths is not None:
            self.warp_depths = self.warp_depths.cpu()

    def to(self, device):
        if str(device) == 'cpu':
            self.cpu()
        elif str(device) == 'cuda':
            self.cuda()
        else:
            raise NotImplementedError
        
    def load_pretrain(self):
        path = os.path.expanduser("<PATH TO lite-mono-8m-pretrain.pth>")
        model_dict = self.lite_encoder.state_dict()
        pretrained_dict = torch.load(path)['model']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and not k.startswith('norm'))}
        model_dict.update(pretrained_dict)
        self.lite_encoder.load_state_dict(model_dict)
        print('MULTI ENCODER loaded.')

    def load_cv_pretrain(self):
        path = os.path.expanduser("<PATH TO lite-mono-8m-pretrain.pth>")
        model_dict = self.cv_encoder.state_dict()
        pretrained_dict = torch.load(path)['model']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and not k.startswith('norm'))}
        model_dict.update(pretrained_dict)
        self.cv_encoder.load_state_dict(model_dict)
        print('CV ENCODER loaded.')

class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """

    def __init__(self, num_layers, pretrained, num_input_images=1, **kwargs):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features
