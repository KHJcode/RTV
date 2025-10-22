import copy
import cv2
import romp
from romp import ROMP
from romp.post_parser import SMPL_parser, body_mesh_projection2image, parsing_outputs
from romp.utils import convert_cam_to_3d_trans, convert_tensor2numpy
from SMPL.smpl_np_base import SMPLModel
from SMPL.projection2screen import projection2screencoord
import torch
import glm
import numpy as np
import bev
from bev import BEV
from SMPL.my_bev import MyBEV
from scipy.spatial.transform import Rotation as R
from util.image_process import blur_image
from util.torch_device import ensure_compatible_cuda






# settings = romp.main.default_settings
# settings is just a argparse Namespace. To change it, for instance, you can change mode via
# settings.mode='video'
# romp_model = romp.ROMP(settings)
# outputs = romp_model(cv2.imread('./humanbody.png')) # please note that we take the input image in BGR format (cv2.imread).
# dict_keys=list(outputs.keys())
# print(outputs)
class MyROMP(ROMP):
    def __init__(self, romp_settings):
        super(MyROMP, self).__init__(romp_settings)

    def forward(self, image, signal_ID=0, **kwargs):
        outputs, image_pad_info = self.single_image_forward(image)
        if outputs is None:
            return None
        if self.settings.temporal_optimize:
            outputs = self.temporal_optimization(outputs, signal_ID)
        outputs['cam_trans'] = convert_cam_to_3d_trans(outputs['cam'])

        outputs = self.smpl_parser(outputs, root_align=self.settings.root_align)
        outputs.update(body_mesh_projection2image(outputs['joints'], outputs['cam'], vertices=outputs['verts'],
                                                  input2org_offsets=image_pad_info))
        # print(outputs.keys())
        outputs = self.smpl_parser.forward(outputs)

        return outputs  # convert_tensor2numpy(outputs)


# https://github.com/Arthur151/ROMP/blob/master/simple_romp/README.md
class SMPL_Regressor:
    def __init__(self,use_bev=True, fix_body=False):

        self.regressor_model = self.create_romp_model() if not use_bev else self.create_bev_model(fix_body)
        self.smpl = SMPLModel()
        #self.smpl.beta[0]+=1.2
        self.smpl.beta[1]+=1.2#todo:???
        self.smpl.update()

    def create_bev_model(self,fix_body=False):
        settings = copy.deepcopy(bev.main.default_settings)
        settings.mode = 'video'
        device = ensure_compatible_cuda()
        settings.GPU = device.index if device.type == 'cuda' else -1
        bev_model = MyBEV(settings,fix_body=fix_body)
        return bev_model

    def create_romp_model(self):
        settings = copy.deepcopy(romp.main.default_settings)
        settings.temporal_optimize = False
        settings.calc_smpl = True
        settings.render_mesh = True
        settings.smooth_coeff = 3.0
        # settings is just a argparse Namespace. To change it, for instance, you can change mode via
        settings.mode = 'video'
        device = ensure_compatible_cuda()
        settings.GPU = device.index if device.type == 'cuda' else -1
        romp_model = MyROMP(settings)
        return romp_model

    def get_rotation_angle(self, smpl_param):

        thetas = smpl_param['smpl_thetas']
        cam_trans = smpl_param['cam_trans']
        depth_order = torch.sort(cam_trans[:, 2].cpu(), descending=False).indices.numpy()
        thetas = thetas[depth_order]
        thetas = thetas.cpu().numpy()
        #print(theta.shape)
        #print(thetas.shape)
        rot_vec = R.from_rotvec(thetas[0][0:3])
        angles = rot_vec.as_euler('XYZ', degrees=False)
        return angles[1]

    def get_thetas(self, smpl_param):

        thetas = smpl_param['smpl_thetas']
        cam_trans = smpl_param['cam_trans']
        depth_order = torch.sort(cam_trans[:, 2].cpu(), descending=False).indices.numpy()
        thetas = thetas[depth_order]
        thetas = thetas.cpu().numpy()
        return thetas[0]

    def get_thin_verts(self, smpl_param):

        thetas = smpl_param['smpl_thetas']
        beta = smpl_param['smpl_betas']
        #print(beta)
        cam_trans = smpl_param['cam_trans']
        depth_order = torch.sort(cam_trans[:, 2].cpu(), descending=False).indices.numpy()
        verts_tran = smpl_param['verts'][depth_order]
        verts_tran = verts_tran[0].cpu().numpy()
        thetas = thetas[depth_order]
        thetas = thetas.cpu().numpy()
        theta = thetas[0]
        self.smpl.pose=theta
        self.smpl.update()
        verts = self.smpl.verts
        verts = verts - verts.mean(0) + verts_tran.mean(0)

        return verts

    def get_tshirt_verts(self, smpl_param):

        thetas = smpl_param['smpl_thetas']
        beta = smpl_param['smpl_betas']
        #print(beta)
        cam_trans = smpl_param['cam_trans']
        depth_order = torch.sort(cam_trans[:, 2].cpu(), descending=False).indices.numpy()
        verts_tran = smpl_param['verts'][depth_order]
        verts_tran = verts_tran[0].cpu().numpy()
        thetas = thetas[depth_order]
        thetas = thetas.cpu().numpy()
        theta = thetas[0]
        theta=theta.reshape(24,3)
        #print(theta.shape)
        theta[[18,19],:]=0
        theta=theta.reshape(72)
        self.smpl.pose=theta
        self.smpl.update()
        #print(theta.shape)
        self.smpl.tshirt.set_params(pose=theta.reshape(24,3))
        #verts = self.smpl.verts
        #verts = verts - verts.mean(0) + verts_tran.mean(0)
        tshirt_verts = self.smpl.tshirt.verts#- verts.mean(0) + verts_tran.mean(0)
        tshirt_verts[:,1]+=0.25 #larger-> down

        return tshirt_verts, self.smpl.tshirt.uv, self.smpl.tshirt.faces

    @classmethod
    def get_raw_verts(cls, smpl_param):

        thetas = smpl_param['smpl_thetas']
        beta = smpl_param['smpl_betas']
        #print(beta)
        cam_trans = smpl_param['cam_trans']
        depth_order = torch.sort(cam_trans[:, 2].cpu(), descending=False).indices.numpy()
        verts_tran = smpl_param['verts'][depth_order]
        #print(verts_tran.shape,cam_trans.shape)
        verts_tran[0] +=cam_trans[depth_order][0]
        verts_tran[:,:,2]*=-1
        verts_tran = verts_tran[0].cpu().numpy()
        return verts_tran


    def forward(self, img, roi=False,size=1.2,roi_img_size=512):
        # ['cam', 'global_orient', 'body_pose', 'smpl_betas', 'smpl_thetas', 'center_preds', 'center_confs', 'cam_trans', 'verts', 'joints', 'pj2d_org']
        outputs = self.regressor_model.forward(img)
        #outputs = self.romp_model.smpl_parser.forward(outputs)
        #print(outputs['pj2d_org'].shape)
        #print(outputs['joints'].shape)
        if roi:
            if outputs is None:
                return outputs, None,None
            cam_trans = outputs['cam_trans']
            depth_order = torch.sort(cam_trans[:, 2].cpu(), descending=False).indices.numpy()
            #verts_tran = (outputs['verts'] + cam_trans.unsqueeze(1))[depth_order]

            #verts_tran[:, :, 2] = verts_tran[:, :, 2] * -1
            #verts_tran = verts_tran[0].cpu().numpy()
            Joints = outputs['pj2d_org'][depth_order][0].cpu().numpy()

            height = img.shape[0]
            width = img.shape[1]
            trans2roi, inv_trans2roi = self.get_trans2roi(Joints,s=size,img_size=roi_img_size)
            return outputs, trans2roi, inv_trans2roi
        else:
            return outputs

    def get_face_region(self, outputs):
        cam_trans = outputs['cam_trans']
        depth_order = torch.sort(cam_trans[:, 2].cpu(), descending=False).indices.numpy()
        verts_tran = (outputs['verts'] + cam_trans.unsqueeze(1))[depth_order]

        verts_tran[:, :, 2] = verts_tran[:, :, 2] * -1
        verts_tran = verts_tran[0].cpu().numpy()
        Joints = outputs['pj2d_org'][depth_order][0].cpu().numpy()
        face_center = Joints[15] + (Joints[15]-Joints[12])*1.3
        face_radius = np.linalg.norm((Joints[15]+Joints[12])*0.5 - face_center)*0.6
        return face_center, face_radius

    def blur_face(self,raw_img,smpl_param):
        face_center, face_radius = self.get_face_region(smpl_param)
        raw_image = blur_image(raw_img.copy(), face_center, face_radius)
        return raw_image


    def get_trans2roi(self, Joints,s=1.2,img_size=512):

        src = np.zeros([3, 2], np.float32)
        center = Joints[9]*0.8+Joints[12]*0.2
        size = np.linalg.norm(Joints[15]-Joints[0])*s
        src[0, :] = center + np.array([-size,size],np.float32)
        src[1, :] = center + np.array([-size,-size],np.float32)
        src[2, :] = center + np.array([size,-size],np.float32)


        dst = np.zeros([3, 2], np.float32)
        dst[0, :] = np.array([0,img_size-1],np.float32)
        dst[1, :] = np.array([0,0],np.float32)
        dst[2, :] = np.array([img_size-1,0],np.float32)
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        inv_trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        return trans, inv_trans

    def get_fullbody_trans2roi(self, smpl_param,s=1.38,new_h=1024,new_w=768):
        cam_trans = smpl_param['cam_trans']
        depth_order = torch.sort(cam_trans[:, 2].cpu(), descending=False).indices.numpy()
        # verts_tran = (outputs['verts'] + cam_trans.unsqueeze(1))[depth_order]

        # verts_tran[:, :, 2] = verts_tran[:, :, 2] * -1
        # verts_tran = verts_tran[0].cpu().numpy()
        Joints = smpl_param['pj2d_org'][depth_order][0].cpu().numpy()

        src = np.zeros([3, 2], np.float32)
        center = Joints[0]
        size = np.linalg.norm(Joints[15]-Joints[10]*0.5-Joints[11]*0.5)*s
        half_h = size/2
        half_w = half_h*(new_w/new_h)
        src[0, :] = center + np.array([-half_w,half_h],np.float32)
        src[1, :] = center + np.array([-half_w,-half_h],np.float32)
        src[2, :] = center + np.array([half_w,-half_h],np.float32)


        dst = np.zeros([3, 2], np.float32)
        dst[0, :] = np.array([0,new_h-1],np.float32)
        dst[1, :] = np.array([0,0],np.float32)
        dst[2, :] = np.array([new_w-1,0],np.float32)
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        inv_trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        return trans, inv_trans
