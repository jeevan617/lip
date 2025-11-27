import torch
import torchvision
import numpy as np
import cv2
import os
import sys

# Add the Pytorch_Retinaface directory to the Python path
sys.path.append(os.path.join(os.getcwd(), 'Pytorch_Retinaface'))

from data import cfg_re50, cfg_mnet
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, device):
    print(f'-- Loading pretrained model from {pretrained_path}')
    # Use a string for the device, as that's what's passed in
    if str(device) == 'cpu':
        print('-- Loading model to CPU')
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        print('-- Loading model to GPU')
        # This will still fail if CUDA is not available, but the logic is now corrected
        # to check the string value of the device.
        device_id = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device_id))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    print('-- Model loaded successfully')
    return model

class LandmarksDetector:
    def __init__(self, device="cuda:0", model_name='resnet50'):
        print('-- Initializing LandmarksDetector')
        self.device = device
        print(f'-- Device: {self.device}')
        self.cfg = cfg_mnet
        print('-- Loading RetinaFace model')
        self.net = RetinaFace(cfg=self.cfg, phase = 'test')
        print('-- RetinaFace model loaded')
        
        model_path = os.path.join(os.getcwd(), 'Pytorch_Retinaface/weights/mobilenet0.25_Final.pth')
        
        self.net = load_model(self.net, model_path, self.device)
        self.net.eval()
        # The model is already on the correct device from load_model
        # self.net = self.net.to(self.device)
        print('-- LandmarksDetector initialized successfully')


    def __call__(self, filename):
        video_frames = torchvision.io.read_video(filename, pts_unit='sec')[0].numpy()
        landmarks = []
        for frame in video_frames:
            img = np.float32(frame)
            im_height, im_width, _ = img.shape
            scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            img -= (104, 117, 123)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.to(self.device)
            scale = scale.to(self.device)

            loc, conf, landms = self.net(img)

            priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
            priors = priorbox.forward()
            priors = priors.to(self.device)
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
            boxes = boxes * scale
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
            landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
            scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2]])
            scale1 = scale1.to(self.device)
            landms = landms * scale1
            landms = landms.cpu().numpy()

            # ignore low scores
            inds = np.where(scores > 0.02)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]

            # keep top-K before NMS
            order = scores.argsort()[::-1][:5000]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, 0.4)
            dets = dets[keep, :]
            landms = landms[keep]

            # Lower threshold for better webcam detection (0.1 for very sensitive detection)
            confidence_mask = dets[:, 4] >= 0.1
            dets = dets[confidence_mask]
            landms = landms[confidence_mask]
            
            # keep top-K faster NMS
            dets = dets[:750, :]
            landms = landms[:750, :]

            if len(dets) == 0:
                landmarks.append(None)
            else:
                # Get the largest face
                max_id, max_size = 0, 0
                for idx, bbox in enumerate(dets):
                    bbox_size = (bbox[2] - bbox[0]) + (bbox[3] - bbox[1])
                    if bbox_size > max_size:
                        max_id, max_size = idx, bbox_size
                
                # Reshape landmarks to be compatible
                face_landmarks = landms[max_id].reshape(-1, 2)
                landmarks.append(face_landmarks)
                
        return landmarks