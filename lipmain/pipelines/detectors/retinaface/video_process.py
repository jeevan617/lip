#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os
import cv2
import numpy as np
from skimage import transform as tf


def linear_interpolate(landmarks, start_idx, stop_idx):
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx-start_idx):
        landmarks[start_idx+idx] = start_landmarks + idx/float(stop_idx-start_idx) * delta
    return landmarks


def warp_img(src, dst, img, std_size):
    tform = tf.estimate_transform('similarity', src, dst)
    warped = tf.warp(img, inverse_map=tform.inverse, output_shape=std_size)
    warped = (warped * 255).astype('uint8')
    return warped, tform


def apply_transform(transform, img, std_size):
    warped = tf.warp(img, inverse_map=transform.inverse, output_shape=std_size)
    warped = (warped * 255).astype('uint8')
    return warped


def cut_patch(img, landmarks, height, width, threshold=5):
    center_x, center_y = np.mean(landmarks, axis=0)
    # Check for too much bias in height and width
    if abs(center_y - img.shape[0] / 2) > height + threshold:
        raise Exception('too much bias in height')
    if abs(center_x - img.shape[1] / 2) > width + threshold:
        raise Exception('too much bias in width')
    # Calculate bounding box coordinates
    y_min = int(round(np.clip(center_y - height, 0, img.shape[0])))
    y_max = int(round(np.clip(center_y + height, 0, img.shape[0])))
    x_min = int(round(np.clip(center_x - width, 0, img.shape[1])))
    x_max = int(round(np.clip(center_x + width, 0, img.shape[1])))
    # Cut the image
    cutted_img = np.copy(img[y_min:y_max, x_min:x_max])
    return cutted_img


class VideoProcess:
    def __init__(self, mean_face_path="20words_mean_face.npy", crop_width=96, crop_height=96,
                 start_idx=3, stop_idx=5, window_margin=12, convert_gray=True):
        self.reference = np.load(os.path.join(os.path.dirname(__file__), mean_face_path))
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.start_idx = start_idx
        self.stop_idx = stop_idx
        self.window_margin = window_margin
        self.convert_gray = convert_gray

    def __call__(self, video, landmarks):
        # If no landmarks (face_track=False), process raw video
        if landmarks is None:
            import cv2
            # Convert to grayscale and center crop
            processed_frames = []
            for frame in video:
                # Convert to grayscale
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                else:
                    gray = frame
                
                # Center crop to 96x96 (same as crop_width/crop_height)
                h, w = gray.shape
                crop_h, crop_w = self.crop_height, self.crop_width
                start_h = (h - crop_h) // 2
                start_w = (w - crop_w) // 2
                cropped = gray[start_h:start_h+crop_h, start_w:start_w+crop_w]
                processed_frames.append(cropped)
            
            result = np.array(processed_frames)
            print(f"DEBUG: Fallback result shape: {result.shape}")
            return result
        
        # Check if landmarks list length matches video frames
        # If not, face detection failed on many frames - fall back to raw processing
        if len(landmarks) != len(video):
            print(f"⚠️  Landmarks mismatch: {len(landmarks)} landmarks for {len(video)} frames")
            print(f"⚠️  Face detection failed on many frames, using fallback processing")
            # Process without landmarks
            return self.__call__(video, None)
            
        # Pre-process landmarks: interpolate frames that are not detected
        preprocessed_landmarks = self.interpolate_landmarks(landmarks)
        # Exclude corner cases: no landmark in all frames or number of frames is less than window length
        if not preprocessed_landmarks or len(preprocessed_landmarks) < self.window_margin:
            print(f"⚠️  Not enough valid landmarks, using fallback processing")
            return self.__call__(video, None)
        # Affine transformation and crop patch
        sequence = self.crop_patch(video, preprocessed_landmarks)
        assert sequence is not None, f"cannot crop a patch."
        return sequence


    def crop_patch(self, video, landmarks):
        sequence = []
        for frame_idx, frame in enumerate(video):
            window_margin = min(self.window_margin // 2, frame_idx, len(landmarks) - 1 - frame_idx)
            smoothed_landmarks = np.mean([landmarks[x] for x in range(frame_idx - window_margin, frame_idx + window_margin + 1)], axis=0)
            smoothed_landmarks += landmarks[frame_idx].mean(axis=0) - smoothed_landmarks.mean(axis=0)
            transformed_frame, transformed_landmarks = self.affine_transform(frame,smoothed_landmarks,self.reference,grayscale=self.convert_gray)
            patch = cut_patch(transformed_frame, transformed_landmarks[self.start_idx:self.stop_idx], self.crop_height//2, self.crop_width//2,)
            sequence.append(patch)
        return np.array(sequence)


    def interpolate_landmarks(self, landmarks):
        valid_frames_idx = [idx for idx, lm in enumerate(landmarks) if lm is not None]

        if not valid_frames_idx:
            return None

        for idx in range(1, len(valid_frames_idx)):
            if valid_frames_idx[idx] - valid_frames_idx[idx - 1] > 1:
                landmarks = linear_interpolate(landmarks, valid_frames_idx[idx - 1], valid_frames_idx[idx])

        valid_frames_idx = [idx for idx, lm in enumerate(landmarks) if lm is not None]

        # Handle corner case: keep frames at the beginning or at the end that failed to be detected
        if valid_frames_idx:
            landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
            landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])

        assert all(lm is not None for lm in landmarks), "not every frame has landmark"

        return landmarks


    def affine_transform(self, frame, landmarks, reference, grayscale=True,
                         target_size=(256, 256), reference_size=(256, 256), stable_points=(0, 1, 2, 3, 4),
                         interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, border_value=0):
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        stable_reference = self.get_stable_reference(reference, stable_points, reference_size, target_size)
        transform = self.estimate_affine_transform(landmarks, stable_points, stable_reference)
        transformed_frame, transformed_landmarks = self.apply_affine_transform(frame, landmarks, transform, target_size, interpolation, border_mode, border_value)

        return transformed_frame, transformed_landmarks


    def get_stable_reference(self, reference, stable_points, reference_size, target_size):
        # Map 68-point reference to 5-point RetinaFace format
        # 0: Left Eye (36-41 mean)
        # 1: Right Eye (42-47 mean)
        # 2: Nose Tip (33)
        # 3: Left Mouth Corner (48)
        # 4: Right Mouth Corner (54)
        
        stable_reference = np.vstack([
            np.mean(reference[36:42], axis=0), # Left Eye
            np.mean(reference[42:48], axis=0), # Right Eye
            reference[33],                     # Nose Tip
            reference[48],                     # Left Mouth Corner
            reference[54]                      # Right Mouth Corner
        ])
        
        stable_reference[:, 0] -= (reference_size[0] - target_size[0]) / 2.0
        stable_reference[:, 1] -= (reference_size[1] - target_size[1]) / 2.0
        return stable_reference


    def estimate_affine_transform(self, landmarks, stable_points, stable_reference):
        return cv2.estimateAffinePartial2D(np.vstack([landmarks[x] for x in stable_points]), stable_reference, method=cv2.LMEDS)[0]


    def apply_affine_transform(self, frame, landmarks, transform, target_size, interpolation, border_mode, border_value):
        transformed_frame = cv2.warpAffine(frame, transform, dsize=(target_size[0], target_size[1]),
                                           flags=interpolation, borderMode=border_mode, borderValue=border_value)
        transformed_landmarks = np.matmul(landmarks, transform[:, :2].transpose()) + transform[:, 2].transpose()
        return transformed_frame, transformed_landmarks
