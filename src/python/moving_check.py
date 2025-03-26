import cv2
import numpy as np
import torch
import os
import argparse
import sys
from fastsam import FastSAM, FastSAMPrompt
from ultralytics import YOLO

class DynamicDetector:
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.lk_params = dict(winSize=(15, 15),
                             maxLevel=2,
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.prev_gray = None
        self.prev_kps = None
        self.fastsam = FastSAM('FastSAM-x.pt')
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def _motion_consistency_check(self, curr_gray):
        if self.prev_gray is None:
            return []

        # LK光流跟踪
        curr_kps, st, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, curr_gray, 
            self.prev_kps, None, **self.lk_params)

        # 筛选有效跟踪点
        valid_prev = self.prev_kps[st == 1]
        valid_curr = curr_kps[st == 1]

        # 计算运动向量
        motion_vectors = valid_curr - valid_prev
        motion_magnitude = np.linalg.norm(motion_vectors, axis=1)

        # 动态点筛选
        dynamic_indices = np.where(motion_magnitude > 2.0)[0]
        return valid_curr[dynamic_indices].reshape(-1, 2)

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kps = self.orb.detect(gray, None)
        
        if len(kps) > 0:
            curr_pts = np.array([kp.pt for kp in kps], dtype=np.float32)
            dynamic_pts = self._motion_consistency_check(gray)
            
            self.prev_gray = gray.copy()
            self.prev_kps = curr_pts
            
            return dynamic_pts
        return []

def generate_mask(source, maskpath):
    detector = DynamicDetector()
    
    for frame_path in sorted(os.listdir(source)):
        if not frame_path.endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        frame = cv2.imread(os.path.join(source, frame_path))
        if frame is None:
            continue
            
        # 检测动态点
        dynamic_points = detector.process_frame(frame)
        
        if len(dynamic_points) == 0:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        else:
            # 转换点坐标格式
            points = dynamic_points[:, ::-1].cpu().numpy()  # 转换为(y,x)格式
            prompt_process = FastSAMPrompt(frame, detector.fastsam.device)
            ann = prompt_process.point_prompt(points=points, pointlabel=[1]*len(points))
            mask = ann[0].astype(np.uint8) * 255
        
        output_path = os.path.join(maskpath, os.path.basename(frame_path))
        cv2.imwrite(output_path, mask)
        print(f"Generated mask: {output_path}")

def remove_dyna(rgb_folder, maskpath, output_folder):
    # 保持原remove_dyna函数不变
    for filename in os.listdir(rgb_folder):
        if filename.endswith(".png"):
            rgb_path = os.path.join(rgb_folder, filename)
            mask_path = os.path.join(maskpath, filename)

            if os.path.exists(mask_path):
                rgb_image = cv2.imread(rgb_path)
                mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask_image = cv2.bitwise_not(mask_image)
                result_image = cv2.bitwise_and(rgb_image, rgb_image, mask=mask_image)
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, result_image)
                print(f"Processed: {filename}")
            else:
                print(f"Mask missing: {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="rgb")
    parser.add_argument("--maskpath", type=str, default="mask")
    parser.add_argument("--removedpath", type=str, default="removed")
    args = parser.parse_args()

    print(f"Masks will be saved to {args.maskpath}/")
    print(f"Cleaned frames will be saved to {args.removedpath}/")
    
    confirm = input("Confirm processing? (y/n): ")
    if confirm.lower() not in ['y', 'yes']:
        sys.exit("Operation cancelled")
        
    os.makedirs(args.maskpath, exist_ok=True)
    os.makedirs(args.removedpath, exist_ok=True)
    
    generate_mask(args.source, args.maskpath)
    remove_dyna(args.source, args.maskpath, args.removedpath)