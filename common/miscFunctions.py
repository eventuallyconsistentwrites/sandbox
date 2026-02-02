import cv2
import glob
import os
import re
from tqdm import tqdm

def create_video_from_dir(directory_path, output_name="output_video.mp4", fps=25):
    target_size = (1280, 720)
    
    # 1. Get all images (adjust extensions as needed)
    images = glob.glob(os.path.join(directory_path, "*.png"))
    
    # 2. Sort files numerically (so frame10 comes after frame2)
    images.sort(key=lambda f: int(re.sub(r'\D', '', f) or 0))

    if not images:
        print("No images found in the directory.")
        return

    # 3. Setup Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = os.path.join(directory_path, output_name)
    out = cv2.VideoWriter(video_path, fourcc, fps, target_size)


    for filename in tqdm(images, desc=f"Processing {len(images)} frames..."):
        img = cv2.imread(filename)
        if img is not None:
            # Standardize size
            img = cv2.resize(img, target_size)
            out.write(img)
        else:
            print(f"Warning: Could not read {filename}")

    out.release()
    print(f"Video saved successfully: {video_path}")