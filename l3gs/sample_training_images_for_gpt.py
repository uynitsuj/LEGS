import os
import numpy as np
import shutil
import glob

which_bag = "full_kitchen_figure8_pose_bag"
images_dir = f"/home/lifelong/ros2_ws/src/py_pubsub/py_pubsub/{which_bag}_colmap/images/"
sampled_output_dir = f"/home/lifelong/L3GS/l3gs/sampled_images/{which_bag}_colmap/"

image_names = [name for name in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, name))]
sampled = np.random.choice(image_names, 30, replace=False)
print(sampled)

if os.path.isdir(sampled_output_dir):
    files = glob.glob(sampled_output_dir + '*')
    for f in files:
        os.remove(f)

if not os.path.isdir(sampled_output_dir):
    os.makedirs(sampled_output_dir)
for name in sampled:
    shutil.copy(images_dir + name, os.path.join(sampled_output_dir))