# Language Embedded Gaussian Splats (LEGS)
<div align="center">

[[Website]](https://berkeleyautomation.github.io/LEGS/)
[[PDF]](https://autolab.berkeley.edu/assets/publications/media/2024_IROS_LEGS_CR.pdf)
[[Arxiv]](https://arxiv.org/abs/2409.18108)

[![Kitchen Queries](media/KitchenQueries.gif)](https://youtu.be/SubSWU1wJak)

[![Grocery Store Queries](media/GroceryStoreQueries.gif)](https://youtu.be/NA3m16Cgdm4)

</div>

This repository contains the code for the paper "Language-Embedded Gaussian Splats (LEGS): Incrementally Building Room-Scale Representations with a Mobile Robot".


# Installation
Language Embedded Gaussian Splats follows the integration guidelines described [here](https://docs.nerf.studio/developer_guides/new_methods.html) for custom methods within Nerfstudio.

To learn more about the code we use to interface with the robot and collect image poses, see this repo [here](https://github.com/BerkeleyAutomation/legs_ros_ws), which outlines our ROS2 interface.
### 0. Install Nerfstudio dependencies
[Follow these instructions](https://docs.nerf.studio/quickstart/installation.html) up to and including "tinycudann" to install dependencies.

 ***If you'll be using ROS messages do not use a conda environment and enter the dependency install commands below instead*** (ROS and conda don't play well together)

 ```
 pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
 pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
 ```
### 1. Clone and install repo
```
git clone https://github.com/BerkeleyAutomation/L3GS
cd L3GS/l3gs/
python -m pip install -e .
ns-install-cli
```

### Checking the install
Run `ns-train -h`: you should see a list of "subcommands" with lllegos and llgs included among them.


- Launch training with `ns-train l3gs` and start publishing an imagepose topic or playing an imagepose ROS bag. 
- Connect to the viewer by forwarding the viewer port (we use VSCode to do this), and click the link to `viewer.nerf.studio` provided in the output of the train script

## Bibtex
If you find LEGS useful for your work please consider citing:
```
@inproceedings{yu2024language,
        title={Language-embedded gaussian splats (legs): Incrementally building room-scale representations with a mobile robot},
        author={Yu, Justin and Hari, Kush and Srinivas, Kishore and El-Refai, Karim and Rashid, Adam and Kim, Chung Min and Kerr, Justin and Cheng, Richard and Irshad, Muhammad Zubair and Balakrishna, Ashwin and others},
        booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
        pages={13326--13332},
        year={2024},
        organization={IEEE}
      }
```
