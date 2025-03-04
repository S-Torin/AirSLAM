## :eyes: Updates
* [2025.03] Fork from AirSLAM `https://github.com/sair-lab/AirSLAM.git`
* [2025.03] Modify CMakeLists.txt


## :checkered_flag: Test Environment
### Dependencies
* OpenCV 4.2
* Eigen 3
* Ceres 2.0.0
* G2O (tag:20230223_git)
* TensorRT 8.6.1.6
* CUDA 12.1
* python
* ROS noetic
* Boost

## :book: Data
The data for mapping should be organized in the following Autonomous Systems Lab (ASL) dataset format (imu data is optional):

```
dataroot
├── cam0
│   └── data
│       ├── t0.jpg
│       ├── t1.jpg
│       ├── t2.jpg
│       └── ......
├── cam1
│   └── data
│       ├── t0.jpg
│       ├── t1.jpg
│       ├── t2.jpg
│       └── ......
└── imu0
    └── data.csv

```
After the map is built, the relocalization requires only monocular images. Therefore, you only need to place the query images in a folder.


## :computer: Build
```
    cd ~/catkin_ws/src
    git clone https://github.com/S-Torin/AirSLAM.git
    cd ../
    catkin_make
    source ~/catkin_ws/devel/setup.bash
```

## :running: Run

The launch files for VO/VIO, map optimization, and relocalization are placed in [VO folder](launch/visual_odometry), [MR folder](launch/map_refinement), and [Reloc folder](launch/relocalization), respectively. Before running them, you need to modify the corresponding configurations according to your data path and the desired map-saving path. The following is an example of mapping, optimization, and relocalization with the EuRoC dataset.


### Mapping
**1**: Change "dataroot" in [VO launch file](launch/visual_odometry/vo_euroc.launch) to your own data path. For the EuRoC dataset, "mav0" needs to be included in the path.

**2**: Change "saving_dir" in the same file to the path where you want to save the map and trajectory. **It must be an existing folder.**

**3**: Run the launch file:

```
roslaunch air_slam vo_euroc.launch
```

### Map Optimization
**1**: Change "map_root" in [MR launch file](launch/map_refinement/mr_euroc.launch) to your own map path.

**2**: Run the launch file:

```
roslaunch air_slam mr_euroc.launch
```

### Relocalization
**1**: Change "dataroot" in [Reloc launch file](launch/relocalization/reloc_euroc.launch) to your own query data path.

**2**: Change "map_root" in the same file to your own map path.

**3**: Run the launch file:

```
roslaunch air_slam reloc_euroc.launch
```

### Other datasets
[Launch folder](launch) and [config folder](configs) respectively provide the launch files and configuration files for other datasets in the paper. If you want to run AirSLAM with your own dataset, you need to create your own camera file, configuration file, and launch file.


## :pencil: Citation
```bibtex
@article{xu2024airslam,
  title = {{AirSLAM}: An Efficient and Illumination-Robust Point-Line Visual SLAM System},
  author = {Xu, Kuan and Hao, Yuefan and Yuan, Shenghai and Wang, Chen and Xie, Lihua},
  journal = {IEEE Transactions on Robotics (TRO)},
  year = {2024},
  url = {https://arxiv.org/abs/2408.03520},
  code = {https://github.com/sair-lab/AirSLAM},
}

@inproceedings{xu2023airvo,
  title = {{AirVO}: An Illumination-Robust Point-Line Visual Odometry},
  author = {Xu, Kuan and Hao, Yuefan and Yuan, Shenghai and Wang, Chen and Xie, Lihua},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year = {2023},
  url = {https://arxiv.org/abs/2212.07595},
  code = {https://github.com/sair-lab/AirVO},
  video = {https://youtu.be/YfOCLll_PfU},
}
```
