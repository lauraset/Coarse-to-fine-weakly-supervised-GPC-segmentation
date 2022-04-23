# Coarse-to-fine-weakly-supervised-GPC-segmentation

Title: A coarse-to-fine weakly supervised learning method for green plastic cover segmentation using high-resolution remote sensing images  

Author: Yinxia Cao, Xin Huang  

Author affiliation: School of Remote Sensing and Information Engineering, Wuhan University, Wuhan 430079, PR China  

Journal: ISPRS Journal of Photogrammetry and Remote Sensing  

Paper state: accepted on 4.14, 2022  

## Getting Started

### Prerequisites

```
python >=3.6
pytorch >= 1.7.0 (lower version may be also work)
GPU: NVIDIA GTX 1080 Ti GPU (11G memory)
```
## Prepare datasets
Google earth images download steps:
1. Add a XYZ tile in the browser panel. 
For google earth images, a available link for XYZ connection is https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}.
2. Load the created link to QGIS, and open the python in QGIS. 
3. Run the following code
```
python qgis_download_google.py
```
![image](https://user-images.githubusercontent.com/39206462/147477579-ecdb5dc8-961a-47e6-ba8a-5b3ab30f38a4.png)
![image](https://user-images.githubusercontent.com/39206462/147477947-4489ce26-903d-4e04-a37e-b2a4d94881cf.png)

## Model development
The workflow of the proposed method is as follows:   
![image](https://user-images.githubusercontent.com/39206462/164873266-3a94972b-ecee-4b0e-8055-a8f3921d3148.png)

## One-by-one step
Step 1: training the classification network and generating grad-cam++ maps   
```
cd tttcls_google_gpc
python train_regent040_0.6_balance.py
python generate_cam_balance.py
```
Step 2: applying unsupervised segmentation to images   
refer to the [link](https://github.com/kanezaki/pytorch-unsupervised-segmentation)  
revise the input path and then run the following code. Note that two versions were provided.   
There is little difference between the two versions. We just used the v1 in the paper.
```
ptyhon generate_unsupervised_segmentation_v1.py
[or python generate_unsupervised_segmentation_v2.py]
```
Step 3: carrying out segmentation 
```
cd tttseg_google_gpc
python train_regnet040_0.6_update.py
```

## Acknowledgement
We used the package "segmentation_models_pytorch" and "pytorch-grad-cam".   
Thanks for their contributions.  

```
@misc{Yakubovskiy:2019,
  Author = {Pavel Yakubovskiy},
  Title = {Segmentation Models Pytorch},
  Year = {2020},
  Publisher = {GitHub},
  Journal = {GitHub repository},
  Howpublished = {\url{https://github.com/qubvel/segmentation_models.pytorch}}
}
@misc{jacobgilpytorchcam,
  title={PyTorch library for CAM methods},
  author={Jacob Gildenblat and contributors},
  year={2021},
  publisher={GitHub},
  howpublished={\url{https://github.com/jacobgil/pytorch-grad-cam}},
}
```


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

