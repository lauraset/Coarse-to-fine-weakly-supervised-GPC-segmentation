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


This part will be available within 7 days.


## Contributing


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


## Acknowledgments

