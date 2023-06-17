# Self-supervised Non-uniform Kernel Estimation with Flow-based Motion Prior for Blind Image Deblurring (CVPR 2023)
Zhenxuan Fang, Fangfang Wu, Weisheng Dong, Xin Li, Jinjian Wu and Guangming Shi

[**[Paper]**](https://openaccess.thecvf.com/content/CVPR2023/papers/Fang_Self-Supervised_Non-Uniform_Kernel_Estimation_With_Flow-Based_Motion_Prior_for_Blind_CVPR_2023_paper.pdf)
[**[Website Page]**](https://see.xidian.edu.cn/faculty/wsdong/Projects/UFPNet.htm.)



## Architecture
<p align="center">
<img src="/illustrations/network.png" width="1200">
</p>
Fig. 1 The overall framework of the proposed KULNet for blind SR.

## Usage
This implementation based on [BasicSR](https://github.com/xinntao/BasicSR) and [NAFNet](https://github.com/megvii-research/NAFNet)
### Download the repository
1. Requirements 
``Python 3.7 and PyTorch 1.8.0.``
2. Download this repository via git
```
git clone https://github.com/Fangzhenxuan/UFPDeblur
```
or download the [zip file](https://github.com/Fangzhenxuan/UFPDeblur/archive/main.zip) manually.


### Quick Start
Download the pretrained checkpoints ([Google Drive](https://drive.google.com/drive/folders/1ZVKMz7JHv3FlocoizgyBZOzL5LOXLMYB?usp=drive_link)), the directory structure will be arranged as:
```
experiments
    |- pretrained_models
        |- train_on_GoPro
        |- train_on_RealBlurJ
        |- train_on_RealBlurR
```
Put the test datasets in dir ``./datasets/``
```
datasets
    |- GoPro
        |- test
            |- target
            |- input
    |- ...
        
```

* Test on GoPro testset, run
    ```
    python ./basicsr/test.py -opt options/test/GoPro/UFPNet-GoPro.yml 
    ```
* Test on RealBlur-J testset
    * To use the model trained on GoPro, run
        ```
        python ./basicsr/test.py -opt options/test/RealBlur-J/UFPNet-RealBlurJ-Train-on-GoPro.yml 
        ```
    * To use the model trained on RealBlur-J, run
        ```
        python ./basicsr/test.py -opt options/test/RealBlur-J/UFPNet-RealBlurJ.yml  
        ```


### Citations
If UFPNet helps your research or work, please consider citing UFPNet.
```
@inproceedings{fang2023self,
  title={Self-supervised Non-uniform Kernel Estimation with Flow-based Motion Prior for Blind Image Deblurring},
  author={Fang, Zhenxuan and Wu, Fangfang and Dong, Weisheng and Li, Xin and Wu, Jinjian and Shi, Guangming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={18105--18114},
  year={2023}
}
```

## Acknowledgements
The codes are built on [NAFNet](https://github.com/megvii-research/NAFNet) [1]. We thank the authors for sharing their codes.

## References
[1] Liangyu Chen, et al. "Simple Baselines for Image Restoration." In European Conference on Computer Vision 2022.


