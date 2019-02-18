# Single Image Super-Resolution with GAN on HCI Data

Please see [Examples](#Examples).

All results, including PSNR and SSIM, are available in the folder **test_out_images**.

This repo is a reimplementation of [this paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Ledig_Photo-Realistic_Single_Image_CVPR_2017_paper.pdf).

## For Cell Images

Please refer to [this link](https://zhaopku.github.io/sr.html) for similar experiments for cell images. 

The results for cell images are better. One explanation is that cell images are of 1-channel, and the structure of
cell images is simple.

## Dataset

**Train&Val**: [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)

**Test**: [HCI lightfield Dataset](http://lightfieldgroup.iwr.uni-heidelberg.de/?page_id=713)

## Requirements
    1. PyTorch 1.0
    2. tqdm

### Example Usage
    
    python main.py --model SRGAN --lr 0.001 --upscale_factor 4 --batch_size 100 --epochs 100 --n_save 2 --gamma 0.001 --theta 0.01 --sigma 0.001
        
The above command trains a GAN-based model, with upscale factor 4.
        
## Examples

Below are examples from **papillon**, **statue**, and **stillLife**. 

The **GAN-based** model produced sharper super-resolution images compared with **bicubic**. 

**Especially when the scaling factor is large**. However, when scaling factor is 8, the performance is not as good
as in [cell images](https://zhaopku.github.io/sr.html), but GAN is still better than bicubic.

### up scale by 2 (full image)

**low-res**   |  **bicubic** | **GAN** | **original**
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![low-res](test_out_images/papillon/up_2/index_0_lr.jpg)  |  ![bicubic](test_out_images/papillon/up_2/index_0_bi.jpg) |  ![GAN](test_out_images/papillon/up_2/index_0_sr.jpg)|  ![original](test_out_images/papillon/up_2/index_0_hr.jpg)
![low-res](test_out_images/statue/up_2/index_0_lr.jpg)  |  ![bicubic](test_out_images/statue/up_2/index_0_bi.jpg) |  ![GAN](test_out_images/statue/up_2/index_0_sr.jpg)|  ![original](test_out_images/statue/up_2/index_0_hr.jpg)
![low-res](test_out_images/stillLife/up_2/index_0_lr.jpg)  |  ![bicubic](test_out_images/stillLife/up_2/index_0_bi.jpg) |  ![GAN](test_out_images/stillLife/up_2/index_0_sr.jpg)|  ![original](test_out_images/stillLife/up_2/index_0_hr.jpg)

### up scale by 2 (details)

 **bicubic** | **GAN** | **original**
:-------------------------:|:-------------------------:|:-------------------------:
![bicubic](test_out_images/papillon/up_2/cr_index_0_bi.jpg) |  ![GAN](test_out_images/papillon/up_2/cr_index_0_sr.jpg)|  ![original](test_out_images/papillon/up_2/cr_index_0_hr.jpg)
![bicubic](test_out_images/statue/up_2/cr_index_0_bi.jpg) |  ![GAN](test_out_images/statue/up_2/cr_index_0_sr.jpg)|  ![original](test_out_images/statue/up_2/cr_index_0_hr.jpg)
![bicubic](test_out_images/stillLife/up_2/cr_index_0_bi.jpg) |  ![GAN](test_out_images/stillLife/up_2/cr_index_0_sr.jpg)|  ![original](test_out_images/stillLife/up_2/cr_index_0_hr.jpg)


### up scale by 4 (full image)

**low-res**   |  **bicubic** | **GAN** | **original**
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![low-res](test_out_images/papillon/up_4/index_0_lr.jpg)  |  ![bicubic](test_out_images/papillon/up_4/index_0_bi.jpg) |  ![GAN](test_out_images/papillon/up_4/index_0_sr.jpg)|  ![original](test_out_images/papillon/up_4/index_0_hr.jpg)
![low-res](test_out_images/statue/up_4/index_0_lr.jpg)  |  ![bicubic](test_out_images/statue/up_4/index_0_bi.jpg) |  ![GAN](test_out_images/statue/up_4/index_0_sr.jpg)|  ![original](test_out_images/statue/up_4/index_0_hr.jpg)
![low-res](test_out_images/stillLife/up_4/index_0_lr.jpg)  |  ![bicubic](test_out_images/stillLife/up_4/index_0_bi.jpg) |  ![GAN](test_out_images/stillLife/up_4/index_0_sr.jpg)|  ![original](test_out_images/stillLife/up_4/index_0_hr.jpg)

### up scale by 4 (details)


 **bicubic** | **GAN** | **original**
:-------------------------:|:-------------------------:|:-------------------------:
![bicubic](test_out_images/papillon/up_4/cr_index_0_bi.jpg) |  ![GAN](test_out_images/papillon/up_4/cr_index_0_sr.jpg)|  ![original](test_out_images/papillon/up_4/cr_index_0_hr.jpg)
![bicubic](test_out_images/statue/up_4/cr_index_0_bi.jpg) |  ![GAN](test_out_images/statue/up_4/cr_index_0_sr.jpg)|  ![original](test_out_images/statue/up_4/cr_index_0_hr.jpg)
![bicubic](test_out_images/stillLife/up_4/cr_index_0_bi.jpg) |  ![GAN](test_out_images/stillLife/up_4/cr_index_0_sr.jpg)|  ![original](test_out_images/stillLife/up_4/cr_index_0_hr.jpg)

### up scale by 8 (full image)

**low-res**   |  **bicubic** | **GAN** | **original**
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![low-res](test_out_images/papillon/up_8/index_0_lr.jpg)  |  ![bicubic](test_out_images/papillon/up_8/index_0_bi.jpg) |  ![GAN](test_out_images/papillon/up_8/index_0_sr.jpg)|  ![original](test_out_images/papillon/up_8/index_0_hr.jpg)
![low-res](test_out_images/statue/up_8/index_0_lr.jpg)  |  ![bicubic](test_out_images/statue/up_8/index_0_bi.jpg) |  ![GAN](test_out_images/statue/up_8/index_0_sr.jpg)|  ![original](test_out_images/statue/up_8/index_0_hr.jpg)
![low-res](test_out_images/stillLife/up_8/index_0_lr.jpg)  |  ![bicubic](test_out_images/stillLife/up_8/index_0_bi.jpg) |  ![GAN](test_out_images/stillLife/up_8/index_0_sr.jpg)|  ![original](test_out_images/stillLife/up_8/index_0_hr.jpg)

### up scale by 8 (details)

 **bicubic** | **GAN** | **original**
:-------------------------:|:-------------------------:|:-------------------------:
![bicubic](test_out_images/papillon/up_8/cr_index_0_bi.jpg) |  ![GAN](test_out_images/papillon/up_8/cr_index_0_sr.jpg)|  ![original](test_out_images/papillon/up_8/cr_index_0_hr.jpg)
![bicubic](test_out_images/statue/up_8/cr_index_0_bi.jpg) |  ![GAN](test_out_images/statue/up_8/cr_index_0_sr.jpg)|  ![original](test_out_images/statue/up_8/cr_index_0_hr.jpg)
![bicubic](test_out_images/stillLife/up_8/cr_index_0_bi.jpg) |  ![GAN](test_out_images/stillLife/up_8/cr_index_0_sr.jpg)|  ![original](test_out_images/stillLife/up_8/cr_index_0_hr.jpg)

[Examples of Cell Images](https://zhaopku.github.io/sr.html)
