# Single Image Super-Resolution with GAN on HCI Data

Please see the [Examples](## Examples)

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

The **GAN-based** model produced sharper super-resolution images compared with **bicubic**. 

**Especially when the scaling factor is large**.

### up scale by 2

**low-res**   |  **bicubic** | **GAN** | **original**
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![bicubic](test_out_images/papillon/up_2/index_0_lr.jpg)  |  ![bicubic](test_out_images/papillon/up_2/index_0_bi.jpg) |  ![bicubic](test_out_images/papillon/up_2/index_0_sr.jpg)|  ![bicubic](test_out_images/papillon/up_2/index_0_hr.jpg)
![bicubic](test_out_images/statue/up_2/index_0_lr.jpg)  |  ![bicubic](test_out_images/statue/up_2/index_0_bi.jpg) |  ![bicubic](test_out_images/statue/up_2/index_0_sr.jpg)|  ![bicubic](test_out_images/statue/up_2/index_0_hr.jpg)
![bicubic](test_out_images/stillLife/up_2/index_0_lr.jpg)  |  ![bicubic](test_out_images/stillLife/up_2/index_0_bi.jpg) |  ![bicubic](test_out_images/stillLife/up_2/index_0_sr.jpg)|  ![bicubic](test_out_images/stillLife/up_2/index_0_hr.jpg)

### up scale by 4

**low-res**   |  **bicubic** | **GAN** | **original**
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![bicubic](test_out_images/papillon/up_4/index_0_lr.jpg)  |  ![bicubic](test_out_images/papillon/up_4/index_0_bi.jpg) |  ![bicubic](test_out_images/papillon/up_4/index_0_sr.jpg)|  ![bicubic](test_out_images/papillon/up_4/index_0_hr.jpg)
![bicubic](test_out_images/statue/up_4/index_0_lr.jpg)  |  ![bicubic](test_out_images/statue/up_4/index_0_bi.jpg) |  ![bicubic](test_out_images/statue/up_4/index_0_sr.jpg)|  ![bicubic](test_out_images/statue/up_4/index_0_hr.jpg)
![bicubic](test_out_images/stillLife/up_4/index_0_lr.jpg)  |  ![bicubic](test_out_images/stillLife/up_4/index_0_bi.jpg) |  ![bicubic](test_out_images/stillLife/up_4/index_0_sr.jpg)|  ![bicubic](test_out_images/stillLife/up_4/index_0_hr.jpg)

### up scale by 8

**low-res**   |  **bicubic** | **GAN** | **original**
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![bicubic](test_out_images/papillon/up_8/index_0_lr.jpg)  |  ![bicubic](test_out_images/papillon/up_8/index_0_bi.jpg) |  ![bicubic](test_out_images/papillon/up_8/index_0_sr.jpg)|  ![bicubic](test_out_images/papillon/up_8/index_0_hr.jpg)
![bicubic](test_out_images/statue/up_8/index_0_lr.jpg)  |  ![bicubic](test_out_images/statue/up_8/index_0_bi.jpg) |  ![bicubic](test_out_images/statue/up_8/index_0_sr.jpg)|  ![bicubic](test_out_images/statue/up_8/index_0_hr.jpg)
![bicubic](test_out_images/stillLife/up_8/index_0_lr.jpg)  |  ![bicubic](test_out_images/stillLife/up_8/index_0_bi.jpg) |  ![bicubic](test_out_images/stillLife/up_8/index_0_sr.jpg)|  ![bicubic](test_out_images/stillLife/up_8/index_0_hr.jpg)


##### details

**low-res**   |  **bicubic** | **GAN** | **original**
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![bicubic](test_out_images/statue/up_2/index_0_lr.jpg)  |  ![bicubic](test_out_images/statue/up_2/index_0_bi.jpg) |  ![bicubic](test_out_images/statue/up_2/index_0_sr.jpg)|  ![bicubic](test_out_images/statue/up_2/index_0_hr.jpg)
