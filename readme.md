# Texture Extraction using PWAN

This is the official code for Sec 5 of my PhD thesis. The code is used to extract texture from an image at a given position:


|Input with position| Extracted texture | Corresponding mask| 
|--------------|--------------|--------------|
<img src="readme_image\mask_image\Fish\fishold.png" width="256"/>  | <img src="readme_image\Partial\Fish\4990.png" width="256"/> |<img src="readme_image\mask_image\Fish\fishmask_old.png " width="256"/>

More examples can be found in folder `readme_image` and `Collect`.




## Usage
Train the model using:
`python texture_syn.py --style Texture1/blotchy_0118.jpg --mode part --Init_cor 144,224 --ratio 10.0 --lr_G 1e-4 --num_steps 5000`
More scripts can be found at `a.py`

## Requirement 
- Pytorch 

#### Note: This code is not cleaned, use it at your own risk. Non-commercial purpose only!


## Reference
If you find this code useful, please cite the paper

    @inproceedings{wang2022partial,
        title={Partial Wasserstein Adversarial Network for Non-rigid Point Set Registration},
        author={Zi-Ming Wang and Nan Xue and Ling Lei and Gui-Song Xia},
        booktitle={International Conference on Learning Representations (ICLR)},
        year={2022}
    }

For any question, please contact me (wzm2256@gmail.com).