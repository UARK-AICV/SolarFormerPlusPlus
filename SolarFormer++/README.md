# Custom Mask2Former

Original Repo: https://github.com/facebookresearch/Mask2Former. This repository has some customization 
developed on top of Mask2Former.
- Configurable the number of multiscale features that go in Transformer Decoder
- Support ResNet 18 and 34 configs
- ResNetv2 with H/2 x W/2 resolution feature map in the backbone output
- Support IDRID dataset (Medical Segmentation) with dataset registration, inference and evaluator.

## 1. Configurable number of feature layers for Transformer Decoder
```
cfg.MODEL.MASKFORMER.NUM_FEATURE_LEVELS = ... # 3 is default
```
This config is to set the number of feature levels that go in Transformer Decoder.
This variable is hardcoded as 3 in the original repo. (3 scales with the smallest resolution 
from the output of pixel decoder multiscale features)



## 2. Pixel Decoder v2: 
- Implementation: [`mask2former/modeling/pixel_decoder/msdeformattn.py`](https://github.com/trqminh/Mask2Former/blob/887181d77726e5915f898de50cd44788e494a9b8/mask2former/modeling/pixel_decoder/msdeformattn.py#L362)
- Configs:
```
MODEL:
  SEM_SEG_HEAD:
    PIXEL_DECODER_NAME: MSDeformAttnPixelDecoderv2
```
- This module is based on `MSDeformAttnPixelDecoder` but instead of outputting 4 features map `(H/4xW/4 -> H/32xW/32)`,
this outputs 6 features map `(HxW -> H/32xW/32)`, by using two `ConvTranspose2d` from the `H/4xW/4` feature map, the upsampling 
is implemented [here](https://github.com/trqminh/Mask2Former/blob/887181d77726e5915f898de50cd44788e494a9b8/mask2former/modeling/pixel_decoder/msdeformattn.py#L566)

## 3. SolarFormer++
Configuration for SolarFormer++ model, which utilizes the two developed customization above:
```
MODEL:
  SEM_SEG_HEAD:
    PIXEL_DECODER_NAME: "MSDeformAttnPixelDecoderv2"
    DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES: "['res2', 'res3','res4','res5']"
  MASK_FORMER:
    NUM_FEATURE_LEVELS: 4
```

### 4.1. register data:
- `mask2former/data/datasets/register_idrid_semseg.py`
- `mask2former/data/datasets/__init__.py`
- `configs/idrid/`
### 4.2. build evaluator 
- `mask2former/__init__.py`
- `mask2former/evaluation/idrid_semseg_evaluation.py`
- `train_net.py`
### 4.3. training and inference
- `train_net.py`
- `mask2former/maskformer_model.py`
- `mask2former/config.py`
