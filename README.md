# LEDFusion
The project is being continuously updated

## Recommended Environment
This is our environment: Ubuntu22.04:
- Note: We encourage you to try other environments:
 - [ ] torch  1.9.0
 - [ ] cuda   11.1
 - [ ] cudnn 8
 - [ ] kornia 
 - [ ] piq
 - [ ] torchsummary
 - [ ] tqdm
 - [ ] pillow

## Notice
- This program is trained by 3090Ti GPU
## Test
`python test.py`

## Train
`python train.py`

## Pre-trained Weights

Download pre-trained model weights: [Baidu Netdisk](https://pan.baidu.com/s/14JrEGoMWjUQ-e8m5LxVGWQ)  
Extraction code: `3dak`

## Data 
The dataset is organized in a hierarchical directory structure as follows:

```
LEDFusion/
├── train/
│   ├── vi/         # Training Visible Images
│   └── ir/         # Training Infrared Images
├── test/
│   └── LLVIP/      # Example Dataset Name
│       ├── vi/     # Testing Visible Images
│       └── ir/     # Testing Infrared Images
├── eval/           # Evaluation Data (Optional)
   ├── ir/
   └── vi_en           # Dependency folder for Enhancement Net
```

# You can Download such datasets as follows:
### LLVIP:[download](https://github.com/bupt-ai-cz/LLVIP)
### MSRS:[download](https://pan.baidu.com/s/1TVup567GLe0fke13C4-ntQ?pwd=xhmq)
### M3FD:[download](https://pan.baidu.com/s/1sq-rksPEU5SjUebLNlBlMQ?pwd=4wqu)
### TNO:[download](https://pan.baidu.com/s/1dWX43SolSXv9-H_FIVhS_g?pwd=ea6g)


## Evaluation

### Quantitative Metrics


Our quantitative evaluation is based on the metrics implementation from:
- **Code Source**: [Linfeng-Tang/Image-Fusion](https://github.com/Linfeng-Tang/Image-Fusion)
- **Metrics Used**: EN, SF，SD, MI,VIF，AG

We thank the authors for making their evaluation code publicly available.
