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



graph TD
    %% --- 定义样式 (模仿样图) ---
    classDef feature fill:#D0E1F5,stroke:#4A86E8,stroke-width:2px,rx:5,ry:5,color:#000000;
    classDef operation fill:#D5E8D4,stroke:#82B366,stroke-width:2px,rx:5,ry:5,color:#000000;
    classDef sumNode fill:#F5F5F5,stroke:#666666,stroke-width:2px,circle,font-size:20px;
    classDef groupStyle fill:none,stroke:#AAAAAA,stroke-width:2px,stroke-dasharray: 5 5;
    linkStyle default stroke:#555555,stroke-width:2px,fill:none;

    %% --- 主体结构 ---

    %% 1. 输入
    Input_F[输入特征 Input F_in<br>H x W x C_in]:::feature
    
    Input_F --> SplitNode( )

    %% 2. 左侧分支：解耦分支 (Decoupling Branch)
    subgraph LeftBranch [解耦分支 Decoupling Branch - 全局信息]
        direction TB
        
        SplitNode -.-> ProjOp
        
        %% 操作步骤
        ProjOp[投影 Projection<br>(固定基 B^T)]:::operation
        ProjOp --"系数 C"--> WeightOp[加权 Weighting<br>(可学习参数 W_D)]:::operation
        WeightOp --"变换系数 C'"--> ReconOp[重建 Reconstruction<br>(固定基 B)]:::operation
        
        %% 分支输出
        ReconOp --> F_dec[解耦特征 F_dec<br>H x W x C_out]:::feature
        
        %% 注释：数学等价性
        NoteNode[注: 数学上等效于<br>超大核卷积 (Large Kernel Conv)]-->F_dec
        style NoteNode fill:none,stroke:none,color:#666666,font-style:italic,font-size:12px
    end

    %% 3. 右侧分支：聚合分支 (Aggregating Branch)
    subgraph RightBranch [聚合分支 Aggregating Branch - 局部细节]
        direction TB
        
        SplitNode -.-> LocalConvOp
        
        %% 操作步骤
        LocalConvOp[局部卷积 Local Conv<br>k=3x3, s=1, p=1<br>(可学习参数 W_A)]:::operation
        
        %% 分支输出
        LocalConvOp --> F_agg[聚合特征 F_agg<br>H x W x C_out]:::feature
    end

    %% 4. 融合与输出
    F_dec --> Sum((+)):::sumNode
    F_agg --> Sum
    
    Sum --> Output_F[输出特征 Output F_out<br>H x W x C_out]:::feature

    %% --- 应用分组样式 ---
    class LeftBranch,RightBranch groupStyle;

    %% --- 调整布局间距 ---
    linkStyle 0,4,8 stroke-width:2px,fill:none,stroke:#555555;

Our quantitative evaluation is based on the metrics implementation from:
- **Code Source**: [Linfeng-Tang/Image-Fusion](https://github.com/Linfeng-Tang/Image-Fusion)
- **Metrics Used**: EN, SF，SD, MI,VIF，AG

We thank the authors for making their evaluation code publicly available.
