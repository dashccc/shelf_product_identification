---
title: "Os2d模型设置"
date: 2021-11-05T14:52:27+08:00
weight: 0300
draft: false
---

我们现在会用sagemaker进行一个Os2d的模型设置，使用ML.P3.2xlarge机型。
打开文件 `deploy_models/os2d/demo.ipynb`，逐行运行以实现测试。

## 环境配置

首先下载代码
```
cd SageMaker
git clone git://github.com/ultralytics/yolov5
cd yolov5
pip install -qr requirements.txt  # install dependencies

import torch
from IPython.display import Image, clear_output  # to display images

clear_output()
print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")
```
输出

        Setup complete. Using torch 1.9.1+cu102 (Tesla V100-SXM2-16GB)

即代表环境配置完成

## 模型以及数据准备


```
!aws s3 cp s3://lhr/bestxl.pt ./
!aws s3 cp s3://lhr/sku.zip ./os2d/ 
!aws s3 cp s3://lhr/os2d.yaml ./data/
%cd os2d 
!unzip sku.zip > /dev/null
%cd ..
!cp /home/ec2-user/SageMaker/xilian/yolov5_sagemaker/test.jpg ./
```


## 模型训练

接下来我们运行训练
```
!python train.py --img 640 --batch 16 --epochs 1 --data os2d.yaml --weights bestxl.pt --cache
```

这里使用我们已经训练好的模型作为训练起点，为了演示目的，我们在640分辨率的输入尺寸下运行一个epoch，大约需要5min

## 结果本地测试

我们可以直接用这个产生的模型文件进行本地推理。注意这里的模型文件地址的指定为你刚刚训练产生的。

```
!python detect.py --iou-thres 0.6 --conf-thres 0.4 --weights './runs/train/exp2/weights/best.pt' --img 640 --source './test.jpg'
Image(filename='runs/detect/exp/test.jpg', width=1000)
```

输出如下

![](../pics/yolo/res.png)

到这里，就完成了一个模型的训练过程。