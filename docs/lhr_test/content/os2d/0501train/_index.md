---
title: "Os2d模型测试"
date: 2021-11-05T14:52:27+08:00
weight: 0300
draft: false
---

我们现在会用sagemaker进行一个Os2d的模型设置，使用ML.P3.2xlarge机型。
打开文件 `deploy_models/os2d/demo.ipynb`，逐行运行以实现测试。

## 环境配置

使用SageMaker原生的pytorch_latest_p37即可，无需额外配置环境。

## 模型以及数据准备

```
%cd ./deploy_models/os2d
!aws s3 cp s3://lhr/os2d_v2-train.pth ./models/
!aws s3 cp s3://lhr/classes.zip ./data/demo/
!aws s3 cp s3://lhr/5.jpg ./data/demo/
%cd ./data/demo
!unzip classes.zip > /dev/null
%cd ..
%cd ..
```


## 模型测试

接下来我们进行测试，顺着`demo.ipynb`往下进行即可，整个逻辑即为使用`classes`文件夹内的class图像在`5.jpg`内进行特征匹配并实现目标检测，最终的测试结果如下：

![](../pics/os2d/res.png)

到这里，就完成了Os2d的测试过程。