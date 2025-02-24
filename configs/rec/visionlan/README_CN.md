[English](README.md) | 中文

# VisionLAN

<!--- Guideline: use url linked to abstract in ArXiv instead of PDF for fast loading.  -->

> VisionLAN: [From Two to One: A New Scene Text Recognizer with Visual Language Modeling Network](https://arxiv.org/abs/2108.09661)

## 1. 简介

### 1.1 VisionLAN

视觉语言建模网络（VisionLAN）[<a href="#5-引用文献">1</a>]是一种文本识别模型，它通过在训练阶段使用逐字符遮挡的特征图来同时学习视觉和语言信息。这种模型不需要额外的语言模型来提取语言信息，因为视觉和语言信息可以作为一个整体来学习。
<p align="center">
  <img src="https://raw.githubusercontent.com/wtomin/mindocr-asset/main/images/visionlan_architecture.PNG" width=450 />
</p>
<p align="center">
  <em> 图 1. Visionlan 的模型结构 [<a href="#5-引用文献">1</a>] </em>
</p>

如上图所示，VisionLAN的训练流程由三个模块组成：

- 骨干网络从输入图像中提取视觉特征图；
- 掩码语言感知模块（MLM）以视觉特征图和一个随机选择的字符索引作为输入，并生成位置感知的字符掩码图，以创建逐字符遮挡的特征图；
- 最后，视觉推理模块（VRM）以遮挡的特征图作为输入，并在完整的单词级别的监督下进行预测。

但在测试阶段，MLM不被使用。只有骨干网络和VRM被用于预测。

## 2.精度结果

根据我们实验结果，在10个公开数据集上的评估结果如下：

<div align="center">

| **Model** | **Context** | **Backbone**|  **Train Dataset** | **Model Params**|**Avg Accuracy** | **Train Time** | **Per Step Time** | **FPS** | **Recipe** | **Download** |
| :-----: | :-----------: | :--------------: | :----------: | :--------: | :--------: |:----------: |:--------: | :--------: |:--------: |:----------: |
| visionlan  | D910x4-MS2.0-G | resnet45 | MJ+ST| 42.2M | 90.61%  |  7718s/epoch  | 417 ms/step  | 1,840 img/s | [yaml(LF_1)](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/visionlan/visionlan_resnet45_LF_1.yaml) [yaml(LF_2)](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/visionlan/visionlan_resnet45_LF_2.yaml) [yaml(LA)](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/visionlan/visionlan_resnet45_LA.yaml)| [ckpt files](https://download.mindspore.cn/toolkits/mindocr/visionlan/visionlan_resnet45_ckpts-7d6e9c04.tar.gz) \| [mindir(LA)](https://download.mindspore.cn/toolkits/mindocr/visionlan/visionlan_resnet45_LA-e9720d9e-71b38d2d.mindir)|

</div>

<details open markdown>
  <div align="center">
  <summary>Detailed accuracy results for ten benchmark datasets</summary>

  | **Model** |  **Context** | **IC03_860**| **IC03_867**| **IC13_857**|**IC13_1015** |  **IC15_1811** |**IC15_2077** | **IIIT5k_3000** |  **SVT** | **SVTP** | **CUTE80** | **Average** |
  | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: | :------: |:------: |
  | visionlan | D910x4-MS2.0-G | 96.16% | 95.16%  |  95.92%|   94.19%  | 84.04%  | 77.46%  | 95.53%  | 92.27%  | 85.74%  |89.58% | 90.61%  |

  </div>

</details>

**注**

- 训练环境表示为`{device}x{pieces}-{MS版本}-{MS模式}`。MindSpore模式可以是`G`（Graph模式）或`F`（Pynative模式）。例如，`D910x4-MS2.0-G`表示使用MindSpore版本2.0.0在4块910 NPUs上使用图模式进行训练。
- 训练数据集：`MJ+ST`代表两个合成数据集SynthText（800k）和MJSynth的组合。
- 要在其他训练环境中重现结果，请确保全局批量大小相同。
- 这些模型是从头开始训练的，没有任何预训练。有关训练和评估的更多数据集详细信息，请参阅[3.2数据集准备](#32数据集准备)部分。
- VisionLAN的MindIR导出时的输入Shape均为(1, 3, 64, 256)。

## 3.快速入门

### 3.1安装

请参考[MindOCR中的安装说明](https://github.com/mindspore-lab/mindocr#installation)。

### 3.2数据集准备

**训练集**

VisionLAN的作者使用了两个合成文本数据集进行训练：SynthText（800k）和MJSynth。请按照[原始VisionLAN repository](https://github.com/wangyuxin87/VisionLAN)的说明进行操作，下载这两个LMDB数据集。

下载`SynthText.zip`和`MJSynth.zip`后，请解压缩并将它们放置在`./datasets/train`目录下。训练集数据总共包括 14,200,701 个样本。更多关于训练集的信息如下:

- [SynText](https://academictorrents.com/details/2dba9518166cbd141534cbf381aa3e99a087e83c): 25GB, 6,976,115 samples
- [MJSynth](http://www.robots.ox.ac.uk/~vgg/data/text/): 21GB, 7,224,586 samples



**验证集**

VisionLAN的作者使用了六个真实文本数据集进行评估：IIIT5K Words（IIIT5K_3000）、ICDAR 2013（IC13_857）、Street View Text（SVT）、ICDAR 2015（IC15_1811）、Street View Text-Perspective（SVTP）、CUTE80（CUTE）。

请按照[原始VisionLAN repository](https://github.com/wangyuxin87/VisionLAN)的说明进行操作，下载验证数据集。

下载`evaluation.zip`后，请解压缩并将其放置在`./datasets`目录下。在 `./datasets/evaluation`路径下，一共有7个文件夹:

- [IIIT5K](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html): 50M, 3000 samples<br>
- [IC13](http://rrc.cvc.uab.es/?ch=2): 72M, 857 samples<br>
- [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset): 2.4M, 647 samples<br>
- [IC15](http://rrc.cvc.uab.es/?ch=4): 21M, 1811 samples<br>
- [SVTP](http://openaccess.thecvf.com/content_iccv_2013/papers/Phan_Recognizing_Text_with_2013_ICCV_paper.pdf): 1.8M, 645 samples<br>
- [CUTE](http://cs-chan.com/downloads_CUTE80_dataset.html): 8.8M, 288 samples<br>
- Sumof6benchmarks: 155M, 7248 samples


训练中，我们只用到了 `./datasets/evaluation/Sumof6benchmarks` 作为验证集。 用户可以选择将 `./datasets/evaluation` 下其他无关的文件夹删除。

**测试集**

我们选择用10个公开数据集来测试模型精度。用户可以从[这里](https://www.dropbox.com/sh/i39abvnefllx2si/AAAbAYRvxzRp3cIE5HzqUw3ra?dl=0) (ref: [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark#download-lmdb-dataset-for-traininig-and-evaluation-from-here))下载测试集。测试中我们只需要用到 `evaluation.zip`。

在下载 `evaluation.zip`完成后, 请解压缩, 并把文件夹名称从 `evaluation` 改为 `test`。请把这个`test`文件夹放在 `./datasets/`路径下

测试集总共包含 12,067 个样本。详细信息如下：

- [CUTE80](http://cs-chan.com/downloads_CUTE80_dataset.html): 8.8 MB, 288 samples<br>
- [IC03_867](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2003_Robust_Reading_Competitions): 4.9 MB, 867 samples<br>
- [IC13_857](http://rrc.cvc.uab.es/?ch=2): 72 MB, 857 samples<br>
- [IC13_1015](http://rrc.cvc.uab.es/?ch=2): 77 MB, 1015 samples<br>
- [IC15_1811](http://rrc.cvc.uab.es/?ch=4): 21 MB, 1811 samples<br>
- [IC15_2077](http://rrc.cvc.uab.es/?ch=4): 25 MB, 2077 samples<br>
- [IIIT5k_3000](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html): 50 MB, 3000 samples<br>
- [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset): 2.4 MB, 647 samples<br>
- [SVTP](http://openaccess.thecvf.com/content_iccv_2013/papers/Phan_Recognizing_Text_with_2013_ICCV_paper.pdf): 1.8 MB, 645 samples


准备好的数据集文件结构应如下所示：
``` text
datasets
├── test
│   ├── CUTE80
│   ├── IC03_860
│   ├── IC03_867
│   ├── IC13_857
│   ├── IC13_1015
│   ├── IC15_1811
│   ├── IC15_2077
│   ├── IIIT5k_3000
│   ├── SVT
│   ├── SVTP
├── evaluation
│   ├── Sumof6benchmarks
│   ├── ...
└── train
    ├── MJSynth
    └── SynText
```

### 3.3 更新yaml配置文件

如果数据集放置在`./datasets`目录下，则无需更改yaml配置文件`configs/rec/visionlan/visionlan_L*.yaml`中的`train.dataset.dataset_root`。
否则，请相应地更改以下字段：


```yaml
...
train:
  dataset_sink_mode: False
  dataset:
    type: LMDBDataset
    dataset_root: dir/to/dataset          <--- 更新
    data_dir: train                       <--- 更新
...
eval:
  dataset_sink_mode: False
  dataset:
    type: LMDBDataset
    dataset_root: dir/to/dataset          <--- 更新
    data_dir: evaluation/Sumof6benchmarks <--- 更新
...
```

> 您也可以选择根据CPU的线程数量来修改 `train.loader.num_workers`.

除了数据集设定以外, 请检查下列的重要参数: `system.distribute`, `system.val_while_train`, `common.batch_size`。这些参数的含义解释如下:


```yaml
system:
  distribute: True                                                    # 分布式训练选择`True`， 单卡训练选择 `False`
  amp_level: 'O0'
  seed: 42
  val_while_train: True                                               # 训练途中进行验证
common:
  ...
  batch_size: &batch_size 192                                          # 训练batch size
...
  loader:
      shuffle: False
      batch_size: 64                                                  # 验证/测试 batch size
...
```

**注意：**
- 由于全局批大小 （batch_size x num_devices） 是对结果复现很重要，因此当GPU/NPU卡数发生变化时，调整batch_size以保持全局批大小不变，或将学习率线性调整为新的全局批大小。

### 3.4 训练

训练阶段包括无语言（LF）和有语言（LA）过程，总共有三个训练步骤：

```text
LF_1：训练骨干网络和VRM，不训练MLM
LF_2：训练MLM并微调骨干网络和VRM
LA：使用MLM生成的掩码遮挡特征图，训练骨干网络、MLM和VRM
```

我们接下来使用分布式训练进行这三个步骤。对于单卡训练，请参考[识别教程](../../../docs/cn/tutorials/training_recognition_custom_dataset.md#单卡训练)。

```shell
mpirun --allow-run-as-root -n 4 python tools/train.py --config configs/rec/visionlan/visionlan_resnet45_LF_1.yaml
mpirun --allow-run-as-root -n 4 python tools/train.py --config configs/rec/visionlan/visionlan_resnet45_LF_2.yaml
mpirun --allow-run-as-root -n 4 python tools/train.py --config configs/rec/visionlan/visionlan_resnet45_LA.yaml
```

训练结果（包括checkpoints、每个阶段的性能和loss曲线）将保存在yaml配置文件中由参数`ckpt_save_dir`解析的目录中。默认目录为`./tmp_visionlan`。


### 3.5 测试

在完成上述三个训练步骤以后, 用户需要在测试前，将 `configs/rec/visionlan/visionlan_resnet45_LA.yaml` 文件中的`system.distribute`改为 `False`。

若要评估已训练模型的准确性, 有以下两个方法可供选择：


- 方法一: 先对每一个数据集进行评估： CUTE80, IC03_860, IC03_867, IC13_857, IC131015, IC15_1811, IC15_2077, IIIT5k_3000, SVT, SVTP。 然后再计算平均精度。

CUTE80 数据集的评估脚本如下：
```shell
model_name="e8"
yaml_file="configs/rec/visionlan/visionlan_resnet45_LA.yaml"
training_step="LA"

python tools/eval.py --config $yaml_file --opt eval.dataset.data_dir=test/CUTE80 eval.ckpt_load_path="./tmp_visionlan/${training_step}/${model_name}.ckpt"

```

- 方法二: 先将全部的测试集放在同一个文件夹下，例如 `datasets/test/`, 然后用 `tools/benchmarking/multi_dataset_eval.py`进行多数据集评估。示例脚本如下:

```shell
model_name="e8"
yaml_file="configs/rec/visionlan/visionlan_resnet45_LA.yaml"
training_step="LA"

python tools/benchmarking/multi_dataset_eval.py --config $yaml_file --opt eval.dataset.data_dir="test" eval.ckpt_load_path="./tmp_visionlan/${training_step}/${model_name}.ckpt"
```


## 4. 推理

### 4.1 准备 MINDIR 文件

请从上面的表格中中下载[MINDIR](https://download.mindspore.cn/toolkits/mindocr/visionlan/visionlan_resnet45_LA-e9720d9e-71b38d2d.mindir)文件，或者您可以使用`tools/export.py`将任何检查点文件手动转换为 MINDIR 文件：
```bash
# 有关更多参数使用详细信息，请执行 `python tools/export.py -h`
python tools/export.py --model_name_or_config visionlan_resnet45 --data_shape 64 256 --local_ckpt_path /path/to/visionlan-ckpt
```

此命令将在当前工作目录下保存一个`visionlan_resnet45.mindir`文件。

### 4.2 Mindspore Lite Converter Tool

如果您尚未下载 MindSpore Lite，请通过此[链接](https://www.mindspore.cn/lite/docs/en/master/use/downloads.html)进行下载。有关如何在 Linux 环境中使用 MindSpore Lite 的更多详细信息，请参阅[此文档](https://www.mindspore.cn/lite/docs/en/master/use/cloud_infer/converter_tool.html#linux-environment-usage-instructions)。

`converter_lite`工具是`mindspore-lite-{version}-linux-x64/tools/converter/converter`下的可执行程序。使用`converter_lite`，我们可以将 `MindIR`文件转换为`MindSpore Lite MindIR`文件。

首先，我们需要准备一个配置文件`config.txt`：
```text
[ascend_context]
input_format=NCHW
input_shape=args0:[1,3,64,256]
```
第一行`[ascend_context]`表示后续内容是`Ascend`后端的相关设置。通常，在创建配置文件时，必须添加此行以指示 Ascend 后端的相关设置内容；

第二行`input_format=NCHW`表示模型的输入格式为`[batch_size, channel_num, Height, Width]`；

第三行`input_shape=args0:[1,3,64,256]`表示模型输入的变量名为`args0`，`args0`的形状为`[1,3,64,256]`；如果使用其他模型或骨干网络，则形状的设置需要参考模型支持列表中的数据形状列；

准备好`config.txt`后，我们可以运行以下命令：
```bash
converter_lite \
     --saveType=MINDIR \
     --NoFusion=false \
     --fmk=MINDIR \
     --device=Ascend \
     --modelFile=path/to/mindir/file \
     --outputFile=visionlan_resnet45_lite \
     --configFile=config.txt
```
运行此命令将在当前工作目录下保存一个`visionlan_resnet45_lite.mindir`文件。这是我们可以在`Ascend310`或`310P`平台上进行推理的`MindSpore Lite MindIR`文件。您还可以通过更改`--outputFile`参数来定义不同的文件名。


### 4.3 对图像文件夹进行推理

以`SVT`测试集为例，数据集文件夹下的数据结构如下：
```text
svt_dataset
├── test
│   ├── 16_16_0_ORPHEUM.jpg
│   ├── 12_13_4_STREET.jpg
│   ├── 17_08_TOWN.jpg
│   ├── ...
└── test_gt.txt

```
我们使用以下命令对`visionlan_resnet45_lite.mindir`文件进行推理：
```bash
python deploy/py_infer/infer.py  \
    --input_images_dir=/path/to/svt_dataset/test  \
    --device_id=0  \
    --parallel_num=1  \
    --rec_model_path=/path/to/visionlan_resnet45_lite.mindir  \
    --rec_model_name_or_config=configs/rec/visionlan/visionlan_resnet45_LA.yaml \
    --res_save_dir=rec_svt
```

运行此命令将在当前工作目录下创建一个名为`rec_svt`的文件夹，并保存一个预测文件`rec_svt/rec_results.txt`。该文件中的一些预测示例如下所示：
```text
16_16_0_ORPHEUM.jpg "orpheum"
12_13_4_STREET.jpg  "street"
17_08_TOWN.jpg  "town"
...
```

然后，我们可以使用以下命令计算预测准确率：
```bash
python deploy/eval_utils/eval_rec.py  \
    --pred_path=rec_svt/rec_results.txt  \
    --gt_path=/path/to/svt_dataset/test_gt.txt
```
评估结果如下所示：
```text
{'acc': 0.9227202534675598, 'norm_edit_distance': 0.9720136523246765}
```

## 5. 引用文献
<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Yuxin Wang, Hongtao Xie, Shancheng Fang, Jing Wang, Shenggao Zhu, Yongdong Zhang: From Two to One: A New Scene Text Recognizer with Visual Language Modeling Network. ICCV 2021: 14174-14183
