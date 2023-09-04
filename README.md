# C2FVL

Congratulations to our C2FVL has been accepted by IEEE International Conference on Acoustics, Speech and Signal Processing 2023 (ICASSP 2023)!

This repo is the official implementation of "**COARSE-TO-FINE COVID-19 SEGMENTATION VIA VISION-LANGUAGE ALIGNMENT**" [Arxiv](https://arxiv.org/abs/2303.00279). 

You can also refer to our another work [LViT](https://github.com/HUANGLIZI/LViT), which is basically built by the same Codebase.

![image](https://github.com/HUANGLIZI/C2FVL/blob/main/IMG/C2FVL.png)

## Requirements

Install from the ```requirements.txt``` using:
```angular2html
pip install -r requirements.txt
```
Questions about NumPy version conflict. The NumPy version we use is 1.17.5. We can install bert-embedding first, and install NumPy then.

## Usage

### 1. Data Preparation
#### 1.1. QaTa-COVID and MosMedData+ Datasets
The original data can be downloaded in following links:
* QaTa-COVID Dataset - [Link (Original)](https://www.kaggle.com/datasets/aysendegerli/qatacov19-dataset)

* MosMedData+ Dataset - [Link (Original)](https://www.kaggle.com/datasets/maedemaftouni/covid19-ct-scan-lesion-segmentation-dataset)

  *Note: The text annotation of QaTa-COVID train datasets [download link](https://1drv.ms/x/s!AihndoV8PhTDguFoqa9YVfXdadWtsA?e=YpQ1pL).*
  
  *The text annotation of QaTa-COVID val datasets [download link](https://1drv.ms/x/s!AihndoV8PhTDguFmGZlojiUgiUK_Bw?e=dCt72d).*
  
  *The text annotation of QaTa-COVID test dataset [download link](https://1drv.ms/x/s!AihndoV8PhTDguFndDhGZ_w09BwQCw?e=1CrgtK).*
  
  *(Note: The text annotation of MosMedData+ dataset will be released in the future. And you can email to me for the datasets)*

#### 1.2. Format Preparation

Then prepare the datasets in the following format for easy use of the code:

```angular2html
├── datasets
    ├── QaTa-COVID
    │   ├── Test_Folder
    |   |   ├── Test_text.xlsx
    │   │   ├── img
    │   │   └── labelcol
    │   ├── Train_Folder
    |   |   ├── Train_text.xlsx
    │   │   ├── img
    │   │   └── labelcol
    │   └── Val_Folder
    |	    ├── Val_text.xlsx
    │       ├── img
    │       └── labelcol
    └── MosMedData+
        ├── Test_Folder
        |   ├── Test_text.xlsx
        │   ├── img
        │   └── labelcol
        ├── Train_Folder
        |   ├── Train_text.xlsx
        │   ├── img
        │   └── labelcol
        └── Val_Folder
            ├── Val_text.xlsx
            ├── img
            └── labelcol
```



### 2. Training


You can train to get your own model.

```angular2html
python train_model.py
```



### 3. Evaluation
#### 3.1. Get Pre-trained Models
Here, we provide pre-trained weights on QaTa-COVID and MosMedData+, if you do not want to train the models by yourself, you can download them in the following links:

*(Note: the pre-trained model will be released in the future.)*

* QaTa-COVID: 
* MosMedData+: 
#### 3.2. Test the Model and Visualize the Segmentation Results
First, change the session name in ```Config.py``` as the training phase. Then run:
```angular2html
python test_model.py
```
You can get the Dice and IoU scores and the visualization results. 



### 4. Results

| Dataset    | 	  Dice (%) | IoU (%) |
| ---------- | ------------------- | -------- |
| QaTa-COVID | 83.40    | 74.62   |
| MosMedData+    | 74.56      | 61.15  |



### 5. Reproducibility

In our code, we carefully set the random seed and set cudnn as 'deterministic' mode to eliminate the randomness. However, there still exsist some factors which may cause different training results, e.g., the cuda version, GPU types, the number of GPUs and etc. The GPU used in our experiments are on four NVIDIA GeForce GTX 1080 Ti GPUs and the cuda version is 11.2. And the upsampling operation has big problems with randomness for multi-GPU cases.
See https://pytorch.org/docs/stable/notes/randomness.html for more details.



## Reference

* [LViT](https://github.com/HUANGLIZI/LViT) 
* [TransUNet](https://github.com/Beckschen/TransUNet) 
* [MedT](https://github.com/jeya-maria-jose/Medical-Transformer)
* [UCTransNet](https://github.com/McGregorWwww/UCTransNet)


# Citation

```bash
@inproceedings{shan2023coarse,
  title={Coarse-to-Fine Covid-19 Segmentation via Vision-Language Alignment},
  author={Shan, Dandan and Li, Zihan and Chen, Wentao and Li, Qingde and Tian, Jie and Hong, Qingqi},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```
