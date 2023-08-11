# MASC: Multi-task Anomaly Segmentation and Classification

MASC is a Pytorch and MONAI (https://monai.io/) based multi-task framework for multi-class segmentation and classification. 

All code was created by Paula Ramirez Gilliand. 

(Work under development) 

Our framework is suitable for datasets with distinct segmentation types (in our case, three aortic arch anomalies). The intent is to leverage valuable class-specific information learnt from the classifier, to improve the inter-class segmentation predictions, i.e. to enforce the segmenter to learn important topological segmentation network for each class. 
Individual segmentation and classification training is also supported. 

The default architectures are Attention U-Net for segmentation [1] and DenseNet121 [2] (see networks/prepare_networks.py). 


![image](https://user-images.githubusercontent.com/93882352/231525071-c67e6777-2417-4abd-8f1b-6574e98ec0f5.png)



## Requirements

To install the required packages 

```bash
conda env create --file environment.yml
```

## Train an experiment 

To train or test an experiment, run ``` python main.py --config config.yml```, specifying the experiment information in the config file. 
Training from scratch of classifier, segmenter and joint multi-task training is supported. Training and validation losses are saved in results_dir folder. 

Any existing checkpoints in the directories specified in the config file will be loaded and used as starting point for experiment training. 
Inference and testing is also supported for each network. 


For segmentation, testing includes saving the .nii.gz segmentation predictions in the specified results_dir folder, as well as a .csv file (seg_overlap_metrics.csv) containing similarity metrics to ground truth (GT) for each segmentation label. 
For classification, testing saves the classification report in a .csv (classification_report.csv), alongside confusion matrices. 

Inference is done for the multi-task framework, where for an input image the segmentation prediction is saved in res_dir, with the predicted condition (class_out) in the filename ("cnn-lab-" + class_out + "-" + original_filename.nii.gz). 

## Config file description 

A config.yml file may be used to specify experiment parameters. This includes the parameters described below. 

### Directory information 
- ckpt_dir: directory specifying where to save network checkpoints (.ckpt files). If this folder doesn't exist it will be created.  
- res_dir: directory specifying where to save network results, including predicted segmentation nifti files, and csv files containing segmentation metrics (dice scores, 95th percentile Hausdorff distance, average surface distance, precision), and classification metrics, as well as training and validation loss plots. If this folder doesn't exist it will be created.  
- data_dir: directory containing JSON file with data information, and data files (subdirectories may be specified in JSON file) 
- data_JSON_file: data.json JSON file containing dataset information (described in section below). 
- ckpt_name_seg: name of the segmenter checkpoint filename, if loading pre-trained network
- ckpt_name_class: name of the segmenter checkpoint filename, if loading pre-trained network

### Experiment parameters 

- experiment_type: defines experiment type (str). One of: "segment" (only train segmenter), "classify" (only train classifier), "joint" (multi-task joint classifier + segmenter)
- input_type_class: str defining expected input to classifier (str). One of: "multi" (use multi-class softmax segmentation labels), "binary" (use binary segmentation labels, adds multi-class preds), "img": use input volume image
- training: boolean. If True will perform training, otherwise will go straight to testing or inference. 
- infer: boolean. If True will perform inference on "inference" dictionary datalist. 
- eval_num: int defining every how many epochs to perform validation. 
- max_iterations: int defining maximum training iterations
- batch_size: int defining batch size
- gpu_ids: int defining GPU ID 

### Classifier parameters
- dropout_class: float defining dropout probability for classifier network 
- lr_class: float defining initialised classifier learning rate (uses linearly decaying) 
- weight_decay_class: float defining classifier optimizer weight decay 

### Segmenter parameters
- dropout_seg: segmenter dropout probability
- lr_seg: segmenter initial learning rate 
- weight_decay_seg: weight decay of segmenter optimizer 
- chann_segnet: tuple specifying output channels for each network layer, e.g. (32, 64, 128, 256, 512)
- strides_segnet: tuple specifying output channels for each network layer, (2, 2, 2, 2)
- ksize_segnet: int or tuple specifying segmentation network convolutional kernel size
- up_ksize_segnet: int or tuple specifying segmentation network upsampling convolutional kernel size
- binary_seg_weight: weight in overall segmentation loss of the binary segmentation label term 
- multi_seg_weight: weight in overall segmentation loss of the multi-class segmentation label term 
- multi_task_weight: for joint training, weight balancing between classifier loss and segmenter loss. It multiplies the classifier loss  

### Data parameters
- spatial_dims: int specifying input image spatial dims 
- N_diagnosis: int specifying number of classes for classifier network 
- N_seg_labels: int specifying number of segmentation labels - NOTE: background should be included as a label 


## JSON dataset file 
The dataset information is read from a .JSON file, creating a Decathlon style datalist. An example of what the dataset file should include is described below, where "image" contains the image filename (and any subdirectories after the defined data_dir); "mask" contains the GT segmentation filename for testing, and binary masks for training; "LP" contains the multi-class labels filenames(not required for test, where "mask" should contain this); and "label" specifies the classifier label (aortic arch anomaly). 

```
{
"description": "example-dataset",
"labels": {
    "0": "background",
    "1": "SVC",
    "2": "LPA",
    "3": "RPA",
    "4": "AO",
    "5": "AD",
    "6": "DAO",
    "7": "MPA",
    "8": "LSA",
    "9": "BCA",
    "10": "LCCA",
    "11": "Pvs"
},
"licence": "yt",
"modality": {
    "0": "MRI"
},
"name": "cmr",
"reference": "KCL",
"release": "1.0 11/2021",
"tensorImageSize": "3D",

"testing": [
  {
  "image": "data/test/testcase1.nii.gz",
  "mask": "data/test/testcase1_mask.nii.gz",
  "label": 1.0
  }
],

  "training": [
  
    {
    "image": "data/train/traincase1.nii.gz",
    "mask": "data/train/traincase1_mask.nii.gz",
    "LP": "data/train/traincase1_LP.nii.gz",
    "label": 0.0
    },
    {
    "image": "data/train/traincase2.nii.gz",
    "mask": "data/train/traincase2_mask.nii.gz",
    "LP": "data/train/traincase2_LP.nii.gz",
    "label": 2.0
    }
   
  ],
  "validation": [
    
    {
    "image": "data/train/validcase1.nii.gz",
    "mask": "data/train/validcase1_mask.nii.gz",
    "LP": "data/train/validcase1_LP.nii.gz",
    "label": 2.0
    }
  ]
  }, 
    "inference": [
    
    {
    "image": "data/train/validcase1.nii.gz"
    }
  ]
  }
  
```


[1] Oktay, Ozan, et al. "Attention U-Net: Learning Where to Look for the Pancreas." Medical Imaging with Deep Learning.

[2] Zhang, Zhengxin, Qingjie Liu, and Yunhong Wang. "Road extraction by deep residual u-net." IEEE Geoscience and Remote Sensing Letters 15.5 (2018): 749-753.

# License
The SVRTK Fetal MRI Segmentation package is distributed under the terms of the Apache License Version 2. The license enables usage of SVRTK in both commercial and non-commercial applications, without restrictions on the licensing applied to the combined work.

# Citation and acknowledgements
In case you found SVRTK Fetal MRI Segmentation useful please give appropriate credit to the software by providing the corresponding link to our github repository:

SVRTK toolbox for fetal cardiac MRI segmentation and classification: https://github.com/SVRTK/MASC-multi-task-segmentation-and-classification/.

This is an extension of our workshop paper: 

Ramirez Gilliland, P., Uus, A., van Poppel, M.P., Grigorescu, I., Steinweg, J.K., Lloyd, D.F., Pushparajah, K., King, A.P. and Deprez, M., 2022, September. Automated Multi-class Fetal Cardiac Vessel Segmentation in Aortic Arch Anomalies Using T2-Weighted 3D Fetal MRI. *In Perinatal, Preterm and Paediatric Image Analysis: 7th International Workshop, PIPPI 2022, Held in Conjunction with MICCAI 2022, Singapore, September 18, 2022, Proceedings* (pp. 82-93). Cham: Springer Nature Switzerland.

The full paper version is submitted and will be added soon.


