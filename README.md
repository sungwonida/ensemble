# ensemble

## Algorithm

Borrowed the idea mainly from [here](https://github.com/ahrnbom/ensemble-objdet/blob/master/ensemble.py).  
The idea is used for ensembling the boxes that have same `image_id` and `category_id`.  

## Test

The test has been conducted using COCO 2017 val dataset and Detectron2.  
In order to reproduce what I've got during the test, you can follow the below.  

### 1. Prepare conda environment
Run the command line in Linux.
``` shell
$ conda env create -f environment_linux.yml
```
The linux environment has been tested without using GPU due to WSL limitation.

or Windows (this version utilizes the GPU)
``` shell
$ conda env create -f environment_win32.yml
```

Activate the environment.
``` shell
$ conda activate torch
```

### 2. Prepare COCO 2017 val dataset
``` shell
$ wget http://images.cocodataset.org/zips/val2017.zip
$ wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
$ mkdir -p datasets/coco
$ unzip val2017.zip -d datasets/coco
$ unzip annotations_trainval2017.zip -d datasets/coco 
```

### 3. Emit estimation results of coco_2017_val for object detection models
``` shell
$ python evaluate_pretrained_model.py
$ ls -A1 datasets/coco
coco_instances_results_faster_rcnn_R_50_DC5_1x.json
coco_instances_results_faster_rcnn_R_50_FPN_1x.json
coco_instances_results_faster_rcnn_R_50_FPN_3x.json
```

### 4. Create an ensemble
``` shell
$ python ensemble.py
saved to ensemble_results.json
```

### 5. Get the result of estimation on the ensemble
``` shell
$ python evaluate_output.py
Loading and preparing results...
DONE (t=0.23s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
COCOeval_opt.evaluate() finished in 8.24 seconds.
Accumulating evaluation results...
COCOeval_opt.accumulate() finished in 0.87 seconds.
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.386
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.572
...
```

| pretrained model        | box AP (at IoU=0.50:0.95, area=all) |
| ----------------------- | ----------------------------------- |
| faster_rcnn_R_50_DC5_1x | 35.02630019513202                   |
| faster_rcnn_R_50_FPN_1x | 34.35279506894608                   |
| faster_rcnn_R_50_FPN_3x | 36.66724655820816                   |
| Ensemble                | 38.61288520614146                   |
