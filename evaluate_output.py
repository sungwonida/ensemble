# AUTHOR: David Jung
# EMAIL: sungwonida@gmail.com
# DATE: 2020-09-06

import os
import sys
import logging
import torch
import cv2
from ensemble import Ensemble
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.structures import Boxes, Instances


def evaluate_output(infer_output, data_loader, evaluator):
    """
    Evaluate precomputed output using data_loader and evaluate the metrics with evaluator.

    Args:
        infer_output: a dictionary that holds image_id as key and Instances object as its value
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))

    evaluator.reset()

    for idx, inputs in enumerate(data_loader):
        filename = inputs[0]['file_name']
        basename = os.path.basename(filename)
        image_id = str(int(os.path.splitext(basename)[0]))

        if image_id in infer_output.keys():
            outputs = infer_output[image_id]
        else:
            height, width = cv2.imread(filename).shape[:2]
            instances = Instances((height, width))
            instances.pred_boxes = Boxes(torch.Tensor([]))
            instances.scores = torch.Tensor([])
            instances.pred_classes = torch.LongTensor([])
            outputs = [{'instances': instances}]
            # print(f"empty: {image_id}")

        evaluator.process(inputs, outputs)

        if idx % 10 == 0:
            sys.stdout.write('.'); sys.stdout.flush();

    print()
    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


if __name__ == '__main__':
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cpu"
    det_output = Ensemble.load('ensemble_results.json',
                               as_instances=True,
                               dataset_path='datasets/coco/val2017')
    evaluator = COCOEvaluator("coco_2017_val", cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, "coco_2017_val")
    print(evaluate_output(det_output, val_loader, evaluator))
