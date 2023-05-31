from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
import os
from detectron2.data.datasets import register_coco_instances
def evaluation():
    register_coco_instances("custom", {}, "datasets/coco/annotations/instances_test2014.json", "datasets/coco/test2014")
    custom_metadata = MetadataCatalog.get("custom")
    DatasetCatalog.get("custom")
    cfg = get_cfg()
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.merge_from_file("configs/BlendMask/R_50_1x.yaml")
    cfg.DATASETS.TEST = ("custom",)
    cfg.MODEL.WEIGHTS = os.path.join("training_dir/blendmask_R_50_1x/blendmask1.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        512)
    cfg.DATALOADER.NUM_WORKERS = 0

    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("custom", cfg, False, output_dir="./output/")
    # evaluator = COCOEvaluator("custom",  output_dir="./training_dir/")
    val_loader = build_detection_test_loader(cfg, "custom")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))

if __name__ == "__main__":
    evaluation()