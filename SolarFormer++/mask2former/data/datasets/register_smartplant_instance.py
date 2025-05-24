import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_sem_seg, register_coco_instances



def register_smart_plant_single(root):
    root = os.path.join(root, 'smart_plant/front_2_class_new_COCO')
    splits = ['train', 'val', 'test']
    for split in splits:
        register_coco_instances("smart_plant_single_{}".format(split), {},
                os.path.join(root, "annotations/instances_{}2017.json".format(split)),
                os.path.join(root, "{}2017".format(split))
        )

        meta = MetadataCatalog.get('smart_plant_single_{}'.format(split))
        meta.thing_classes = ['normal', 'defect']
        meta.thing_colors = [(0,127.5,127.5), (127.5,127.5,0)]
        meta.stuff_colors = [(0,127.5,127.5), (127.5,127.5,0)]


def register_smart_plant_overlap(root):
    root = os.path.join(root, 'smart_plant/front_2_class_overlap')
    splits = ['train', 'val', 'test']
    for split in splits:
        register_coco_instances("smart_plant_overlap_{}".format(split), {},
                os.path.join(root, "annotations/instances_{}2017.json".format(split)),
                os.path.join(root, "{}2017".format(split))
        )

        meta = MetadataCatalog.get('smart_plant_overlap_{}'.format(split))
        meta.thing_classes = ['normal', 'defect']
        meta.thing_colors = [(0,127.5,127.5), (127.5,127.5,0)]
        meta.stuff_colors = [(0,127.5,127.5), (127.5,127.5,0)]


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_smart_plant_single(_root)
register_smart_plant_overlap(_root)
