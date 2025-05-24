from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.cityscapes import load_cityscapes_semantic
from detectron2.utils.file_io import PathManager

import os



def load_idrid_semantic(image_dir, gt_dir):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "".
        gt_dir (str): path to the raw annotations. e.g., "".
    Returns:
        list[dict]: a list of dict, each has "file_name" and
            "sem_seg_file_name".
    """
    ret = []
    # gt_dir is small and contain many small files. make sense to fetch to local first
    #gt_dir = PathManager.get_local_path(gt_dir)
    for image_filename, label_filename in zip(sorted(os.listdir(image_dir)), sorted(os.listdir(gt_dir))):
        image_file = os.path.join(image_dir, image_filename)
        label_file = os.path.join(gt_dir, label_filename)

        ret.append(
            {
                "file_name": image_file,
                "sem_seg_file_name": label_file,
                "height": 2848,
                "width": 4288,
            }
        )

    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(
        ret[0]["sem_seg_file_name"]
    ), "idrid gt not found"  # noqa
    return ret



def register_idrid_semseg(root, split='train'):
    meta = {
        "thing_classes": ["1","2","3","4","5"],
        "stuff_classes": ["1","2","3","4","5"],
        "thing_colors": [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255)],
        "stuff_colors": [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255)]
    }
    root = os.path.join(root, 'IDRID_seg/CitiScapesSem_format')
    image_dir = os.path.join(root, '{}_imgs'.format(split))
    gt_dir = os.path.join(root, '{}_gt'.format(split))

    sem_key = 'idrid_semseg_{}'.format(split)
    DatasetCatalog.register(
        sem_key, lambda x=image_dir, y=gt_dir: load_idrid_semantic(x, y)
    )
    MetadataCatalog.get(sem_key).set(
        image_dir=image_dir,
        gt_dir=gt_dir,
        evaluator_type="med_sem_seg",
        ignore_label=255,
        **meta,
    )


def register_idrid_bg_semseg(root, split='train'):
    meta = {
        "thing_classes": ["0","1","2","3","4","5","6"],
        "stuff_classes": ["0","1","2","3","4","5","6"],
        "thing_colors": [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (0,0,0), (255,255,255)],
        "stuff_colors": [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (0,0,0), (255,255,255)]
    }
    root = os.path.join(root, 'IDRID_seg/CitiScapesSem_format_BG')
    image_dir = os.path.join(root, '{}_imgs'.format(split))
    gt_dir = os.path.join(root, '{}_gt'.format(split))

    sem_key = 'idrid_bg_semseg_{}'.format(split)
    DatasetCatalog.register(
        sem_key, lambda x=image_dir, y=gt_dir: load_idrid_semantic(x, y)
    )
    MetadataCatalog.get(sem_key).set(
        image_dir=image_dir,
        gt_dir=gt_dir,
        evaluator_type="med_sem_seg",
        ignore_label=255,
        **meta,
    )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_idrid_semseg(_root, 'train')
register_idrid_semseg(_root, 'test')
register_idrid_bg_semseg(_root, 'train')
register_idrid_bg_semseg(_root, 'test')
