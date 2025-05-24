from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager

import os



def load_dsfrance_semantic(image_dir, gt_dir):
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
                "height": 400,
                "width": 400,
            }
        )

    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(
        ret[0]["sem_seg_file_name"]
    ), "dsfrance gt not found"  # noqa
    return ret



def register_dsfrance_semseg_dataset(root, split='train'):
    meta = {
        "thing_classes": ["1"],
        "stuff_classes": ["1"],
        "thing_colors": [(255,0,0)],
        "stuff_colors": [(255,0,0)]
    }
    root = os.path.join(root, 'google-cityscape/')
    image_dir = os.path.join(root, '{}_imgs'.format(split))
    gt_dir = os.path.join(root, '{}_gt'.format(split))

    sem_key = 'dsfrance_semseg_{}'.format(split)
    DatasetCatalog.register(
        sem_key, lambda x=image_dir, y=gt_dir: load_dsfrance_semantic(x, y)
    )
    MetadataCatalog.get(sem_key).set(
        image_dir=image_dir,
        gt_dir=gt_dir,
        evaluator_type="sem_seg",
        ignore_label=255,
        **meta,
    )




_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_dsfrance_semseg_dataset(_root, 'train')
register_dsfrance_semseg_dataset(_root, 'test')
