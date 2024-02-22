
# the new config inherits the base configs to highlight the necessary modification
_base_ = './rotated_retinanet/rotated_retinanet_hbb_r50_fpn_1x_dota_oc.py'
# 1. dataset settings
dataset_type = 'DOTADataset'
classes = ('den', 'num')
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='../tools/data/marc/train/annots',
        img_prefix='../tools/data/marc/train/image'),
    val=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='../tools/data/marc/val/annots',
        img_prefix='../tools/data/marc/val/image'),
    test=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='../tools/data/marc/test/annots',
        img_prefix='../tools/data/marc/test/image'))

# 2. model settings
model = dict(
    bbox_head=dict(
        type='RotatedRetinaHead',
        # explicitly over-write all the `num_classes` field from default 15 to 5.
        num_classes=2))