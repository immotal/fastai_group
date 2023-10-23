# 训练模型，Mac 上不咋行，需要有 GPU 的 MPS 支持的不好, 直接在 kaggle 上训练更好(最后选的这个方案)
from fastai.vision.all import *

def get_image_and_train
path = untar_data(URLs.PETS)
dls = ImageDataLoaders.from_name_re(
    path, get_image_files(path/'images'), pat='(.+)_\d+.jpg', item_tfms=Resize(460), batch_tfms=aug_transforms(size=224, min_scale=0.75)
)
learn = vision_learner(dls, models.resnet50, metrics=accuracy)
learn.fine_tune(1)
learn.path = Path('.')
learn.export()