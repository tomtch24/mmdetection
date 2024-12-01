import mmcv
import mmengine
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
#from mmseg.models import SEGMENTORS
#from mmdet.models import MODELS

# Register all MMSegmentation segmentors with MMDetection
#for name, cls in SEGMENTORS.module_dict.items():
#    MODELS.register_module(module=cls)

# Choose to use a config and initialize the detector
config_file = 'configs/segformer/segformer_faster_rcnn_cityscapes.py'
# Setup a checkpoint file to load
checkpoint_file = 'checkpoints/segformer.b0.1024x1024.city.160k.pth'

# Register all modules in mmdet into the registries
register_all_modules()

# Build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')  # or device='cuda:0'