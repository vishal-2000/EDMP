'''Guide Class

self.cfg is a dictionary. Example of self.cfg:

index: 1

hyperparameters:
  batch_size: 10
  obstacle_clearance:
    range: [0., 0.]
  obstacle_expansion:
    ind_set_range1: [255, 150]
    val_range1: [0.4, 0.4]
    ind_set_range2: [150, 20]
    val_range2: [0.4, 0.]
    ind_set_range3: [20, 0]
    val_range3: [0., 0.]
  guidance_method: 'iv' # 'iv': intersection_volume; 'sv': swept_volume
  grad_norm: True
  
  guidance_schedule:
    type: 'constant' # 'constant' or 'varying' (1.4 + (t/self.T))
    scale_val: 0.05
'''
from autolab_core import YamlConfig

class Guide:
    def __init__(self, path_to_yaml='./guides/cfgs/guide1.yaml') -> None:
        self.cfg = YamlConfig(path_to_yaml) # Dictionary

if __name__=='__main__':
    classifier = Guide()
    