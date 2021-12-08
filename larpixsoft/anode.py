from .detector import Detector

class Anode():
  def __init__(self, module_id, detector : Detector):
    # id, coordinate, dimension, how many rows behind 'front' anode
    self.id = module_id
    self.x = detector.tpc_offsets[module_id][0]
    self.y = detector.tpc_offsets[module_id][1]
    self.z = detector.tpc_offsets[module_id][2]
