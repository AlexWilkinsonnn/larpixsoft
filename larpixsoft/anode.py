from .detector import Detector

class Anode():
  def __init__(self, module_id, io_group, detector : Detector):
    # id, coordinate, dimension, how many rows behind 'front' anode
    self.id = module_id
    # the coordinates of the tpc
    self.tpc_x = detector.tpc_offsets[module_id][0]
    self.tpc_y = detector.tpc_offsets[module_id][1]
    self.tpc_z = detector.tpc_offsets[module_id][2]
    # want the coordinates of the anode-cathode box, can be either the lower or upper z anode plane
    if io_group in [1,2]: # lower z anode
      self.borders = detector.tpc_borders[(module_id + 1)*2 - 2]
    else:
      self.borders = detector.tpc_borders[(module_id + 1)*2 - 1]

  
      
