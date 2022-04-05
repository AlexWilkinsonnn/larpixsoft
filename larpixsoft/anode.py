from .detector import Detector

class Anode():
  def __init__(self, module_id, io_group, detector : Detector):
    self.detector = detector
    self.id = module_id
    self.io_group = io_group
    # the coordinates of the tpc
    self.tpc_x = detector.tpc_offsets[module_id][0]
    self.tpc_y = detector.tpc_offsets[module_id][1]
    self.tpc_z = detector.tpc_offsets[module_id][2]
    # want the coordinates of the anode-cathode box, can be either the lower or upper z anode plane
    if io_group in [1,2]: # lower z anode, not sure how general this is but looks like it works for 5x7
      self.vol_borders = detector.tpc_borders[(module_id + 1)*2 - 2]
    else:
      self.vol_borders = detector.tpc_borders[(module_id + 1)*2 - 1]
    self.z = self.vol_borders[2, 0]

  def drift_time_lowerz(self):
    return ((self.z - self.detector.get_zlims()[0])/self.detector.vdrift)*(1/self.detector.time_sampling) 

  def drift_time_upperz(self):
    return ((self.detector.get_zlims()[1] - self.z)/self.detector.vdrift)*(1/self.detector.time_sampling) 
