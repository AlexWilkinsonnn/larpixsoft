from chardet import detect
from larpixsoft.detector import Detector

from .anode import Anode

class TriggerPacket():
  def __init__(self, packet):
    self.t = packet['timestamp']

class DataPacket():
  def __init__(self, packet, geometry, detector : Detector, id=-1):
    self.timestamp = packet['timestamp']
    self.ADC = packet['dataword'] if packet['dataword'] == 0 else packet['dataword'] - detector.adc_ped
    self.t_0 = 0.0

    io_group, io_channel, chip, channel = packet['io_group'], packet['io_channel'], packet['chip_id'], packet['channel_id']
    module_id = (io_group - 1)//4 # need modules to start from 0 because of tpc_offsets
    io_group = io_group - ((io_group - 1)//4)*4 # tile to io is only defined for first 4 io groups of first module
    self.io_group = io_group
    self.x, self.y = geometry[(io_group, io_channel, chip, channel)] 
    self.anode = Anode(module_id, io_group, detector)

    self.objectid = id

  def __eq__(self, other):
    if type(other) == type(self):
      if self.objectid != -1:
        return self.objectid == other.objectid
      else:
        return super(DataPacket, self).__eq__(other)

    else:
      return False

  def __hash__(self):
    if self.objectid != -1:
      return hash(self.objectid)
    else:
      return super(DataPacket, self).__hash__()

  def add_trigger(self, trigger_packet : TriggerPacket):
    self.t_0 = trigger_packet.t

  def t(self):
    return self.timestamp - self.t_0

  def project_lowerz(self): 
    """
    Project onto anodes at large negative z side of the detector.
    """
    if self.io_group in [1,2]:
      t_d = self.anode.drift_time_lowerz() + self.t()
    else:
      t_d = self.anode.drift_time_lowerz() - self.t()
  
    return t_d

  def project_upperz(self): 
      """
      Project onto anodes at large positive z side of the detector.
      """
      if self.io_group in [1,2]:
        t_d = self.anode.drift_time_upperz() - self.t()
      else:
        t_d = self.anode.drift_time_upperz() + self.t()
    
      return t_d
    