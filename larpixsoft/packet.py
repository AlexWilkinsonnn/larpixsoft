from dataclasses import dataclass

from .anode import Anode

class TriggerPacket():
  def __init__(self, packet):
    self.timestamp = packet['timestamp']


class DataPacket():
  def __init__(self, packet, geometry):
    self.timestamp = packet['timestamp']
    self.ADC = packet['dataword']

    io_group, io_channel, chip, channel = packet['io_group'], packet['io_channel'], packet['chip_id'], packet['channel_id']
    module_id = (io_group - 1)//4 # need modules to start from 0 because of tpc_offsets
    io_group = io_group - ((io_group - 1)//4)*4 # tile to io is only defined for first 4 io groups of first module
    self.x, self.y = geometry[(io_group, io_channel, chip, channel)]
    self.anode = Anode(module_id)
    self.projected = False

  def project(): 
    """
    Project onto anodes at large z side of the detector.
    """
    pass



    


    
    