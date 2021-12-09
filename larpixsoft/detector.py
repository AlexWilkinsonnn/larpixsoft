"""
Set detector constants
"""
import numpy as np
import yaml

from collections import defaultdict
from dataclasses import dataclass, field

@dataclass
class Detector:
  """
  Detector constants
  """
  mm2cm: float = 0.1
  cm2mm: float = 10
  lar_density: float = 1.38 # g/cm^3
  E_field: float = 0.50 # kV/cm
  vdrift: float = 0.1648 # cm/us
  lifetime: float = 2.2e3 # us
  time_sampling: float = 0.1 # us
  time_interval: tuple = (0, 200.) # us
  time_padding: float = 10 # us
  sample_points: int = 40
  long_diff: float = 4.0e-6 # cm^2/us
  tran_diff: float = 8.8e-6 # cm^2/us 
  time_window: float = 8.9 # us
  drift_length: float = 0 # cm
  response_sampling: float = 0.1 # us
  tpc_borders: np.ndarray = np.zeros((0, 3, 2)) # cm
  tpc_offsets: np.ndarray = np.zeros((0, 3, 2)) # cm
  tile_borders: np.ndarray = np.zeros((2, 2)) # cm
  N_pixels: tuple = (0, 0)
  N_pixels_per_tile: tuple = (0, 0)
  pixel_connection_dict: dict = field(default_factory=dict)
  pixel_pitch: float = 0.4434 # cm
  tile_positions: dict = field(default_factory=dict) # mm
  tile_orientations: dict = field(default_factory=dict) # cm
  tile_map: tuple = ()
  tile_chip_to_io: dict = field(default_factory=dict)
  module_to_io_groups: dict = field(default_factory=dict)

  def get_time_ticks(self) -> np.ndarray:
    return np.linspace(self.time_interval[0], self.time_interval[1], 
      int(round(self.time_interval[1] - self.time_interval[0])/self.time_sampling) + 1)

  def get_zlims(self) -> tuple:
    return (np.min(self.tpc_borders[:, 2, :]), np.max(self.tpc_borders[:, 2, :]))


def set_detector_properties(detprop_file, pixel_file) -> Detector:
    """
    The function loads the detector properties and the pixel geometry YAML files and stores the 
    constants in a Detector dataclass
    Args:
        detprop_file (str): detector properties YAML
            filename
        pixel_file (str): pixel layout YAML filename
    """
    default_detector = Detector()
    consts = {}

    with open(detprop_file) as df:
        detprop = yaml.load(df, Loader=yaml.FullLoader)

    consts['drift_length'] = detprop['drift_length']

    consts['tpc_offsets'] = np.array(detprop['tpc_offsets'])
    consts['tpc_offsets'][:, [2, 0]] = consts['tpc_offsets'][:, [0, 2]] # Inverting x and z axes

    consts['time_interval'] = np.array(detprop['time_interval'])

    for key in ['time_padding', 'time_window', 'vdrift', 'lifetime', 'long_diff', 'tran_diff', 'response_sampling']:
      if key in detprop:
        consts[key] = detprop[key]

    with open(pixel_file, 'r') as pf:
        tile_layout = yaml.load(pf, Loader=yaml.FullLoader)

    consts['pixel_pitch'] = tile_layout['pixel_pitch'] * default_detector.mm2cm
    chip_channel_to_position = tile_layout['chip_channel_to_position']
    consts['pixel_connection_dict'] = { tuple(pix) : (chip_channel//1000, chip_channel%1000) for chip_channel, pix in chip_channel_to_position.items() }
    consts['tile_chip_to_io'] = tile_layout['tile_chip_to_io']

    xs = np.array(list(chip_channel_to_position.values()))[:,0] * consts['pixel_pitch']
    ys = np.array(list(chip_channel_to_position.values()))[:,1] * consts['pixel_pitch']
    consts['tile_borders'] = np.array([
      [-(max(xs) + consts['pixel_pitch'])/2, (max(xs) + consts['pixel_pitch'])/2],
      [-(max(ys) + consts['pixel_pitch'])/2, (max(ys) + consts['pixel_pitch'])/2]])

    tile_indeces = tile_layout['tile_indeces']
    consts['tile_orientations'] = tile_layout['tile_orientations']
    consts['tile_positions'] = tile_layout['tile_positions']
    tpc_ids = np.unique(np.array(list(tile_indeces.values()))[:,0], axis=0)

    anodes = defaultdict(list)
    for tpc_id in tpc_ids:
        for tile in tile_indeces:
            if tile_indeces[tile][0] == tpc_id:
                anodes[tpc_id].append(consts['tile_positions'][tile])

    consts['drift_length'] = detprop['drift_length']

    consts['tpc_offsets'] = np.array(detprop['tpc_offsets'])
    consts['tpc_offsets'][:, [2, 0]] = consts['tpc_offsets'][:, [0, 2]] # Inverting x and z axes

    consts['tpc_borders'] = np.empty((consts['tpc_offsets'].shape[0] * tpc_ids.shape[0], 3, 2))

    for ia, tpc_offset in enumerate(consts['tpc_offsets']):
        for ib, anode in enumerate(anodes):
            tiles = np.vstack(anodes[anode])
            tiles *= default_detector.mm2cm
            drift_direction = 1 if anode == 1 else -1
            x_border = min(tiles[:,2]) + consts['tile_borders'][0][0] + tpc_offset[0], \
                       max(tiles[:,2]) + consts['tile_borders'][0][1] + tpc_offset[0]
            y_border = min(tiles[:,1]) + consts['tile_borders'][1][0] + tpc_offset[1], \
                       max(tiles[:,1]) + consts['tile_borders'][1][1] + tpc_offset[1]
            z_border = min(tiles[:,0]) + tpc_offset[2], \
                       max(tiles[:,0]) + consts['drift_length'] * drift_direction + tpc_offset[2]
            consts['tpc_borders'][ia*2 + ib] = (x_border, y_border, z_border)

    consts['tile_map'] = detprop['tile_map']

    ntiles_x = len(consts['tile_map'][0])
    ntiles_y = len(consts['tile_map'][0][0])

    consts['N_pixels'] = len(np.unique(xs))*ntiles_x, len(np.unique(ys))*ntiles_y
    consts['N_pixels_per_tile'] = len(np.unique(xs)), len(np.unique(ys))
    consts['module_to_io_groups'] = detprop['module_to_io_groups']

    return Detector(**consts)