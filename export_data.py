import os, argparse, sys, importlib, collections

import h5py
import numpy as np

from larpixsoft.detector import set_detector_properties
from larpixsoft.geometry import get_geom_map

from larpixsoft.funcs import get_wires, get_events, get_wire_hits

# NOTE move away from importing classes and functions and just use the module name space
#     eg. `import larpixsoft.funcs as funcs` then do funcs.get_wires

def main(INPUT_FILES, N, OUTPUT_DIR, EXCLUDED_NUMS_FILE):
  detector = set_detector_properties('data/detector/ndlar-module.yaml', 
    'data/pixel_layout/multi_tile_layout-3.0.40.yaml')
  geometry = get_geom_map('data/pixel_layout/multi_tile_layout-3.0.40.yaml')

  pitch = 0.479
  x_start = 480
  segment_length = 0.04 # The max step length for LArG4 in [cm]
  projection_anode = 'upper_z'

  excluded_nums = []
  if EXCLUDED_NUMS_FILE:
    with open(EXCLUDED_NUMS_FILE, 'r') as f:
      for line in f:
        excluded_nums.append(int(line))
  if excluded_nums:
    print("{} events are being excluded".format(len(excluded_nums)))

  if not OUTPUT_DIR:
    out_dirname = ''
    for input_file in INPUT_FILES:
      out_dirname += os.path.splitext(os.path.basename(input_file))[0]
      out_dirname += '-'
    out_dirname = out_dirname[:-1] 
    out_dir = os.path.join('data/out', out_dirname)
  else:
    out_dir = OUTPUT_DIR

  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

  n_passed, num = 0, 0
  n_z_failed, n_x_failed, n_adc_failed = 0, 0, 0
  for input_file in INPUT_FILES:
    f = h5py.File(input_file, 'r')

    wires = get_wires(pitch, x_start)
    data_packets, tracks = get_events(f['packets'], f['mc_packets_assn'], f['tracks'], geometry,
      detector, N=N, x_min_max=(479.7605, 709.92)) # min/max wire x +- pitch/2

    for i, (event_data_packets, event_tracks) in enumerate(zip(data_packets, tracks)):
      if i + 1 == len(data_packets):
        print("[{}/{}] - {} passed cuts: {} failed z cuts {} failed x cuts {} failed adc cut".format(
          i + 1, len(data_packets), n_passed, n_z_failed, n_x_failed, n_adc_failed))
      else:
        print("[{}/{}] - {} passed cuts: {} failed z cuts {} failed x cuts {} failed adc cut".format(
          i + 1, len(data_packets), n_passed, n_z_failed, n_x_failed, n_adc_failed), end='\r')

      if num in excluded_nums:
        n_passed += 1 
        num += 1
        continue

      x_min_track = min(event_tracks, key=lambda track: min(track.x_start, track.x_end)) 
      x_min = min([x_min_track.x_start, x_min_track.x_end])
      x_max_track = max(event_tracks, key=lambda track: max(track.x_start, track.x_end))
      x_max = max([x_max_track.x_start, x_max_track.x_end])
      y_min_track = min(event_tracks, key=lambda track: min(track.y_start, track.y_end)) 
      y_min = min([y_min_track.y_start, y_min_track.y_end])
      y_max_track = max(event_tracks, key=lambda track: max(track.y_start, track.y_end))
      y_max = max([y_max_track.y_start, y_max_track.y_end])
      z_min_track = min(event_tracks, key=lambda track: min(track.z_start, track.z_end)) 
      z_min = min([z_min_track.z_start, z_min_track.z_end])
      z_max_track = max(event_tracks, key=lambda track: max(track.z_start, track.z_end))
      z_max = max([z_max_track.z_start, z_max_track.z_end])
      t_min_track = min(event_tracks, key=lambda track: min(track.t_start, track.t_end)) 
      t_min = min([t_min_track.t_start, t_min_track.t_end])
      t_max_track = max(event_tracks, key=lambda track: max(track.t_start, track.t_end))
      t_max = max([t_max_track.t_start, t_max_track.t_end])

      # cuts
      if z_max - z_min >= 300:
        num += 1
        n_z_failed += 1
        continue
      
      if x_min <= 479.7605 or x_max >= 709.92:
        num += 1
        n_x_failed += 1
        continue

      total_adc = sum([ packet.ADC for packet in event_data_packets ])
      if total_adc < 5000:
        num += 1
        n_adc_failed += 1
        continue

      wire_hits = get_wire_hits(event_data_packets, pitch, wires, tick_scaledown=5, 
        projection_anode=projection_anode)

      tick_shift = 500 - min(wire_hits, key=lambda hit: hit['tick'])['tick']
      for hit in wire_hits:
        hit['tick'] += tick_shift
      if max(wire_hits, key=lambda hit: hit['tick'])['tick'] >= 4492:
        print("cut failed on crop")
        num += 1
        continue

      arr_det = np.zeros((1, 512, 4608))
      for hit in wire_hits:
        arr_det[0, hit['ch'] + 16, hit['tick'] + 58] += hit['adc']

      np.save(os.path.join(out_dir, "ND_detsim_{}.npy".format(num)), arr_det)

      with open(os.path.join(out_dir, "ND_depos_{}.txt".format(num)), 'w') as f:
        f.write("input_file:{},event_num:{},segment_length:{},view:Z,projection_anode:{}," + 
          "first_wire:{},last_wire={},earliest_tick={}".format(
          input_file, i, segment_length, projection_anode, min(wires.values()), max(wires.values()), 500))
        f.write("x_min:{},x_max:{},y_min:{},y_max:{},z_min:{},z_max:{},t_min:{},t_max:{}\n".format(
          x_min, x_max, y_min, y_max, z_min, z_max, t_min, t_max))
        
        for track in event_tracks:
          segments = track.segments(segment_length)
          for segment in segments:
            f.write("{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
              track.trackid, track.pdg,
              segment['x_start'], segment['x_end'], segment['y_start'], segment['y_end'],
              segment['z_start'], segment['z_end'], segment['t_start'], segment['t_end'], 
              segment['electrons'], segment['dE']))

      n_passed += 1
      num += 1

  print("{} passed cuts".format(n_passed))
    
def parse_arguments():
  parser = argparse.ArgumentParser()

  parser.add_argument("input_files", nargs='+')

  parser.add_argument("-n", type=int, default=0)
  parser.add_argument("-o", type=str, default='', help='output dir name')
  parser.add_argument("--excluded_nums_file", type=str, default='')

  args = parser.parse_args()

  return (args.input_files, args.n, args.o, args.excluded_nums_file)

if __name__ == '__main__':
  arguments = parse_arguments()
  main(*arguments)


