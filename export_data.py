import os, argparse, sys, importlib, collections

import h5py
import numpy as np
from matplotlib import pyplot as plt

from larpixsoft.detector import set_detector_properties
from larpixsoft.geometry import get_geom_map

from larpixsoft.funcs import get_wires, get_events, get_wire_hits, get_events_vertex_cuts, get_wire_segmenthits, get_num_cols_to_wire

# NOTE move away from importing classes and functions and just use the module name space
#     eg. `import larpixsoft.funcs as funcs` then do funcs.get_wires

def main(INPUT_FILES, N, OUTPUT_DIR, EXCLUDED_NUMS_FILE, VERTICES_FILE, PEDESTAL, ND_ONLY, MORE_CHANNELS):
  detector = set_detector_properties('data/detector/ndlar-module.yaml', 
    'data/pixel_layout/multi_tile_layout-3.0.40.yaml', pedestal=PEDESTAL)
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

  vertices = {}
  with open(VERTICES_FILE, 'r') as f:
    for line in f:
      vals = line.split(',')
      vertices[int(vals[0])] = (float(vals[3])/10, float(vals[2])/10, float(vals[1])/10, float(vals[4])) # (x, y, z, t) in ND (z, y, x, t) in FD

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

  min_nonzero_fd_drifts, max_fd_drifts = [], [] 

  n_passed, num = 0, 0
  n_adc_failed, n_assns_failed = 0, 0
  for input_file in INPUT_FILES:
    f = h5py.File(input_file, 'r')

    wires = get_wires(pitch, x_start) 

    if MORE_CHANNELS:
      io_groups = [ i for i in range(1, 141) ]
      double_col_chs = get_num_cols_to_wire(pitch, wires, geometry, io_groups, detector)  

    # xmin_max = min/max wire x +- pitch/2 then tighten cuts by 5 wire pitches to remove the chance
    # of diffusion from tracks just outside the fake APA contributing to packets.
    data_packets, tracks, file_vertices, n_failed = get_events_vertex_cuts(f['packets'], f['mc_packets_assn'],
      f['tracks'], geometry, detector, vertices, ((479.7605 + 5*pitch), (709.92 - 5*pitch)), N=N) 
    n_assns_failed += n_failed
    # data_packets, tracks = get_events(f['packets'], f['mc_packets_assn'], f['tracks'],
    #   geometry, detector, N=N, x_min_max=((479.7605 + 5*pitch), (709.92 - 5*pitch))) 

    for i, (event_data_packets, event_tracks, vertex) in enumerate(zip(data_packets, tracks, file_vertices)):
      print("[{}/{}] - {} passed cuts: {} failed adc cut {} failed get_events".format(
        i + 1, len(data_packets), n_passed, n_adc_failed, n_assns_failed), end='\r', flush=True)

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
        raise Exception("z cut")
      
      if x_min <= 479.7605 or x_max >= 709.92:
        num += 1
        raise Exception("x cut")

      total_adc = sum([ packet.ADC for packet in event_data_packets ])
      if total_adc < 500:
        num += 1
        n_adc_failed += 1
        continue

      # Keep ND 0.1us tick for now, don't do the rounding
      wire_hits = get_wire_hits(event_data_packets, pitch, wires, tick_scaledown=0, 
        projection_anode=projection_anode)

      vertex_tick = ((detector.get_zlims()[1] - vertex[2])/detector.vdrift)*(1/detector.time_sampling)
      tick_shift = 2000*5 - vertex_tick
      for hit in wire_hits:
        hit['tick'] = round((hit['tick'] + tick_shift)/5) # shift and go to FD 0.5us tick
      if (max(wire_hits, key=lambda hit: hit['tick'])['tick'] >= 4492) or min(wire_hits, key=lambda hit: hit['tick'])['tick'] < 0:
        num += 1
        raise Exception("tick window cut")

      if MORE_CHANNELS:
        # ch0: adc, ch1: drift anode, ch2: drift upper/lower, ch3: num packets stacked, ch4: two pixel columns at this wire?
        arr_det = np.zeros((5, 512, 4608))
        for hit in wire_hits:
          fd_drift = (vertex[2] - hit['z_global']) + 163.705 # 163.705 is the x coordinate of the vertex in the FD for vertex alignment at 2000 ticks
          if fd_drift < 0:
            raise Exception("brokey")

          arr_det[0, hit['ch'] + 16, hit['tick'] + 58] += hit['adc']
          arr_det[1, hit['ch'] + 16, hit['tick'] + 58] += np.sqrt(hit['z_smalldrift'])*hit['adc']
          arr_det[2, hit['ch'] + 16, hit['tick'] + 58] += np.sqrt(fd_drift)*hit['adc']
          if hit['adc']:
            arr_det[3, hit['ch'] + 16, hit['tick'] + 58] += 1

        # Finishing weighted mean of drift distances
        for i, j in zip(arr_det[1].nonzero()[0], arr_det[1].nonzero()[1]):
          if arr_det[0][i, j] != 0:
            arr_det[1][i, j] /= arr_det[0][i, j]

        for i, j in zip(arr_det[2].nonzero()[0], arr_det[2].nonzero()[1]):
          if arr_det[0][i, j] != 0:
            arr_det[2][i, j] /= arr_det[0][i, j]

        for ch in double_col_chs:
          arr_det[4, ch + 16, 58:-58] = 1

      else:
        arr_det = np.zeros((1, 512, 4608))
        for hit in wire_hits:
          arr_det[0, hit['ch'] + 16, hit['tick'] + 58] += hit['adc']

      # Plot new channels to check they make sense
      # arr = arr_det[:, 16:-16, 58:-58]
      # arr_nddrift = arr[1]
      # arr_fddrift = arr[2]
      # arr_numpackets = arr[3]
      # arr_doublechs = arr[4]

      # plt.imshow(np.ma.masked_where(arr_nddrift == 0, arr_nddrift).T, cmap='jet', interpolation='none', aspect='auto')
      # plt.colorbar()
      # plt.show()

      # plt.imshow(np.ma.masked_where(arr_fddrift == 0, arr_fddrift).T, cmap='jet', interpolation='none', aspect='auto')
      # plt.colorbar()
      # plt.show()

      # plt.imshow(np.ma.masked_where(arr_numpackets == 0, arr_numpackets).T, cmap='jet', interpolation='none', aspect='auto')
      # plt.colorbar()
      # plt.show()

      # plt.imshow(arr_doublechs.T, cmap='jet', interpolation='none', aspect='auto')
      # plt.colorbar()
      # plt.show()

      # continue
        
      # Plot projected track segments alongside packets to check all looks good
      # event_segments = []
      # for track in event_tracks:
      #   event_segments.extend(track.segments(0.04, drift_time='upper'))  

      # for segment in event_segments:
      #   segment['drift_time_upperz'] += tick_shift

      # wire_segmenthits = get_wire_segmenthits(event_segments, pitch, wires, tick_scaledown=5, 
      #   projection_anode=projection_anode)

      # arr_true = np.zeros((1, 512, 4608))
      # for hit in wire_segmenthits:
      #   arr_true[0, hit['ch'] + 16, hit['tick'] + 58] += hit['charge']

      # fig, ax = plt.subplots(1,2,tight_layout=True)

      # diffs = { ch : abs(vertex[0] - wire_x) for ch, wire_x in wires.items() }
      # ch = min(diffs, key=diffs.get)
      # tick = round((vertex_tick + tick_shift)/5)
      # print(ch + 16, tick + 58)
      # arr_true[0, ch + 16, tick + 58] = -100000

      # pos = ax[0].imshow(np.ma.masked_where(arr_true == 0, arr_true)[0].T, interpolation='none', aspect='auto', cmap='viridis')
      # ax[0].set_xlabel("ch")
      # ax[0].set_ylabel("tick")

      # pos = ax[1].imshow(np.ma.masked_where(arr_det == 0, arr_det)[0].T, interpolation='none', aspect='auto', cmap='viridis')
      # ax[1].set_xlabel("ch")

      # plt.show()

      if np.min(arr_det) < 0:
        print(np.min(arr_det))
        print(np.argmin(arr_det))
        raise Exception("negative value in ND image")

      np.save(os.path.join(out_dir, "ND_detsim_{}.npy".format(num)), arr_det)

      min_nonzero_fd_drifts.append(np.min(arr_det[2][arr_det[2] != 0]))
      max_fd_drifts.append(np.max(arr_det[2]))

      if ND_ONLY:
        n_passed += 1
        num += 1
        continue

      with open(os.path.join(out_dir, "ND_depos_{}.txt".format(num)), 'w') as f:
        f.write("input_file:{},event_num:{},segment_length:{},view:{},projection_anode:{},".format(
          input_file, i, segment_length, 'Z', projection_anode))
        f.write("first_wire:{},last_wire:{},vtx_x:{},vtx_y:{},vtx_z:{},vtx_tick_anchor:{},".format(
          min(wires.values()), max(wires.values()), vertex[0], vertex[1], vertex[2], 2000))
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

    print("[{}/{}] - {} passed cuts: {} failed adc cut {} failed get_events".format(
      len(data_packets), len(data_packets), n_passed, n_adc_failed, n_assns_failed))  

  print("{} passed cuts : {} failed adc_cut {} failed get_events".format(
    n_passed, n_adc_failed, n_assns_failed))

  print("non_zero_min_fd_drift={}, max_fd_drift={}".format(min(min_nonzero_fd_drifts), max(max_fd_drifts)))
    
def parse_arguments():
  parser = argparse.ArgumentParser()

  parser.add_argument("input_files", nargs='+')

  parser.add_argument("-n", type=int, default=0)
  parser.add_argument("-o", type=str, default='', help='output dir name')
  parser.add_argument("--excluded_nums_file", type=str, default='')
  parser.add_argument("--vertices", type=str, default='')
  parser.add_argument("--ped", type=int, default=0)
  parser.add_argument("--nd_only", action='store_true')
  parser.add_argument("--more_channels", action='store_true',
    help='channels in ND image for ND drift length and FD drift length + for ND->FD downsampling info')

  args = parser.parse_args()

  return (args.input_files, args.n, args.o, args.excluded_nums_file, args.vertices, args.ped,
    args.nd_only, args.more_channels)

if __name__ == '__main__':
  arguments = parse_arguments()

  if arguments[4] == '':
    raise Exception("Provide file path of true vertices")

  main(*arguments)


