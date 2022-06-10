import os, argparse

import ROOT, h5py
from matplotlib import pyplot as plt
import numpy as np

from larpixsoft.detector import set_detector_properties
from larpixsoft.geometry import get_geom_map

from larpixsoft.funcs import get_events_vertex_cuts

def main(INPUT_FILES, N, OUTPUT_NAME, EXCLUDED_NUMS_FILE, VERTICES_FILE, PEDESTAL, SEGMENT_LENGTH, FAKE_FLUCTUATIONS):
  detector = set_detector_properties('data/detector/ndlar-module.yaml', 'data/pixel_layout/multi_tile_layout-3.0.40.yaml', pedestal=PEDESTAL)
  geometry = get_geom_map('data/pixel_layout/multi_tile_layout-3.0.40.yaml')

  pitchZ = 0.479
  pitchUV = 0.4669
  y_start = np.min(detector.tpc_borders[:,1,:])
  y_end = np.max(detector.tpc_borders[:,1,:])
  x_start = 480 - 0.5*pitchZ
  x_end = 480 + 480*pitchZ 
#   segment_length = 0.04 # The max step length for LArG4 in [cm]

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

  ROOT.gROOT.ProcessLine('#include<vector>')
  f_ROOT = ROOT.TFile.Open(OUTPUT_NAME if OUTPUT_NAME != '' else 'out.root', "RECREATE")
  t = ROOT.TTree("ND_depos_packets", "nddepospackets")

  depos = ROOT.vector("std::vector<double>")()
  t.Branch("nd_depos", depos)
  packets = ROOT.vector("std::vector<double>")()
  t.Branch("nd_packets", packets)
  vertex_info = ROOT.vector("double")(4)
  t.Branch("vertex", vertex_info)

  n_passed, num = 0, 0
  n_adc_failed, n_assns_failed = 0, 0
  for input_file in INPUT_FILES:
    f = h5py.File(input_file, 'r')

    # xmin_max = min/max wire x +- pitch/2 then tighten cuts by 5 wire pitches to remove the chance
    # of diffusion from tracks just outside the fake APA contributing to packets.
    data_packets, tracks, file_vertices, n_failed = get_events_vertex_cuts(f['packets'], f['mc_packets_assn'],
      f['tracks'], geometry, detector, vertices, ((x_start + 5*pitchZ), (x_end - 5*pitchZ)),
      y_min_max=((y_start + 5*pitchUV), (y_end - 5*pitchUV)), N=N) 
    n_assns_failed += n_failed
    # data_packets, tracks = get_events(f['packets'], f['mc_packets_assn'], f['tracks'],
    #   geometry, detector, N=N, x_min_max=((479.7605 + 5*pitch), (709.92 - 5*pitch))) 

    for i, (event_data_packets, event_tracks, vertex) in enumerate(zip(data_packets, tracks, file_vertices)):
      print("[{}/{}] - {} passed cuts: {} failed adc cut {} failed get_events".format(i + 1, len(data_packets), n_passed, n_adc_failed, n_assns_failed), end='\r', flush=True)

      depos.clear()
      packets.clear()
      for i in range(vertex_info.size()):
        vertex_info[i] = -9999.0

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

      # cuts, these shouldn't be needed if get_events_vertex_cuts is working as intended
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

      # fig = plt.figure()
      # ax = fig.add_subplot(projection='3d')
      # vertex_x, vertex_y, vertex_z = [], [], []
      # depo_x, depo_y, depo_z = [], [], []
      # packet_x, packet_y, packet_z = [], [], []

      # Write vertex info
      vertex_info[0] = vertex[0] # x
      vertex_info[1] = vertex[1] # y
      vertex_info[2] = vertex[2] # z
      vertex_info[3] = vertex[3] # t
      # vertex_x.append(vertex[0])
      # vertex_y.append(vertex[1])
      # vertex_z.append(vertex[2])

      # Write depos
      # total_e = 0
      for track in event_tracks:
        segments = track.segments(SEGMENT_LENGTH, equal_split=False, fake_fluctuations=FAKE_FLUCTUATIONS)

        # k_x = track.x_end - track.x_start
        # k_y = track.y_end - track.y_start
        # k_z = track.z_end - track.z_start

        # import math
        # line_length = math.sqrt(k_x**2 + k_y**2 + k_z**2)
        # print(line_length, k_x, k_y, k_z, sep=' - ')
        # print(track.dE)

        for segment in segments:
          depo = ROOT.vector("double")(12)
          depo[0] = track.trackid
          depo[1] = track.pdg
          depo[2] = segment['x_start']
          depo[3] = segment['x_end']
          depo[4] = segment['y_start']
          depo[5] = segment['y_end']
          depo[6] = segment['z_start']
          depo[7] = segment['z_end']
          depo[8] = segment['t_start']
          depo[9] = segment['t_end']
          depo[10] = segment['electrons']
          depo[11] = segment['dE']
          depos.push_back(depo)
          # total_e += segment['electrons']
          # depo_x.append(depo[2])
          # depo_y.append(depo[4])
          # depo_z.append(depo[6])

      # Wrtie packets
      # total_ADC = 0
      for p in event_data_packets:
        packet = ROOT.vector("double")(6)
        packet[0] = p.x + p.anode.tpc_x
        packet[1] = p.y + p.anode.tpc_y
        packet[2] = p.z_global()
        packet[3] = p.t()
        packet[4] = p.ADC 
        packet[5] = p.z() # nd drift length
        packets.push_back(packet)
        # total_ADC += p.ADC
        # packet_x.append(packet[0])
        # packet_y.append(packet[1])
        # packet_z.append(packet[2])

      # if (total_e/total_ADC < 800 or total_e/total_ADC > 1200):
      #   ax.scatter(vertex_x, vertex_y, vertex_z, label='vertex', marker='o')
      #   ax.scatter(depo_x, depo_y, depo_z, label='depo', marker='o')
      #   ax.scatter(packet_x, packet_y, packet_z, label='packet', marker='o')
      #   plt.show()
      # else:
      #   plt.close()

      t.Fill()

      n_passed += 1
      num += 1

    print("[{}/{}] - {} passed cuts: {} failed adc cut {} failed get_events".format(len(data_packets), len(data_packets), n_passed, n_adc_failed, n_assns_failed))  

  print("{} passed cuts : {} failed adc_cut {} failed get_events".format(n_passed, n_adc_failed, n_assns_failed))

  f_ROOT.Write()
  f_ROOT.Close()
 
def parse_arguments():
  parser = argparse.ArgumentParser()

  parser.add_argument("input_files", nargs='+')

  parser.add_argument("-n", type=int, default=0)
  parser.add_argument("-o", type=str, default='', help="output root file name")
  parser.add_argument("--excluded_nums_file", type=str, default='')
  parser.add_argument("--vertices", type=str, default='')
  parser.add_argument("--ped", type=int, default=0, help="ND has a 74 adc pedestal")
    help="channels in ND image for ND drift length and FD drift length + for ND->FD downsampling info")
  parser.add_argument("--segment_length", type=float, default=0.04,
    help="segment length to chop tracks into [cm]")
  parser.add_argument("--fake_fluctuations", type=float, default=0.0, 
    help="scale of Gaussian fluctuations if wanted")

  args = parser.parse_args()

  return (args.input_files, args.n, args.o, args.excluded_nums_file, args.vertices, args.ped,
    args.nd_only, args.more_channels, args.segment_length, args.fake_fluctuations)

if __name__ == '__main__':
  arguments = parse_arguments()

  if arguments[4] == '':
    raise Exception("Provide file path of true vertices")

  main(*arguments)
