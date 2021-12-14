import sys

import h5py
import numpy as np
from matplotlib import cm, colors
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt

from larpixsoft.detector import Detector, set_detector_properties
from larpixsoft.packet import DataPacket, TriggerPacket
from larpixsoft.geometry import get_geom_map

plt.rc('font', family='serif')

def plot_tpc_borders(detector: Detector):
  fig, ax = plt.subplots(1,1,tight_layout=True, figsize=(12,12))
  for border in detector.tpc_borders:
    rect = plt.Rectangle((border[2,0], border[0,0]), border[2,1] - border[2,0],
      border[0,1] - border[0,0], linewidth=1, fill=False)
    ax.add_patch(rect)
  ax.text(-300,440,"Cathode", rotation='vertical', fontsize=16)
  ax.text(-350,440,"Anode", rotation='vertical', fontsize=16)
  ax.set_xlim(np.min(detector.tpc_borders[:,2,:]) - 5, np.max(detector.tpc_borders[:,2,:]) + 5)
  ax.set_ylim(np.min(detector.tpc_borders[:,0,:]) - 5, np.max(detector.tpc_borders[:,0,:]) + 5)
  ax.spines['top'].set_color('none')
  ax.spines['bottom'].set_color('none')
  ax.spines['left'].set_color('none')
  ax.spines['right'].set_color('none')
  ax.set_xlabel("z [cm]", fontsize=16)
  ax.set_ylabel("x [cm]", fontsize=16)
  # plt.savefig("ND_lar_top.pdf")
  # plt.close()
  plt.show()

def plot_evds(packets, geometry, detector : Detector, N=10, as_pdf=False):
  norm = colors.Normalize(vmin=0, vmax=256)
  cmap = cm.jet
  m = cm.ScalarMappable(norm=norm, cmap=cmap)

  n = 0
  data_packets = []
  for packet in packets:
    if n >= N:
      break

    if as_pdf:
      pdf = PdfPages('evd{}.pdf'.format(n))

    if packet['packet_type'] == 7 and data_packets:
      fig, ax = plt.subplots(1,1,tight_layout=True)

      for p in data_packets:
        rect = plt.Rectangle((p.x + p.anode.tpc_x, p.y + p.anode.tpc_y), detector.pixel_pitch,
          detector.pixel_pitch, fc=m.to_rgba(p.ADC))
        ax.add_patch(rect)
      
      for tpc in detector.tpc_borders:
        tpc_rect = plt.Rectangle((tpc[0][0],tpc[1][0]), 97.28, 304.0, linewidth=0.1, edgecolor='k',
          facecolor=cmap(0),zorder=-1)
        ax.add_patch(tpc_rect)

      ax.set_aspect("auto")
      ax.set_xlabel("x [cm]")
      ax.set_ylabel("y [Cm]")
      ax.add_patch(tpc_rect)
      ax.set_xlim(np.min(detector.tpc_borders[:,0,:]),np.max(detector.tpc_borders[:,0,:]))
      ax.set_ylim(np.min(detector.tpc_borders[:,1,:]),np.max(detector.tpc_borders[:,1,:]))
      if as_pdf:
        pdf.savefig(bbox_inches='tight')
        plt.close()
      else:
        plt.show()

      fig, ax = plt.subplots(1,1,tight_layout=True)

      ts = []
      for p in data_packets:
        ts.append(p.project_lowerz())
        rect = plt.Rectangle((p.x + p.anode.tpc_x, p.project_lowerz()), detector.pixel_pitch,
          -10, fc=m.to_rgba(p.ADC))
        ax.add_patch(rect)

      t_max = min([max(ts) + 100*10, min(ts) + 4492*10])
      t_min = min(ts) - 100*10
      
      for tpc in detector.tpc_borders:
        tpc_rect = plt.Rectangle((tpc[0][0], t_min), 97.28, 4492*10, linewidth=0.1, edgecolor='k',
          facecolor=cmap(0),zorder=-1)
        ax.add_patch(tpc_rect)

      ax.set_aspect("auto")
      ax.set_xlabel("x [cm]")
      ax.set_ylabel("t [us]")
      ax.add_patch(tpc_rect)
      ax.set_xlim(np.min(detector.tpc_borders[:,0,:]),np.max(detector.tpc_borders[:,0,:]))
      ax.set_ylim(t_min, t_max)
      if as_pdf:
        pdf.savefig(bbox_inches='tight')
        plt.close()
        pdf.close()
      else:
        plt.show()
      
      n += 1
      data_packets.clear()

    elif packet['packet_type'] == 7 and not data_packets:
      trigger = TriggerPacket(packet)

    elif packet['packet_type'] == 0:
      p = DataPacket(packet, geometry, detector)
      p.add_trigger(trigger)
      data_packets.append(p)

if __name__ == '__main__':
  detector = set_detector_properties('data/detector/ndlar-module.yaml', 
    'data/pixel_layout/multi_tile_layout-3.0.40.yaml')
  geometry = get_geom_map('data/pixel_layout/multi_tile_layout-3.0.40.yaml')

  f = h5py.File('data/detsim/output_1_radi.h5', 'r')

  plot_tpc_borders(detector)

  #plot_evds(f['packets'], geometry, detector, N=5, as_pdf=False)
