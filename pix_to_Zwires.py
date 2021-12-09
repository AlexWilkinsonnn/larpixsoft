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

def get_wires(pitch,  x_start):
  wires = { i : (i + 0.5)*pitch for i in range(480) }
  for ch, wire_x in wires.items():
    wires[ch] += x_start

  return wires

def plot_pix_wires(packets, wires, pitch, x_start, detector : Detector, N=5, as_pdf=False):
  n = 0

  data_packets = []
  for packet in packets:
    if n >= N:
      break

    if packet['packet_type'] == 7 and data_packets:
      if as_pdf:
        pdf = PdfPages('pix_Zwire{}.pdf'.format(n))

      wire_hits = []
      for p in data_packets:
        x = p.x + p.anode.tpc_x
        if x < x_start or x > max(wires.values()) + 0.5*pitch:
          continue

        diffs = { ch : abs(x - wire_x) for ch, wire_x in wires.items() }

        wire_hits.append({'ch' : min(diffs, key=diffs.get), 'tick' : round(p.project_lowerz()/10), 'adc' : p.ADC})

      fig, ax = plt.subplots(1,1,tight_layout=True)
      norm = colors.Normalize(vmin=0, vmax=256)
      cmap = cm.jet
      m = cm.ScalarMappable(norm=norm, cmap=cmap)

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
      ax.set_xlim(x_start, max(wires.values()) + 0.5*pitch)
      ax.set_ylim(t_min, t_max)
      if as_pdf:
        pdf.savefig(bbox_inches='tight')
        plt.close()
      else:
        plt.show()

      fig, ax = plt.subplots(1,1,tight_layout=True)

      arr = np.zeros((480, 4492))
      ts = []
      for hit in wire_hits:
        ts.append(hit['tick'])
        arr[hit['ch'], hit['tick']] = hit['adc']

      ax.imshow(arr.T, interpolation='none', aspect='auto', cmap='jet')
      ax.set_xlabel("ch")
      ax.set_ylabel("tick")
      ax.set_ylim(min(ts) - 100, max(ts) + 100)
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

  wires = get_wires(0.479, 480)
  plot_pix_wires(f['packets'], wires, 0.479, 480, detector, N=10, as_pdf=True)