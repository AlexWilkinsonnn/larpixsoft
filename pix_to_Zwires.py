import sys

import h5py
import numpy as np
from matplotlib import cm, colors
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from larpixsoft.detector import Detector, set_detector_properties
from larpixsoft.geometry import get_geom_map

from larpixsoft.funcs import get_events, get_wire_trackhits, get_wire_hits, get_wires

plt.rc('font', family='serif')

def plot_pix_wires(data_packets, wires, pitch, x_start, detector : Detector, as_pdf=False, save_array=False, wire_trace=False):
  for n, event_data_packets in enumerate(data_packets):
    if as_pdf:
      pdf = PdfPages('pix_Zwire{}.pdf'.format(n))

    wire_hits = get_wire_hits(event_data_packets, pitch, wires, x_start)

    fig, ax = plt.subplots(1,1,tight_layout=True)
    norm = colors.Normalize(vmin=0, vmax=256)
    cmap = cm.jet
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    ts = []
    for p in event_data_packets:
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
    elif save_array:
      plt.close()
    else:
      plt.show()

    fig, ax = plt.subplots(1,1,tight_layout=True)

    arr = np.zeros((480, 4492)) if not save_array else np.zeros((512, 4608))
    ts = []
    for hit in wire_hits:
      ts.append(hit['tick'])
      if save_array:
        arr[hit['ch'] + 16, hit['tick'] + 58] += hit['adc']
      else:
        arr[hit['ch'], hit['tick']] += hit['adc']

    ax.imshow(arr.T, interpolation='none', aspect='auto', cmap='jet')
    ax.set_xlabel("ch")
    ax.set_ylabel("tick")
    ax.set_ylim(min(ts) - 100, max(ts) + 100)
    if as_pdf:
      pdf.savefig(bbox_inches='tight')
      plt.close()
      pdf.close()
    elif save_array:
      np.save('pix_Zwire{}.npy'.format(n), arr)
      plt.close()
    else:
      plt.show()

    if wire_trace:
      ch = (0, 0)
      for i, col in enumerate(arr):
        if np.abs(col).sum() > ch[1]:
          ch = (i, np.abs(col).sum())
      ch = ch[0]

      ticks_adc = arr[ch, :]
      ticks = np.arange(1, arr.shape[1] + 1)

      fig, ax = plt.subplots(tight_layout=True)

      ax.hist(ticks, bins=len(ticks), weights=ticks_adc, histtype='step', linewidth=0.7, color='b')
      ax.set_ylabel("adc", fontsize=14)
      ax.set_xlabel("tick", fontsize=14)
      ax.set_xlim(min(ts) - 10, max(ts) + 10)
      ax.set_ylim(bottom=-5)
      plt.title("Channel {} in ROP".format(ch), fontsize=16)

      plt.show()

def plot_wires_det_true(data_packets, tracks, wires, pitch, x_start, detector : Detector, as_pdf=False, save_array=False, wire_trace=False):
  for n, (event_data_packets, event_tracks) in enumerate(zip(data_packets, tracks)):
    if as_pdf:
      pdf = PdfPages('pix_Zwire{}.pdf'.format(n))

    print(n, end ='\r')

    wire_hits = get_wire_hits(event_data_packets, pitch, wires, x_start)
    wire_trackhits = get_wire_trackhits(event_tracks, pitch, wires, x_start)

    ts = set()

    arr_det = np.zeros((480, 4492)) if not save_array else np.zeros((512, 4608))
    for hit in wire_hits:
      ts.add(hit['tick'])
      if save_array:
        arr_det[hit['ch'] + 16, hit['tick'] + 58] += hit['adc']
      else:
        arr_det[hit['ch'], hit['tick']] += hit['adc']

    arr_true = np.zeros((480, 4492)) if not save_array else np.zeros((512, 4608))
    for hit in wire_trackhits:
      ts.add(hit['tick'])
      if save_array:
        arr_true[hit['ch'] + 16, hit['tick'] + 58] += hit['charge']
      else:
        arr_true[hit['ch'], hit['tick']] += hit['charge']
    
    fig, ax = plt.subplots(1,2,tight_layout=True)

    pos = ax[0].imshow(np.ma.masked_where(arr_true == 0, arr_true).T, interpolation='none', aspect='auto', cmap='viridis')
    ax[0].set_xlabel("ch")
    ax[0].set_ylabel("tick")
    ax[0].set_ylim(min(ts) - 100, max(ts) + 100)

    pos = ax[1].imshow(np.ma.masked_where(arr_det == 0, arr_det).T, interpolation='none', aspect='auto', cmap='viridis')
    ax[1].set_xlabel("ch")
    ax[1].set_ylim(min(ts) - 100, max(ts) + 100)

    if as_pdf:
      pdf.savefig(bbox_inches='tight')
      plt.close()
      if not wire_trace:
        pdf.close()
    elif save_array:
      np.save('pix_Zwire_true{}.npy'.format(n), arr_true)
      np.save('pix_Zwire_det{}.npy'.format(n), arr_det)
      plt.close()
    else:
      plt.show()

    if wire_trace:
      ch = (0, 0)
      for i, col in enumerate(arr_true):
        if np.abs(col).sum() > ch[1]:
          ch = (i, np.abs(col).sum())
      ch = ch[0]

      ticks_charge = arr_true[ch, :]
      ticks_adc = arr_det[ch, :]
      ticks = np.arange(1, arr_true.shape[1] + 1)

      fig, ax = plt.subplots(tight_layout=True)

      ax.hist(ticks, bins=len(ticks), weights=ticks_adc, histtype='step', linewidth=0.7, color='b', label='ADC')
      ax.set_ylabel("ADC", fontsize=14)
      ax.set_xlabel("tick", fontsize=14)
      ax.set_xlim(min(ts) - 10, max(ts) + 10)
      ax.set_ylim(bottom=-5)

      ax2 = ax.twinx()
      ax2.hist(ticks, bins=len(ticks), weights=ticks_charge, histtype='step', linewidth=0.7, color='g', label='charge')
      ax2.set_ylabel("charge", fontsize=14)

      ax_ylims = ax.axes.get_ylim()
      ax_yratio = ax_ylims[0] / ax_ylims[1]
      ax2_ylims = ax2.axes.get_ylim()
      ax2_yratio = ax2_ylims[0] / ax2_ylims[1]
      if ax_yratio < ax2_yratio:
          ax2.set_ylim(bottom = ax2_ylims[1]*ax_yratio)
      else:
          ax.set_ylim(bottom = ax_ylims[1]*ax2_yratio)

      plt.title("Channel {} in ROP".format(ch), fontsize=16)

      handles, labels = ax.get_legend_handles_labels()
      handles2, labels2 = ax2.get_legend_handles_labels()
      handles += handles2
      labels += labels2
      new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles]
      plt.legend(handles=new_handles, labels=labels, prop={'size': 12})

      if as_pdf:
        pdf.savefig(bbox_inches='tight')
        plt.close()
        pdf.close()
      elif save_array:
        np.save('pix_Zwire_trace_true{}.npy'.format(n), arr_true)
        np.save('pix_Zwire_trace_det{}.npy'.format(n), arr_det)
        plt.close()
      else:
        plt.show()

if __name__ == '__main__':
  detector = set_detector_properties('data/detector/ndlar-module.yaml', 
    'data/pixel_layout/multi_tile_layout-3.0.40.yaml')
  geometry = get_geom_map('data/pixel_layout/multi_tile_layout-3.0.40.yaml')

  f = h5py.File('data/detsim/output_1_radi_numuCC.h5', 'r') # neutrino.0_1635125340.edep.larndsim.h5

  wires = get_wires(0.479, 480)
  data_packets, tracks = get_events(f['packets'], f['mc_packets_assn'], f['tracks'], geometry, detector, N=5)
  
  plot_pix_wires(data_packets, wires, 0.479, 480, detector, as_pdf=False, save_array=False, wire_trace=True)
  plot_wires_det_true(data_packets, tracks, wires, 0.479, 480, detector, wire_trace=True)