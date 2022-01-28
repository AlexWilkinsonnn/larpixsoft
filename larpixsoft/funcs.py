import importlib, collections

import numpy as np

from larpixsoft.packet import DataPacket, TriggerPacket
from larpixsoft.track import Track

def get_wires(pitch, x_start):
  wires = { i : (i + 0.5)*pitch for i in range(480) }
  for ch, wire_x in wires.items():
    wires[ch] += x_start

  return wires

def get_events(packets, mc_packets_assn, tracks, geometry, detector, N=0, x_min_max=(0,0)):
  my_tracks, event_tracks = [], []
  track_ids = set()
  data_packets, event_data_packets = [], []
  n = 0

  for i, packet in enumerate(packets):
    if N and n >= N:
      break

    if packet['packet_type'] == 7 and event_data_packets:
      data_packets.append(event_data_packets.copy())
      event_data_packets.clear()

      for id in track_ids:
        event_tracks.append(Track(tracks[id], detector))
      my_tracks.append(event_tracks.copy())
      track_ids.clear()
      event_tracks.clear()

      n += 1

    elif packet['packet_type'] == 7 and not event_data_packets:
      trigger = TriggerPacket(packet)

    elif packet['packet_type'] == 0:
      p = DataPacket(packet, geometry, detector)
      p.add_trigger(trigger)

      if x_min_max != (0,0): # x cuts are active
        valid = True
        curr_track_ids = [ id for id in mc_packets_assn[i][0] if id != -1 ]

        for id in curr_track_ids:
          if id != -1:
            track = Track(tracks[id], detector)
            x_min = min([track.x_start, track.x_end])
            x_max = max([track.x_start, track.x_end])

            if x_min <= x_min_max[0] or x_max >= x_min_max[1]:
              valid = False
              break

        if valid:
          event_data_packets.append(p)
          for id in mc_packets_assn[i][0]:
            if id != -1:
              track_ids.add(id)

      else:
        event_data_packets.append(p)
        for id in mc_packets_assn[i][0]:
          if id != -1:
            track_ids.add(id)

  return data_packets, my_tracks

def get_wire_hits(event_data_packets, pitch, wires, tick_scaledown=10, projection_anode='lower_z'):
  wire_hits = []
  for p in event_data_packets:
    x = p.x + p.anode.tpc_x
    # print("min(wires_values())={}, min(wires.values()) - 0.5*pitch={}, max(wires.values())={}, max(wires.values()) + 0.5*pitch={}".format(
    #   min(wires.values()), min(wires.values()) - 0.5*pitch, max(wires.values()), max(wires.values()) + 0.5*pitch))
    if x <= min(wires.values()) - 0.5*pitch or x >= max(wires.values()) + 0.5*pitch:
      continue

    diffs = { ch : abs(x - wire_x) for ch, wire_x in wires.items() }

    # FD tick is 0.5us
    if projection_anode == 'lower_z':
      wire_hits.append({'ch' : min(diffs, key=diffs.get), 'tick' : round(p.project_lowerz()/tick_scaledown),
        'adc' : p.ADC})
    elif projection_anode == 'upper_z':
      wire_hits.append({'ch' : min(diffs, key=diffs.get), 'tick' : round(p.project_upperz()/tick_scaledown),
        'adc' : p.ADC})    
    else:
      raise NotImplementedError

  return wire_hits

def get_wire_trackhits(event_tracks, pitch, wires, tick_scaledown=10):
  wire_trackhits = []
  for track in event_tracks:
    segments = track.segments(0.04) # 0.0206) # 0.0824) # 0.1648cm is the smallest movement that moves into another pixel (one 1us tick)
    for segment in segments:
      x, y, z = segment['x'], segment['y'], segment['z']
      if x <= min(wires.values()) - 0.5*pitch or x >= max(wires.values()) + 0.5*pitch:
        continue

      diffs = { ch : abs(x - wire_x) for ch, wire_x in wires.items() }

      wire_trackhits.append({'ch' : min(diffs, key=diffs.get), 'tick' : round(track.drift_time_lowerz(z)/tick_scaledown),
        'charge' : segment['electrons']})

  return wire_trackhits 