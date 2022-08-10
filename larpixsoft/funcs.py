import importlib, collections

import numpy as np
from tqdm import tqdm

from larpixsoft.packet import DataPacket, TriggerPacket
from larpixsoft.track import Track
from larpixsoft.anode import Anode

def get_wires(pitch, x_start):
    wires = { i : (i + 0.5)*pitch for i in range(480) }
    for ch, wire_x in wires.items():
        wires[ch] += x_start

    return wires

def get_events_no_cuts(packets, mc_packets_assn, tracks, geometry, detector):
    my_tracks = []
    data_packets = []

    event_tracks = set()
    event_data_packets = []

    cnt = 0
    for i, packet in enumerate(tqdm(packets)):
        if packet['packet_type'] == 7 and event_data_packets: # End of packet for current trigger.
            data_packets.append(list(event_data_packets)) # Explicit list to copy
            my_tracks.append(list(event_tracks))

            event_tracks.clear()
            event_data_packets.clear()

        elif packet['packet_type'] == 7: # Start of new trigger.
            trigger = TriggerPacket(packet)

        elif packet['packet_type'] == 0: # Data packet of current trigger.
            p = DataPacket(packet, geometry, detector, i)
            p.add_trigger(trigger)
            event_data_packets.append(p)

            track_ids = [ id for id in mc_packets_assn[i][0] if id != -1 ]
            for id in track_ids:
                track = Track(tracks[id], detector, id)
                event_tracks.add(track)

    if event_data_packets:
        data_packets.append(list(event_data_packets))
        my_tracks.append(list(event_tracks))

    return data_packets, my_tracks

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

def get_events_vertex_cuts(packets, mc_packets_assn, tracks, geometry, detector, vertices, x_min_max, y_min_max=(0,0), N=0):
    my_tracks = []
    data_packets = []
    my_vertices = []
    n, n_failed = 0, 0

    packet_tracks_assn = collections.defaultdict(list)
    track_packets_assn = collections.defaultdict(list)

    for i, packet in enumerate(packets):
        if N and n >= N:
            break

        if packet['packet_type'] == 7 and packet_tracks_assn: # End of packets for the new event
            # Get rid of the packets that link to cut tracks
            packet_tracks_assn_old_len = len(packet_tracks_assn)
            valid_tracks = set(track_packets_assn.keys())
            packet_tracks_assn = { p : tracks for p, tracks in packet_tracks_assn.items() if set(tracks).issubset(valid_tracks) }

            # Go back over tracks and get rid of any linked to the packets just cut.
            # Keep going over tracks and packets until only valid tracks <--> valid packets
            while (len(packet_tracks_assn) != packet_tracks_assn_old_len):
                packet_tracks_assn_old_len = len(packet_tracks_assn)
                valid_packets = set(packet_tracks_assn.keys())
                track_packets_assn = { track : ps for track, ps in track_packets_assn.items() if set(ps).issubset(valid_packets) }

                valid_tracks = set(track_packets_assn.keys())
                packet_tracks_assn = { p : tracks for p, tracks in packet_tracks_assn.items() if set(tracks).issubset(valid_tracks) }

            if len(packet_tracks_assn) != 0:
                data_packets.append(list(packet_tracks_assn.keys()))
                my_tracks.append(list(track_packets_assn.keys()))
                my_vertices.append(vertex)

                n += 1

            else:
                n_failed += 1

            packet_tracks_assn = collections.defaultdict(list)
            track_packets_assn = collections.defaultdict(list)

        elif packet['packet_type'] == 7 and not packet_tracks_assn: # Start of packets for new event
            trigger = TriggerPacket(packet)

        elif packet['packet_type'] == 0: # At new event
            p = DataPacket(packet, geometry, detector, i)
            p.add_trigger(trigger)

            # Get vertex using eventid of the first track related to this event
            if not packet_tracks_assn:
                curr_track_ids = [ id for id in mc_packets_assn[i][0] if id != -1 ]
                vertex = vertices[Track(tracks[curr_track_ids[0]], detector).eventid]

            curr_track_ids = [ id for id in mc_packets_assn[i][0] if id != -1 ]
            for id in curr_track_ids:
                if id != -1:
                    track = Track(tracks[id], detector, id)
                    x_min = min([track.x_start, track.x_end])
                    x_max = max([track.x_start, track.x_end])
                    z_min = min([track.z_start, track.z_end])
                    z_max = max([track.z_start, track.z_end])

                    # Store all packets and associated tracks
                    packet_tracks_assn[p].append(track)

                    if x_min <= x_min_max[0] or x_max >= x_min_max[1]:
                        continue

                    if z_min <= (vertex[2] - 150) or z_max >= (vertex[2] + 150):
                        continue

                    if y_min_max != (0,0):
                        y_min = min([track.y_start, track.y_end])
                        y_max = max([track.y_start, track.y_end])

                        if y_min <= y_min_max[0] or y_max >= y_min_max[1]:
                            continue

                    # Store track and associated packet if track passes cuts
                    track_packets_assn[track].append(p)

    return data_packets, my_tracks, my_vertices, n_failed

def get_wire_hits(event_data_packets, pitch, wires, tick_scaledown=10, projection_anode='lower_z'):
    wire_hits = []
    for p in event_data_packets:
        x = p.x + p.anode.tpc_x
        # print("min(wires_values())={}, min(wires.values()) - 0.5*pitch={}, max(wires.values())={}, max(wires.values()) + 0.5*pitch={}".format(
        #       min(wires.values()), min(wires.values()) - 0.5*pitch, max(wires.values()), max(wires.values()) + 0.5*pitch))
        if x <= min(wires.values()) - 0.5*pitch or x >= max(wires.values()) + 0.5*pitch:
            continue

        diffs = { ch : abs(x - wire_x) for ch, wire_x in wires.items() }
        # FD tick is 0.5us
        if projection_anode == 'lower_z':
            if tick_scaledown != 0:
                wire_hits.append({'ch' : min(diffs, key=diffs.get), 'tick' : round(p.project_lowerz()/tick_scaledown),
                    'adc' : p.ADC, 'z_smalldrift' : p.z(), 'z_global' : p.z_global()})
            else:
                wire_hits.append({'ch' : min(diffs, key=diffs.get), 'tick' : p.project_lowerz(), 'adc' : p.ADC,
                    'z_smalldrift' : p.z(), 'z_global' : p.z_global()})

        elif projection_anode == 'upper_z':
            if tick_scaledown != 0:
                wire_hits.append({'ch' : min(diffs, key=diffs.get), 'tick' : round(p.project_upperz()/tick_scaledown),
                    'adc' : p.ADC, 'z_smalldrift' : p.z(), 'z_global' : p.z_global()})
            else:
                wire_hits.append({'ch' : min(diffs, key=diffs.get), 'tick' : p.project_upperz(), 'adc' : p.ADC,
                    'z_smalldrift' : p.z(), 'z_global' : p.z_global()})

        else:
            raise NotImplementedError

    return wire_hits

def get_wire_trackhits(event_tracks, pitch, wires, tick_scaledown=10, projection_anode='lower_z'):
    wire_trackhits = []
    for track in event_tracks:
        segments = track.segments(0.04) # 0.0206) # 0.0824) # 0.1648cm is the smallest movement that moves into another pixel (one 1us tick)
        for segment in segments:
            x, y, z = segment['x'], segment['y'], segment['z']
            if x <= min(wires.values()) - 0.5*pitch or x >= max(wires.values()) + 0.5*pitch:
                continue

            diffs = { ch : abs(x - wire_x) for ch, wire_x in wires.items() }

            if projection_anode == 'lower_z':
                wire_trackhits.append({'ch' : min(diffs, key=diffs.get),
                    'tick' : round(track.drift_time_lowerz(z)/tick_scaledown), 'charge' : segment['electrons']})
            elif projection_anode == 'upper_z':
                wire_trackhits.append({'ch' : min(diffs, key=diffs.get),
                    'tick' : round(track.drift_time_upperz(z)/tick_scaledown), 'charge' : segment['electrons']})

    return wire_trackhits

def get_wire_segmenthits(event_segments, pitch, wires, tick_scaledown=10, projection_anode='lower_z'):
    wire_segmenthits = []
    for segment in event_segments:
        x, y, z = segment['x'], segment['y'], segment['z']
        if x <= min(wires.values()) - 0.5*pitch or x >= max(wires.values()) + 0.5*pitch:
            continue

        diffs = { ch : abs(x - wire_x) for ch, wire_x in wires.items() }

        if projection_anode == 'lower_z':
            wire_segmenthits.append({'ch' : min(diffs, key=diffs.get),
                'tick' : round(segment['drift_time_lowerz']/tick_scaledown), 'charge' : segment['electrons']})
        elif projection_anode == 'upper_z':
            wire_segmenthits.append({'ch' : min(diffs, key=diffs.get),
                'tick' : round(segment['drift_time_upperz']/tick_scaledown), 'charge' : segment['electrons']})

    return wire_segmenthits

def get_num_cols_to_wire(pitch, wires, geometry, io_groups, detector, verbose=False):
    x_values = set()
    for xy in geometry.values():
        x_values.add(xy[0])

    ch_num_cols_cnt = collections.Counter()

    for io_group in io_groups:
        module_id = (io_group - 1)//4
        io_group = io_group - ((io_group - 1)//4)*4

        anode = Anode(module_id, io_group, detector)
        if anode.tpc_z < 300: # only need to process one row of anodes.
            continue

        for x_value in sorted(x_values):
            x = x_value + anode.tpc_x
            if x <= min(wires.values()) - 0.5*pitch or x >= max(wires.values()) + 0.5*pitch:
                continue

            diffs = { ch : abs(x - wire_x) for ch, wire_x in wires.items() }

            ch_num_cols_cnt[min(diffs, key=diffs.get)] += 1

    if verbose:
        print(ch_num_cols_cnt)

        cnts_cntr = collections.Counter()
        for value in ch_num_cols_cnt.values():
            cnts_cntr[value] += 1
        print(cnts_cntr)

    cnts = set(ch_num_cols_cnt.values())
    if len(cnts) > 2:
        raise Exception("brokey")

    double_col_cnt = sorted(cnts)[1]
    single_col_cnt = sorted(cnts)[0]
    if double_col_cnt/single_col_cnt != 2:
        raise Exception("brokey")

    double_col_chs = [ ch for ch, cnt in ch_num_cols_cnt.items() if cnt == double_col_cnt ]

    return double_col_chs
