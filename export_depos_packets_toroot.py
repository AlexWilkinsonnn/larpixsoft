import os, argparse
from array import array

import ROOT, h5py
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

from larpixsoft.detector import set_detector_properties
from larpixsoft.geometry import get_geom_map

from larpixsoft.funcs import get_events_no_cuts

def main(INPUT_FILES, OUTPUT_NAME, PLOT):
    detector = set_detector_properties('data/detector/ndlar-module.yaml', \
                                       'data/pixel_layout/multi_tile_layout-3.0.40.yaml', \
                                       pedestal=74)
    geometry = get_geom_map('data/pixel_layout/multi_tile_layout-3.0.40.yaml')

    ROOT.gROOT.ProcessLine('#include<vector>')
    f_ROOT = ROOT.TFile.Open(OUTPUT_NAME if OUTPUT_NAME != '' else 'out.root', "RECREATE")
    t = ROOT.TTree("ND_depos_packets", "nddepospackets")

    depos = ROOT.vector("std::vector<double>")()
    t.Branch("nd_depos", depos)
    packets = ROOT.vector("std::vector<double>")()
    t.Branch("nd_packets", packets)
    vertex_info = ROOT.vector("double")(4)
    t.Branch("vertex", vertex_info)
    eventID = array('i', [0])
    t.Branch("eventID", eventID, 'eventID/I')

    n_passed, num = 0, 0
    n_adc_failed, n_assns_failed = 0, 0
    for input_file in INPUT_FILES:
        f = h5py.File(input_file, 'r')

        vertices = { vertex['eventID'] : \
                    (vertex['z_vert']/10.0, vertex['y_vert']/10.0, vertex['x_vert']/10.0, 0.0) \
                    for vertex in f['vertices'] }

        data_packets, tracks = get_events_no_cuts(f['packets'], f['mc_packets_assn'], f['tracks'], geometry, detector)

        for i, (event_data_packets, event_tracks) in enumerate(tqdm(zip(data_packets, tracks))):
            depos.clear()
            packets.clear()
            for i in range(vertex_info.size()):
                vertex_info[i] = -9999.0
            eventID[0] = -1

            ids = { track.eventid for track in event_tracks }

            if len(ids) > 1:
                print("ids = {}".format(ids))
                raise Exception("Packets should be for a single event")

            id = ids.pop()
            eventID[0] = id

            vertex = vertices[id]

            # Write vertex info
            vertex_info[0] = vertex[0] # x
            vertex_info[1] = vertex[1] # y
            vertex_info[2] = vertex[2] # z
            vertex_info[3] = vertex[3] # t

            # Write depos
            total_e = 0
            for track in event_tracks:
                depo = ROOT.vector("double")(13)
                depo[0] = track.trackid
                depo[1] = track.pdg
                depo[2] = track.x_start
                depo[3] = track.x_end
                depo[4] = track.y_start
                depo[5] = track.y_end
                depo[6] = track.z_start
                depo[7] = track.z_end
                depo[8] = track.t_start
                depo[9] = track.t_end
                depo[10] = track.electrons
                depo[11] = track.dE
                depo[12] = 1.0 if track.active_volume else -1.0
                depos.push_back(depo)

            # Wrtie packets
            for p in event_data_packets:
                packet = ROOT.vector("double")(7)
                packet[0] = p.x + p.anode.tpc_x
                packet[1] = p.y + p.anode.tpc_y
                packet[2] = p.z_global()
                packet[3] = p.t()
                packet[4] = p.ADC
                packet[5] = p.z() # nd drift length
                packet[6] = p.x # nd module x for knowledge of when approacing edge of module
                packets.push_back(packet)

            if PLOT:
                vertex_x, vertex_y, vertex_z = [vertex[0]], [vertex[1]], [vertex[2]]

                depo_x, depo_y, depo_z = [], [], []
                for track in event_tracks:
                    depo_x.append(track.x_start)
                    depo_y.append(track.y_start)
                    depo_z.append(track.z_start)

                packet_x, packet_y, packet_z = [], [], []
                for p in event_data_packets:
                    packet_x.append(p.x + p.anode.tpc_x)
                    packet_y.append(p.y + p.anode.tpc_y)
                    packet_z.append(p.z_global())

                print("Packets:", len(packet_z), "Depos:",len(depo_z))
                fig = plt.figure()
                ax1 = fig.add_subplot(1, 2, 1, projection='3d')
                ax2 = fig.add_subplot(1, 2, 2, projection='3d')

                ax1.scatter(vertex_z, vertex_x, vertex_y, label='vertex', marker='x', s=40)
                ax1.scatter(packet_z, packet_x, packet_y, label='packet', marker='o', color='g')

                xlims = (413.72, 916.68)
                ylims = (-148.613, 155.387)
                zlims = (-356.7, 356.7)

                ax1.set_xlabel('Z')
                ax1.set_ylabel('X')
                ax1.set_zlabel('Y')
                ax1.set_xlim(zlims[0] - 50, zlims[1] + 50)
                ax1.set_ylim(xlims[0] - 50, xlims[1] + 50)
                ax1.set_zlim(ylims[0] - 50, ylims[1] + 50)
                ax1.legend(loc="lower left")

                ax2.scatter(vertex_z, vertex_x, vertex_y, label='vertex', marker='x', s=40)
                ax2.scatter(depo_z, depo_x, depo_y, label='depo', marker='o', color='r')

                ax2.set_xlabel('Z')
                ax2.set_ylabel('X')
                ax2.set_zlabel('Y')
                ax2.set_xlim(zlims[0] - 50, zlims[1] + 50)
                ax2.set_ylim(xlims[0] - 50, xlims[1] + 50)
                ax2.set_zlim(ylims[0] - 50, ylims[1] + 50)
                ax2.legend(loc="lower left")

                lines = [
                    ((xlims[0], xlims[1]), (zlims[0],) * 2, (ylims[0],) * 2), # left low /
                    ((xlims[0],) * 2, (zlims[0], zlims[1]), (ylims[0],) * 2), # low front -
                    ((xlims[0],) * 2, (zlims[0],) * 2, (ylims[0], ylims[1])), # left front |
                    ((xlims[0], xlims[1]), (zlims[0],) * 2, (ylims[1],) * 2), # left up /
                    ((xlims[0],) * 2, (zlims[0], zlims[1]), (ylims[1],) * 2), # up front -
                    ((xlims[1],) * 2, (zlims[0], zlims[1]), (ylims[1],) * 2), # up back -
                    ((xlims[1],) * 2, (zlims[0],) * 2, (ylims[1], ylims[0])), # left back |
                    ((xlims[0],) * 2, (zlims[1],) * 2, (ylims[0], ylims[1])), # right front |
                    ((xlims[0], xlims[1]), (zlims[1],) * 2, (ylims[0],) * 2), # right low /
                    ((xlims[1],) * 2, (zlims[1],) * 2, (ylims[0], ylims[1])), # right back |
                    ((xlims[1],) * 2, (zlims[1], zlims[0]), (ylims[0],) * 2), # low back -
                    ((xlims[0], xlims[1]), (zlims[1],) * 2, (ylims[1],) * 2)  # right up /
                ]
                for line in lines:
                    # Need to swap x and z from how we did it earlier because other coords are in
                    # edep-sim convention and these are in ND convention
                    ax1.plot(
                        line[1], line[0], zs=line[2], color='black', label='_', linestyle='dashed'
                    )
                    ax2.plot(
                        line[1], line[0], zs=line[2], color='black', label='_', linestyle='dashed'
                    )

                fig.tight_layout()
                plt.show()

            t.Fill()

    f_ROOT.Write()
    f_ROOT.Close()

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_files", nargs='+')

    parser.add_argument("-o", type=str, default='', help="output root file name")
    parser.add_argument("--plot", action='store_true')

    args = parser.parse_args()

    return (args.input_files, args.o, args.plot)

if __name__ == '__main__':
    arguments = parse_arguments()

    main(*arguments)
