import os, argparse

import h5py
import numpy as np
from tqdm import tqdm

from larpixsoft.detector import set_detector_properties
from larpixsoft.geometry import get_geom_map
from larpixsoft.funcs import get_events_no_cuts

PACKETS_3D_DTYPE = np.dtype(
    [
        ("eventID", "u4"),
        ("adc", "f4"),
        ("x", "f4"), ("x_module", "f4"), ("y", "f4"), ("z", "f4"), ("z_module", "f4")
    ]
)


def main(args):
    detector = set_detector_properties(
        'data/detector/ndlar-module.yaml','data/pixel_layout/multi_tile_layout-3.0.40.yaml',
        pedestal=74
    )
    geometry = get_geom_map('data/pixel_layout/multi_tile_layout-3.0.40.yaml')

    in_f = h5py.File(args.input_file, 'r')

    packets, vertices = get_events_no_cuts(
        in_f["packets"][:10000], in_f["mc_packets_assn"][:10000], in_f["tracks"], geometry, detector,
        no_tracks=True, vertices=in_f["vertices"]
    )
    vertices = vertices[:len(packets)]

    with h5py.File(args.output_file, "w") as out_f:
        for key in in_f.keys():
            data = np.array(in_f[key])
            out_f.create_dataset(key, data=data)

        out_f.create_dataset("3d_packets", (0,), dtype=PACKETS_3D_DTYPE, maxshape=(None,))

        packets_3d_list = []
        for event_packets, event_vertex in zip(packets, vertices):
            event_packets_3d = np.empty(len(event_packets), dtype=PACKETS_3D_DTYPE)
            for i_p, p in enumerate(event_packets):
                event_packets_3d[i_p]["eventID"] = event_vertex.eventid
                event_packets_3d[i_p]["adc"] = p.ADC
                event_packets_3d[i_p]["x"] = p.x + p.anode.tpc_x
                event_packets_3d[i_p]["x_module"] = p.x
                event_packets_3d[i_p]["y"] = p.y + p.anode.tpc_y
                event_packets_3d[i_p]["z"] = p.z_global()
                event_packets_3d[i_p]["z_module"] = p.z()
            packets_3d_list.append(event_packets_3d)

        packets_3d = np.concatenate(packets_3d_list, axis=0)
        out_f["3d_packets"].resize((len(packets_3d),))
        out_f["3d_packets"][:] = packets_3d


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file")
    parser.add_argument("output_file")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_arguments()
    main(args)

