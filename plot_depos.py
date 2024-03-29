import os, argparse

import h5py
from matplotlib import pyplot as plt

from larpixsoft.track import Track
from larpixsoft.detector import Detector

def main(INPUT_FILE, PRINT_DETECTORS, DETECTORS, SINGLE_SEGMENT_DSET):
    edep_data = h5py.File(INPUT_FILE)
    segment_keys = [ key for key in edep_data.keys() if key.startswith('segments_') ]

    if PRINT_DETECTORS:
        for key in segment_keys:
            print(key.split('segments_')[1])
        return

    dummyDetector = Detector()
    seen_eventIDs = set()
    for vertex in edep_data['vertices']:
        eventID = vertex['eventID']
        if eventID in seen_eventIDs:
            continue
        seen_eventIDs.add(eventID)
        print(eventID, vertex)

        vertex_x = vertex['x_vert'] / 10
        vertex_y = vertex['y_vert'] / 10
        vertex_z = vertex['z_vert'] / 10

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(vertex_x, vertex_z, vertex_y, marker='x', s=64)
        ax.set_xlabel('Z')
        ax.set_ylabel('X')
        ax.set_zlabel('Y')

        if DETECTORS:
            segment_keys = [ key for key in segment_keys if key.split('segments_')[1] in DETECTORS ]

        cmap = plt.cm.Set1
        ax.set_prop_cycle(color=[ cmap(i) for i in range(len(segment_keys)) ])

        if SINGLE_SEGMENT_DSET:
            segment_keys = (SINGLE_SEGMENT_DSET,)
            ax.set_prop_cycle(color=[cmap(0)])

        for key in segment_keys:
            colour = ''
            for iSegment, segment in enumerate(edep_data[key][edep_data[key]['eventID'] == eventID]):
                if iSegment == 0:
                    # Sometimes need to ensure nonzero \Delta to plot for short StepLimit edep-sim
                    track = ax.plot((segment['x_start'], segment['x_end']), \
                                    (segment['z_start'], segment['z_end']), \
                                    zs=(segment['y_start'], segment['y_end']), \
                                    label=key.split('segments_')[1] if not SINGLE_SEGMENT_DSET else '_')
                    colour = track[0].get_color()
                    continue

                # if segment['dx'] > 0.401:
                #     print(segment['dx'])

                ax.plot((segment['x_start'], segment['x_end']), \
                        (segment['z_start'], segment['z_end']), \
                        zs=(segment['y_start'], segment['y_end']), c=colour)

        xlims = (413.72, 916.68)
        ylims = (-148.613, 155.387)
        zlims = (-356.7, 356.7)

        lines = [ ((xlims[0], xlims[1]), (zlims[0],) * 2, (ylims[0],) * 2),  # left low /
                  ((xlims[0],) * 2, (zlims[0], zlims[1]), (ylims[0],) * 2),  # low front -
                  ((xlims[0],) * 2, (zlims[0],) * 2, (ylims[0], ylims[1])),  # left front |
                  ((xlims[0], xlims[1]), (zlims[0],) * 2, (ylims[1],) * 2),  # left up /
                  ((xlims[0],) * 2, (zlims[0], zlims[1]), (ylims[1],) * 2),  # up front -
                  ((xlims[1],) * 2, (zlims[0], zlims[1]), (ylims[1],) * 2),  # up back -
                  ((xlims[1],) * 2, (zlims[0],) * 2, (ylims[1], ylims[0])),  # left back |
                  ((xlims[0],) * 2, (zlims[1],) * 2, (ylims[0], ylims[1])),  # right front |
                  ((xlims[0], xlims[1]), (zlims[1],) * 2, (ylims[0],) * 2),  # right low /
                  ((xlims[1],) * 2, (zlims[1],) * 2, (ylims[0], ylims[1])),  # right back |
                  ((xlims[1],) * 2, (zlims[1], zlims[0]), (ylims[0],) * 2),  # low back -
                  ((xlims[0], xlims[1]), (zlims[1],) * 2, (ylims[1],) * 2) ] # right up /

        for line in lines:
            # Need to swap x and z from how we did it earlier because other coords are in edep-sim
            # convention and these are in ND convention
            ax.plot(line[1], line[0], zs=line[2], color='black', label='_', linestyle='dashed')

        ax.set_xlim(zlims[0] - 50, zlims[1] + 50)
        ax.set_ylim(xlims[0] - 50, xlims[1] + 50)
        ax.set_zlim(ylims[0] - 50, ylims[1] + 50)

        plt.legend(loc='lower left')
        fig.tight_layout()
        plt.show()

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file")

    parser.add_argument(
        "--print_detectors", action='store_true',
        help="print available detectors and exit"
    )
    parser.add_argument(
        "--single_segment_dset", type=str, default='',
        help=(
            "input file has a single semgment dataset with this name." +
            "USE THIS IS IF DATASET ONLY HAS A SINGLE 'segments'!"
        )
    )
    parser.add_argument(
        "-d", "--detectors", default=[], help="comma delimited list",
        type=lambda dets: [ det for det in dets.split(',') ]
    )

    args = parser.parse_args()

    return (args.input_file, args.print_detectors, args.detectors, args.single_segment_dset)

if __name__ == '__main__':
    arguments = parse_arguments()
    main(*arguments)

