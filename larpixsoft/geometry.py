import numpy as np
import yaml

def get_geom_map(pixel_file):
    """
    Returns map from electronics readout channels to x,y pixel position on the anode.
    geometry yaml: pixel layout yaml (multi_tile_layout-3.0.40.yaml for ND LAr)
    """
    with open(pixel_file) as pf:
        geometry_yaml = yaml.load(pf, Loader=yaml.FullLoader)

    geometry = { }

    pixel_pitch = geometry_yaml['pixel_pitch']
    chip_channel_to_position = geometry_yaml['chip_channel_to_position']
    tile_orientations = geometry_yaml['tile_orientations']
    tile_positions = geometry_yaml['tile_positions']
    xs = np.array(list(chip_channel_to_position.values()))[:, 0]*pixel_pitch
    ys = np.array(list(chip_channel_to_position.values()))[:, 1]*pixel_pitch
    x_size = max(xs) - min(xs) + pixel_pitch
    y_size = max(ys) - min(ys) + pixel_pitch

    for tile in geometry_yaml['tile_chip_to_io']:
            tile_orientation = tile_orientations[tile]
            for chip_channel in geometry_yaml['chip_channel_to_position']:
                    chip = chip_channel//1000
                    channel = chip_channel%1000
                    try:
                            io_group_io_channel = geometry_yaml['tile_chip_to_io'][tile][chip]
                    except KeyError:
                            print("Chip %i on tile %i not present in network" % (chip,tile))
                            continue

                    io_group = io_group_io_channel//1000
                    io_channel = io_group_io_channel%1000
                    x = chip_channel_to_position[chip_channel][0]*pixel_pitch + pixel_pitch/2 - x_size/2
                    y = chip_channel_to_position[chip_channel][1]*pixel_pitch + pixel_pitch/2 - y_size/2

                    x, y = x*tile_orientation[2], y*tile_orientation[1]
                    x += tile_positions[tile][2]
                    y += tile_positions[tile][1]
                    x /= 10 # to cm
                    y /= 10

                    geometry[(io_group, io_channel, chip, channel)] = x, y

    return geometry

def get_tpc_centres(det_yaml):
    tpc_centres = det_yaml['tpc_offsets']

    return tpc_centres
