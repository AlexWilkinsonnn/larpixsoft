import os, argparse, time

import ROOT
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import sparse

def get_high_res(event, MASK):
    arrZ = np.zeros((6, 3840, 35936))
    arrU = np.zeros((6, 6400, 35936))
    arrV = np.zeros((6, 6400, 35936))
    pixel_triggers = {}

    nonzero_indicesZ = set()
    nonzero_indicesU = set()
    nonzero_indicesV = set()

    for hit in event.projection:
        x = round(hit[0], 4) # beam direction
        y = round(hit[1], 4)
        z = hit[2] # drift direction
        chZ = int(hit[3])
        tickZ = int(hit[4])
        chU = int(hit[5])
        tickU = int(hit[6])
        chV = int(hit[7])
        tickV = int(hit[8])
        adc = int(hit[9])
        nd_drift = hit[10]
        fd_driftZ = hit[11]
        fd_driftU = hit[12]
        fd_driftV = hit[13]
        wire_distanceZ = hit[14]
        wire_distanceU = hit[15]
        wire_distanceV = hit[16]

        nonzero_indicesZ.add((chZ, tickZ))
        arrZ[0, chZ, tickZ] += adc
        arrZ[1, chZ, tickZ] += np.sqrt(nd_drift) * adc
        arrZ[2, chZ, tickZ] += np.sqrt(fd_driftZ) * adc
        if adc:
            arrZ[3, chZ, tickZ] += 1

        nonzero_indicesU.add((chU, tickU))
        arrU[0, chU, tickU] += adc
        arrU[1, chU, tickU] += np.sqrt(nd_drift) * adc
        arrU[2, chU, tickU] += np.sqrt(fd_driftU) * adc
        if adc:
            arrU[3, chU, tickU] += 1

        nonzero_indicesV.add((chV, tickV))
        arrV[0, chV, tickV] += adc
        arrV[1, chV, tickV] += np.sqrt(nd_drift) * adc
        arrV[2, chV, tickV] += np.sqrt(fd_driftV) * adc
        if adc:
            arrV[3, chV, tickV] += 1

        if (x, y) not in pixel_triggers:
            pixel_triggers[(x, y)] = {
                'Z' : (chZ, [tickZ]),
                'U' : (chU, [tickU]),
                'V' : (chV, [tickV]) }

        else:
            pixel_triggers[(x,y)]['Z'][1].append(tickZ)
            pixel_triggers[(x,y)]['U'][1].append(tickU)
            pixel_triggers[(x,y)]['V'][1].append(tickV)

    for i, j in nonzero_indicesZ:
        if arrZ[0][i, j] != 0:
            arrZ[1][i, j] /= arrZ[0][i, j]

    for i, j in nonzero_indicesZ:
        if arrZ[0][i, j] != 0:
            arrZ[2][i, j] /= arrZ[0][i, j]

    for i, j in nonzero_indicesU:
        if arrU[0][i, j] != 0:
            arrU[1][i, j] /= arrU[0][i, j]

    for i, j in nonzero_indicesU:
        if arrU[0][i, j] != 0:
            arrU[2][i, j] /= arrU[0][i, j]

    for i, j in nonzero_indicesV:
        if arrV[0][i, j] != 0:
            arrV[1][i, j] /= arrV[0][i, j]

    for i, j in nonzero_indicesV:
        if arrV[0][i, j] != 0:
            arrV[2][i, j] /= arrV[0][i, j]

    for pixel, trigger_data in pixel_triggers.items():
        ticksZ = sorted(trigger_data['Z'][1])
        first_triggersZ = [ tick for i, tick in enumerate(ticksZ) if i == 0 or tick - ticksZ[i - 1] > 15 ]
        for trigger_tick in first_triggersZ:
            arrZ[4, trigger_data['Z'][0], trigger_tick] += 1

        ticksU = sorted(trigger_data['U'][1])
        first_triggersU = [ tick for i, tick in enumerate(ticksU) if i == 0 or tick - ticksU[i - 1] > 15 ]
        for trigger_tick in first_triggersU:
            arrU[4, trigger_data['U'][0], trigger_tick] += 1

        ticksV = sorted(trigger_data['V'][1])
        first_triggersV = [ tick for i, tick in enumerate(ticksV) if i == 0 or tick - ticksV[i - 1] > 15 ]
        for trigger_tick in first_triggersV:
            arrV[4, trigger_data['V'][0], trigger_tick] += 1

    if MASK:
        arrZ_downres = np.zeros((480, 4492))
        arrU_downres = np.zeros((800, 4492))
        arrV_downres = np.zeros((800, 4492))

        start = time.time()

        for arr, arr_nonzero_indices, arr_downres in zip([arrZ[0], arrU[0], arrV[0]], [nonzero_indicesZ, nonzero_indicesU, nonzero_indicesV], [arrZ_downres, arrU_downres, arrV_downres]):
            for ch, tick in arr_nonzero_indices:
                arr_downres[int(ch/8), int(tick/8)] += arr[ch, tick]

        maskZ = get_nd_mask(arrZ_downres, 15, 1)
        maskU = get_nd_mask(arrU_downres, 25, 2)
        maskV = get_nd_mask(arrV_downres, 25, 2)

        maskZ = maskZ.astype(bool).astype(float)
        maskU = maskU.astype(bool).astype(float)
        maskV = maskV.astype(bool).astype(float)

        maskZ = np.pad(maskZ, ((0, 3360), (0, 31444)), mode='constant', constant_values=0)
        maskU = np.pad(maskU, ((0, 5600), (0, 31444)), mode='constant', constant_values=0)
        maskV = np.pad(maskV, ((0, 5600), (0, 31444)), mode='constant', constant_values=0)

        maskZ_nonzero = maskZ.nonzero()
        maskU_nonzero = maskU.nonzero()
        maskV_nonzero = maskV.nonzero()

        for i, j in zip(maskZ_nonzero[0], maskZ_nonzero[1]):
            arrZ[5, i, j] = maskZ[i, j]

        for i, j in zip(maskU_nonzero[0], maskU_nonzero[1]):
            arrU[5, i, j] = maskU[i, j]

        for i, j in zip(maskV_nonzero[0], maskV_nonzero[1]):
            arrV[5, i, j] = maskV[i, j]

    return arrZ, arrU, arrV

def get_nd_mask(arr_nd, max_tick_shift, max_ch_shift):
    nd_mask = np.copy(arr_nd)

    for tick_shift in range(1, max_tick_shift + 1):
            nd_mask[:, tick_shift:] += arr_nd[:, :-tick_shift]
            nd_mask[:, :-tick_shift] += arr_nd[:, tick_shift:]

    for ch_shift in range(1, max_ch_shift + 1):
            nd_mask[ch_shift:, :] += nd_mask[:-ch_shift, :]
            nd_mask[:-ch_shift, :] += nd_mask[ch_shift:, :]

    return nd_mask

def main(INPUT_FILE, N, OUTPUT_DIR, PLOT, MASK, HIGHRES, INFILLMASK, START_I, BUGFIX):
    f = ROOT.TFile.Open(INPUT_FILE, "READ")
    t = f.Get("IonAndScint/packet_projections")

    out_dir_Z = os.path.join(OUTPUT_DIR, 'Z')
    out_dir_U = os.path.join(OUTPUT_DIR, 'U')
    out_dir_V = os.path.join(OUTPUT_DIR, 'V')
    for dir in [out_dir_Z, out_dir_U, out_dir_V]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    tree_len = N if N else t.GetEntries()
    for i, event in enumerate(tqdm(t, total=tree_len)):
        if i < START_I:
            continue
        if N and i >= N + START_I:
            break

        id = event.eventid
        vertex_z = event.vertex[2]

        if HIGHRES: # NOTE this needs updated for the no gap data
            arrZ, arrU, arrV = get_high_res(event, MASK)

            SZ = sparse.COO.from_numpy(arrZ)
            SU = sparse.COO.from_numpy(arrU)
            SV = sparse.COO.from_numpy(arrV)

            sparse.save_npz(os.path.join(out_dir_Z, "ND_detsimZ_sparse_{}.npz".format(id)), SZ)
            sparse.save_npz(os.path.join(out_dir_U, "ND_detsimU_sparse_{}.npz".format(id)), SU)
            sparse.save_npz(os.path.join(out_dir_V, "ND_detsimV_sparse_{}.npz".format(id)), SV)

            continue

        arrZ = np.zeros((7, 480, 4492))
        arrU = np.zeros((7, 800, 4492))
        arrV = np.zeros((7, 800, 4492))
        pixel_triggers = {}

        for hit in event.projection:
            x = round(hit[0], 4) # beam direction
            y = round(hit[1], 4)
            z = hit[2] # drift direction
            chZ = int(hit[3])
            tickZ = int(hit[4])
            chU = int(hit[5])
            tickU = int(hit[6])
            chV = int(hit[7])
            tickV = int(hit[8])
            adc = int(hit[9])
            nd_drift = hit[10]
            fd_driftZ = hit[11]
            fd_driftU = hit[12]
            fd_driftV = hit[13]
            wire_distanceZ = hit[14]
            wire_distanceU = hit[15]
            wire_distanceV = hit[16]
            nd_module_x = round(hit[17], 4)

            if fd_driftZ < 0.0 or fd_driftU < 0.0 or fd_driftV < 0.0:
                print("FD drift is brokey somewhere")
                print(fd_driftZ, fd_driftU, fd_driftV)

            if chZ != -1:
                arrZ[0, chZ, tickZ] += adc
                arrZ[1, chZ, tickZ] += np.sqrt(nd_drift) * adc
                arrZ[2, chZ, tickZ] += np.sqrt(fd_driftZ) * adc
                if adc:
                    arrZ[3, chZ, tickZ] += 1
                arrZ[5, chZ, tickZ] += wire_distanceZ * adc
                arrZ[6, chZ, tickZ] = nd_module_x if abs(arrZ[6, chZ, tickZ]) < abs(nd_module_x) \
                                                  else arrZ[6, chZ, tickZ]

            if chU != -1:
                arrU[0, chU, tickU] += adc
                arrU[1, chU, tickU] += np.sqrt(nd_drift) * adc
                arrU[2, chU, tickU] += np.sqrt(fd_driftU) * adc
                if adc:
                    arrU[3, chU, tickU] += 1
                arrU[5, chU, tickU] += wire_distanceU * adc
                arrU[6, chU, tickU] = nd_module_x if abs(arrU[6, chU, tickU]) < abs(nd_module_x) \
                                                  else arrU[6, chU, tickU]

            if chV != -1:
                arrV[0, chV, tickV] += adc
                arrV[1, chV, tickV] += np.sqrt(nd_drift) * adc
                arrV[2, chV, tickV] += np.sqrt(fd_driftV) * adc
                if adc:
                    arrV[3, chV, tickV] += 1
                arrV[5, chV, tickV] += wire_distanceV * adc
                arrV[6, chV, tickV] = nd_module_x if abs(arrV[6, chV, tickV]) < abs(nd_module_x) \
                                                  else arrV[6, chV, tickV]

            if (x, y) not in pixel_triggers:
                pixel_triggers[(x, y)] = {
                    'Z' : (chZ, [tickZ]),
                    'U' : (chU, [tickU]),
                    'V' : (chV, [tickV]) }

            else:
                pixel_triggers[(x,y)]['Z'][1].append(tickZ)
                pixel_triggers[(x,y)]['U'][1].append(tickU)
                pixel_triggers[(x,y)]['V'][1].append(tickV)


        for i, j in zip(arrZ[1].nonzero()[0], arrZ[1].nonzero()[1]):
            if arrZ[0][i, j] != 0:
                arrZ[1][i, j] /= arrZ[0][i, j]

        for i, j in zip(arrZ[2].nonzero()[0], arrZ[2].nonzero()[1]):
            if arrZ[0][i, j] != 0:
                arrZ[2][i, j] /= arrZ[0][i, j]

        for i, j in zip(arrZ[5].nonzero()[0], arrZ[5].nonzero()[1]):
            if arrZ[0][i, j] != 0:
                arrZ[5][i, j] /= arrZ[0][i, j]

        for i, j in zip(arrU[1].nonzero()[0], arrU[1].nonzero()[1]):
            if arrU[0][i, j] != 0:
                arrU[1][i, j] /= arrU[0][i, j]

        for i, j in zip(arrU[2].nonzero()[0], arrU[2].nonzero()[1]):
            if arrU[0][i, j] != 0:
                arrU[2][i, j] /= arrU[0][i, j]

        for i, j in zip(arrU[5].nonzero()[0], arrU[5].nonzero()[1]):
            if arrU[0][i, j] != 0:
                arrU[5][i, j] /= arrU[0][i, j]

        for i, j in zip(arrV[1].nonzero()[0], arrV[1].nonzero()[1]):
            if arrV[0][i, j] != 0:
                arrV[1][i, j] /= arrV[0][i, j]

        for i, j in zip(arrV[2].nonzero()[0], arrV[2].nonzero()[1]):
            if arrV[0][i, j] != 0:
                arrV[2][i, j] /= arrV[0][i, j]

        for i, j in zip(arrV[5].nonzero()[0], arrV[5].nonzero()[1]):
            if arrV[0][i, j] != 0:
                arrV[5][i, j] /= arrV[0][i, j]

        for pixel, trigger_data in pixel_triggers.items():
            if trigger_data['Z'][0] != -1:
                ticksZ = sorted(trigger_data['Z'][1])
                first_triggersZ = [ tick for i, tick in enumerate(ticksZ) if i == 0 or tick - ticksZ[i - 1] > 15 ]
                for trigger_tick in first_triggersZ:
                    arrZ[4, trigger_data['Z'][0], trigger_tick] += 1

            if trigger_data['U'][0] != -1:
                ticksU = sorted(trigger_data['U'][1])
                first_triggersU = [ tick for i, tick in enumerate(ticksU) if i == 0 or tick - ticksU[i - 1] > 15 ]
                for trigger_tick in first_triggersU:
                    arrU[4, trigger_data['U'][0], trigger_tick] += 1

            if trigger_data['V'][0] != -1:
                ticksV = sorted(trigger_data['V'][1])
                first_triggersV = [ tick for i, tick in enumerate(ticksV) if i == 0 or tick - ticksV[i - 1] > 15 ]
                for trigger_tick in first_triggersV:
                    arrV[4, trigger_data['V'][0], trigger_tick] += 1

        # This doesn't make much sense currently for reasons:
        # Cant just smear along channsl becuase the gaps can be for one block of ticks or channels,
        # or for induction not follow any ticks of channels at all. Just the depos are not enough
        # to produce a solid mask.
        # Also there are depos in the active LAr module dont map to a pixel (just off the edge
        # still drifted I guess) so no-active volume depositions don't even fully define where
        # the infill needs to happen
        if INFILLMASK:
            infill_maskZ = np.zeros((480, 4492))
            for ch_tick in event.infillmaskz:
                ch = int(ch_tick[0])
                tick = int(ch_tick[1])

                infill_maskZ[ch, tick] += 1

            # infill_maskZ_original = np.copy(infill_maskZ)
            # for tick_shift in range(1, 15 + 1):
            #         infill_maskZ[:, tick_shift:] += infill_maskZ_original[:, :-tick_shift]
            #         infill_maskZ[:, :-tick_shift] += infill_maskZ_original[:, tick_shift:]

            infill_maskZ = infill_maskZ.astype(bool).astype(float)
            arrZ = np.concatenate((arrZ, np.expand_dims(infill_maskZ, axis=0)), 0)

            infill_maskU = np.zeros((800, 4492))
            for ch_tick in event.infillmasku:
                ch = int(ch_tick[0]) if not BUGFIX else int(ch_tick[0]) + 1600
                tick = int(ch_tick[1])

                infill_maskU[ch, tick] += 1

            # infill_maskU_original = np.copy(infill_maskU)
            # for tick_shift in range(1, 25 + 1):
            #         infill_maskU[:, tick_shift:] += infill_maskU_original[:, :-tick_shift]
            #         infill_maskU[:, :-tick_shift] += infill_maskU_original[:, tick_shift:]

            infill_maskU = infill_maskU.astype(bool).astype(float)
            arrU = np.concatenate((arrU, np.expand_dims(infill_maskU, axis=0)), 0)

            infill_maskV = np.zeros((800, 4492))
            for ch_tick in event.infillmaskv:
                ch = int(ch_tick[0]) if not BUGFIX else int(ch_tick[0]) + 800
                tick = int(ch_tick[1])

                infill_maskV[ch, tick] += 1

            # infill_maskV_original = np.copy(infill_maskV)
            # for tick_shift in range(1, 25 + 1):
            #         infill_maskV[:, tick_shift:] += infill_maskV_original[:, :-tick_shift]
            #         infill_maskV[:, :-tick_shift] += infill_maskV_original[:, tick_shift:]

            infill_maskV = infill_maskV.astype(bool).astype(float)
            arrV = np.concatenate((arrV, np.expand_dims(infill_maskV, axis=0)), 0)

        if MASK:
            maskZ = get_nd_mask(arrZ[0], 15, 1)
            maskU = get_nd_mask(arrU[0], 25, 2)
            maskV = get_nd_mask(arrV[0], 25, 2)

            maskZ = maskZ.astype(bool).astype(float)
            maskU = maskU.astype(bool).astype(float)
            maskV = maskV.astype(bool).astype(float)

            arrZ = np.concatenate((arrZ, np.expand_dims(maskZ, axis=0)), 0)
            arrU = np.concatenate((arrU, np.expand_dims(maskU, axis=0)), 0)
            arrV = np.concatenate((arrV, np.expand_dims(maskV, axis=0)), 0)

        # Plotting for validation
        if PLOT:
            for name, arr in zip(["arrZ", "arrU", "arrV"], [arrZ, arrU, arrV]):
                arr_adc = arr[0]
                arr_nddrift = arr[1]
                arr_fddrift = arr[2]
                arr_numpackets = arr[3]
                arr_pixeltriggers = arr[4]
                arr_wiredistance = arr[5]
                arr_ndmodulex = arr[6]
                arr_mask = arr[-1]

                plt.imshow(np.ma.masked_where(arr_adc == 0, arr_adc).T, cmap='jet', interpolation='none', aspect='auto')
                plt.title("{} ADC".format(name))
                plt.colorbar()
                plt.show()

                plt.imshow(np.ma.masked_where(arr_nddrift == 0.0, arr_nddrift).T, cmap='jet', interpolation='none', aspect='auto')
                plt.title("{} nd drift".format(name))
                plt.colorbar()
                plt.show()

                plt.imshow(np.ma.masked_where(arr_fddrift == 0.0, arr_fddrift).T, cmap='jet', interpolation='none', aspect='auto')
                plt.title("{} fd drift".format(name))
                plt.colorbar()
                plt.show()

                plt.imshow(np.ma.masked_where(arr_numpackets == 0, arr_numpackets).T, cmap='jet', interpolation='none', aspect='auto')
                plt.title("{} num packets".format(name))
                plt.colorbar()
                plt.show()

                plt.imshow(np.ma.masked_where(arr_pixeltriggers == 0, arr_pixeltriggers).T, cmap='jet', interpolation='none', aspect='auto')
                plt.title("{} num first pixel triggers".format(name))
                plt.colorbar()
                plt.show()

                plt.imshow(np.ma.masked_where(arr_wiredistance == 0, arr_wiredistance).T, cmap='jet', interpolation='none', aspect='auto')
                plt.title("{} wire distance".format(name))
                plt.colorbar()
                plt.show()

                plt.imshow(np.ma.masked_where(arr_ndmodulex == 0, arr_ndmodulex).T, cmap='jet', interpolation='none', aspect='auto')
                plt.title("{} ND x coord relative to drift volume".format(name))
                plt.colorbar()
                plt.show()

                plt.imshow(np.ma.masked_where(arr_mask == 0, arr_mask).T, cmap='jet', interpolation='none', aspect='auto')
                plt.title("{} signal mask from ND packets".format(name))
                plt.colorbar()
                plt.show()

                if INFILLMASK:
                    arr_infillmask = arr[-2]
                    plt.imshow(np.ma.masked_where((arr_adc - (arr_infillmask * 100)) == 0, (arr_adc - (arr_infillmask * 100))).T, cmap='jet', interpolation='none', aspect='auto')
                    plt.title("{} required infill mask from ND depositions in gaps".format(name))
                    plt.colorbar()
                    plt.show()

        SZ = sparse.COO.from_numpy(arrZ)
        SU = sparse.COO.from_numpy(arrU)
        SV = sparse.COO.from_numpy(arrV)

        sparse.save_npz(os.path.join(out_dir_Z, "ND_detsimZ_sparse_{}.npz".format(id)), SZ)
        sparse.save_npz(os.path.join(out_dir_U, "ND_detsimU_sparse_{}.npz".format(id)), SU)
        sparse.save_npz(os.path.join(out_dir_V, "ND_detsimV_sparse_{}.npz".format(id)), SV)

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file")

    parser.add_argument("-n", type=int, default=0)
    parser.add_argument("-i", type=int, default=0, help="starting index to allow for parallel processes")
    parser.add_argument("-o", type=str, default='', help="output folder name")
    parser.add_argument("--plot", action='store_true')
    parser.add_argument("--mask", action='store_true')
    parser.add_argument("--highRes", action='store_true', \
        help="WARNING: need to fix for use with nogaps data. \
              Use high resolution channel and tick \
              (currently assume factors of 8 better wire and tick resolution")
    parser.add_argument("--infillmask", action='store_true')
    parser.add_argument("--bugfix", action='store_true', \
        help="Used Z RID for the U and V channels for the infillmask branches, can correct this \
              here rather than reprocessing")
    args = parser.parse_args()

    return (args.input_file, args.n, args.o, args.plot, args.mask, args.highRes, args.infillmask, \
            args.i, args.bugfix)

if __name__ == '__main__':
    arguments = parse_arguments()

    if arguments[2] == '':
        raise Exception("Specify output directory")

    main(*arguments)

