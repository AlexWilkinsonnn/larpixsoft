import os, argparse

import ROOT
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import sparse

def get_high_res_Z(event, MASK):
    arrZ = np.zeros((5, 1920, 44920))
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

        arrZ[0, chZ, tickZ] += adc
        arrZ[1, chZ, tickZ] += np.sqrt(nd_drift) * adc
        arrZ[2, chZ, tickZ] += np.sqrt(fd_driftZ) * adc
        if adc:
            arrZ[3, chZ, tickZ] += 1

        if (x, y) not in pixel_triggers:
            pixel_triggers[(x, y)] = { 'Z' : (chZ, [tickZ]) }
        else:
            pixel_triggers[(x,y)]['Z'][1].append(tickZ)

    for i, j in zip(arrZ[1].nonzero()[0], arrZ[1].nonzero()[1]):
        if arrZ[0][i, j] != 0:
            arrZ[1][i, j] /= arrZ[0][i, j]

    for i, j in zip(arrZ[2].nonzero()[0], arrZ[2].nonzero()[1]):
        if arrZ[0][i, j] != 0:
            arrZ[2][i, j] /= arrZ[0][i, j]

    for pixel, trigger_data in pixel_triggers.items():
        ticksZ = sorted(trigger_data['Z'][1])
        first_triggersZ = [ tick for i, tick in enumerate(ticksZ) if i == 0 or tick - ticksZ[i - 1] > 15 ] 
        for trigger_tick in first_triggersZ:
            arrZ[4, trigger_data['Z'][0], trigger_tick] += 1

    if MASK:
        maskZ = get_nd_mask(arrZ[0], 15, 1)

        maskZ = maskZ.astype(bool).astype(float)

        arrZ = np.concatenate((arrZ, np.expand_dims(maskZ, axis=0)), 0) 

    return arrZ

def get_nd_mask(arr_nd, max_tick_shift, max_ch_shift):
    nd_mask = np.copy(arr_nd)

    for tick_shift in range(1, max_tick_shift + 1):
            nd_mask[:, tick_shift:] += arr_nd[:, :-tick_shift]
            nd_mask[:, :-tick_shift] += arr_nd[:, tick_shift:]
        
    for ch_shift in range(1, max_ch_shift + 1):
            nd_mask[ch_shift:, :] += nd_mask[:-ch_shift, :]
            nd_mask[:-ch_shift, :] += nd_mask[ch_shift:, :]

    return nd_mask

def main(INPUT_FILE, N, OUTPUT_DIR, PLOT, MASK, HIGHRESZ):
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
        if N and i >= N:
            break

        id = event.eventid  
        vertex_z = event.vertex[2]

        if HIGHRESZ:
            arrZ = get_high_res_Z(event, MASK)
            SZ = sparse.COO.from_numpy(arrZ)
            sparse.save_npz(os.path.join(out_dir_Z, "ND_detsimZ_sparse_{}.npz".format(id)), SZ)
            continue

        arrZ = np.zeros((6, 512, 4608))
        arrU = np.zeros((6, 1024, 4608))
        arrV = np.zeros((6, 1024, 4608))
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

            if fd_driftZ <= 0.0 or fd_driftU <= 0.0 or fd_driftV <= 0.0:
                print("FD drift is brokey somewhere")
                print(fd_driftZ, fd_driftU, fd_driftV, sep=" -- ")

            arrZ[0, chZ + 16, tickZ + 58] += adc
            arrZ[1, chZ + 16, tickZ + 58] += np.sqrt(nd_drift) * adc
            arrZ[2, chZ + 16, tickZ + 58] += np.sqrt(fd_driftZ) * adc
            if adc:
                arrZ[3, chZ + 16, tickZ + 58] += 1
            arrZ[5, chZ + 16, tickZ + 58] += wire_distanceZ * adc

            arrU[0, chU + 112, tickU + 58] += adc
            arrU[1, chU + 112, tickU + 58] += np.sqrt(nd_drift) * adc
            arrU[2, chU + 112, tickU + 58] += np.sqrt(fd_driftU) * adc
            if adc:
                arrU[3, chU + 112, tickU + 58] += 1
            arrU[5, chU + 112, tickU + 58] += wire_distanceU * adc

            arrV[0, chV + 112, tickV + 58] += adc
            arrV[1, chV + 112, tickV + 58] += np.sqrt(nd_drift) * adc
            arrV[2, chV + 112, tickV + 58] += np.sqrt(fd_driftV) * adc
            if adc:
                arrV[3, chV + 112, tickV + 58] += 1
            arrV[5, chV + 112, tickV + 58] += wire_distanceV * adc

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
            ticksZ = sorted(trigger_data['Z'][1])
            first_triggersZ = [ tick for i, tick in enumerate(ticksZ) if i == 0 or tick - ticksZ[i - 1] > 15 ] 
            for trigger_tick in first_triggersZ:
                arrZ[4, trigger_data['Z'][0] + 16, trigger_tick + 58] += 1

            ticksU = sorted(trigger_data['U'][1])
            first_triggersU = [ tick for i, tick in enumerate(ticksU) if i == 0 or tick - ticksU[i - 1] > 15 ] 
            for trigger_tick in first_triggersU:
                arrU[4, trigger_data['U'][0] + 112, trigger_tick + 58] += 1

            ticksV = sorted(trigger_data['V'][1])
            first_triggersV = [ tick for i, tick in enumerate(ticksV) if i == 0 or tick - ticksV[i - 1] > 15 ] 
            for trigger_tick in first_triggersV:
                arrV[4, trigger_data['V'][0] + 112, trigger_tick + 58] += 1

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
            for name, arr in zip(["arrZ", "arrU", "arrV"], [arrZ[:, 16:-16, 58:-58], arrU[:, 16:-16, 112:-112], arrV[:, 16:-16, 112:-112]]):
                arr_adc = arr[0]
                arr_nddrift = arr[1]
                arr_fddrift = arr[2]
                arr_numpackets = arr[3]

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
    parser.add_argument("-o", type=str, default='', help="output folder name")
    parser.add_argument("--plot", action='store_true')
    parser.add_argument("--mask", action='store_true')
    parser.add_argument("--highResZ", action='store_true', help="testing the high res Z")

    args = parser.parse_args()

    return (args.input_file, args.n, args.o, args.plot, args.mask, args.highResZ)

if __name__ == '__main__':
    arguments = parse_arguments()

    if arguments[2] == '':
        raise Exception("Specify output directory")

    main(*arguments)

