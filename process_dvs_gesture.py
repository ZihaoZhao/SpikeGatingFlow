
import tarfile
import os
import h5py
import numpy as np
import struct
from events_timeslices import *
import cfg

def untar(fname, dirs):
    t = tarfile.open(fname)
    t.extractall(path=dirs)

def gather_aedat(directory, start_id, end_id, filename_prefix='user'):
    import glob
    fns = []
    for i in range(start_id, end_id):
        search_mask = directory + os.sep + \
            filename_prefix + "{0:02d}".format(i) + '*.aedat'
        # print(search_mask)
        glob_out = glob.glob(search_mask)
        if len(glob_out) > 0:
            fns += glob_out
    return fns

def aedat_to_events(filename):
    label_filename = filename[:-6] + '_labels.csv'
    labels = np.loadtxt(label_filename,
                        skiprows=1,
                        delimiter=',',
                        dtype='uint32')

    events = []
    with open(filename, 'rb') as f:

        for i in range(5):
            _ = f.readline()

        while True:
            data_ev_head = f.read(28)
            if len(data_ev_head) == 0:
                break

            eventtype = struct.unpack('H', data_ev_head[0:2])[0]
            eventsource = struct.unpack('H', data_ev_head[2:4])[0]
            eventsize = struct.unpack('I', data_ev_head[4:8])[0]
            eventoffset = struct.unpack('I', data_ev_head[8:12])[0]
            eventtsoverflow = struct.unpack('I', data_ev_head[12:16])[0]
            eventcapacity = struct.unpack('I', data_ev_head[16:20])[0]
            eventnumber = struct.unpack('I', data_ev_head[20:24])[0]
            eventvalid = struct.unpack('I', data_ev_head[24:28])[0]

            if (eventtype == 1):
                event_bytes = np.frombuffer(f.read(eventnumber * eventsize),
                                            'uint32')
                event_bytes = event_bytes.reshape(-1, 2)

                x = (event_bytes[:, 0] >> 17) & 0x00001FFF
                y = (event_bytes[:, 0] >> 2) & 0x00001FFF
                p = (event_bytes[:, 0] >> 1) & 0x00000001
                t = event_bytes[:, 1]
                events.append([t, x, y, p])

            else:
                f.read(eventnumber * eventsize)

    events = np.column_stack(events)
    events = events.astype('uint32')

    clipped_events = np.zeros([4, 0], 'uint32')

    for l in labels:
        start = np.searchsorted(events[0, :], l[1])
        end = np.searchsorted(events[0, :], l[2])
        clipped_events = np.column_stack([clipped_events,
                                          events[:, start:end]])

    return clipped_events.T, labels

def create_hdf5(path, save_path):
    print('processing train data...')
    save_path_train = os.path.join(save_path, 'train_label')
    if not os.path.exists(save_path_train):
        os.makedirs(save_path_train)

    fns_train = gather_aedat(path, 1, 24)
    
    for i in range(len(fns_train)):
        print('strat processing ' + str(i + 1) + ' train data')
        data, labels_starttime = aedat_to_events(fns_train[i])
        tms = data[:, 0]
        ads = data[:, 1:]
        lbls = labels_starttime[:, 0]
        start_tms = labels_starttime[:, 1]
        end_tms = labels_starttime[:, 2]

        for lbls_idx in range(len(lbls)):

            s_ = get_slice(tms, ads, start_tms[lbls_idx], end_tms[lbls_idx])
            times = s_[0]
            addrs = s_[1]
            file_name = save_path_train + os.sep + 'DVS-Gesture-train_' + str(lbls[lbls_idx]) + '_' + str(i) + '.hdf5'
            if not os.path.exists(file_name):
                file_name = file_name
            # else:
            #     file_name = save_path_train + os.sep + 'DVS-Gesture-train_' + str(lbls[lbls_idx]) + '_' + str(i) + '_2.hdf5'
            if lbls[lbls_idx] != 11:
                print(file_name)
                with h5py.File(file_name, 'w') as f:
                    tm_dset = f.create_dataset('times', data=times, dtype=np.uint32)
                    ad_dset = f.create_dataset('addrs', data=addrs, dtype=np.uint8)
                    lbl_dset = f.create_dataset('labels', data=lbls[lbls_idx] - 1, dtype=np.uint8)

    print('trainset process finish')

    print('processing test data...')
    save_path_test = os.path.join(save_path, 'test_label')
    if not os.path.exists(save_path_test):
        os.makedirs(save_path_test)

    fns_test = gather_aedat(path, 24, 30)

    for i in range(len(fns_test)):
        print('strat processing ' + str(i + 1) + ' test data')
        data, labels_starttime = aedat_to_events(fns_test[i])
        tms = data[:, 0]
        ads = data[:, 1:]
        lbls = labels_starttime[:, 0]
        start_tms = labels_starttime[:, 1]
        end_tms = labels_starttime[:, 2]

        for lbls_idx in range(len(lbls)):

            s_ = get_slice(tms, ads, start_tms[lbls_idx], end_tms[lbls_idx])
            times = s_[0]
            addrs = s_[1]            
            file_name = save_path_test + os.sep + 'DVS-Gesture-test_' + str(lbls[lbls_idx]) + '_' + str(i) + '.hdf5'
            if not os.path.exists(file_name):
                file_name = file_name
            # else:
            #     file_name = save_path_test + os.sep + 'DVS-Gesture-test_' + str(lbls[lbls_idx]) + '_' + str(i) + '_2.hdf5'

            print(file_name)
            if lbls[lbls_idx] != 11:
                print(file_name)
                with h5py.File(file_name, 'w') as f:
                    tm_dset = f.create_dataset('times', data=times, dtype=np.uint32)
                    ad_dset = f.create_dataset('addrs', data=addrs, dtype=np.uint8)
                    lbl_dset = f.create_dataset('labels', data=lbls[lbls_idx] - 1, dtype=np.uint8)

    test_data_filenames = os.listdir(save_path_test)
    for data_filename in test_data_filenames:
        if 'DVS-Gesture-test_11' in data_filename:
            os.remove(data_filename)  
    print('testset process finish')


def datasets_process(path=None):
    create_hdf5(os.path.join(path, 'DvsGesture'), path)


if __name__ == '__main__':
    datasets_process(path=cfg.data_path)