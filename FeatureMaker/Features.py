__author__ = 'DmitriyBrosalin'

from scipy.io.wavfile import read
from scikits.talkbox.features import mfcc
import os
import numpy as np
from tqdm import tqdm


class Features:
    def __init__(self, path_to_wave):
        self.path_wav_dir = path_to_wave


    def create_features(self):
        Whole_Data = []
        print("Starting processing data...")
        for name in tqdm(os.listdir(self.path_wav_dir)):
            if not ".DS_Store" in name:
                rate, data = read(self.path_wav_dir + "/" + name)
                ceps, mspec, spec = mfcc(data)
                num_ceps = len(ceps)
                cur = np.mean(ceps[int(num_ceps / 10):int(num_ceps * 9 / 10)], axis=0)
                bad_indices = np.where(np.isinf(cur))
                cur[bad_indices] = 3
                bad_indices = np.where(np.isnan(cur))
                cur[bad_indices] = 3
                Whole_Data.append(cur)
        print("Data processed")
        return np.array(Whole_Data)

    def create_vector_fetures(self, path):
        Whole_Data = []
        rate, data = read(path)
        ceps, mspec, spec = mfcc(data)
        num_ceps = len(ceps)
        cur = np.mean(ceps[int(num_ceps / 10):int(num_ceps * 9 / 10)], axis=0)
        bad_indices = np.where(np.isinf(cur))
        cur[bad_indices] = 3
        bad_indices = np.where(np.isnan(cur))
        cur[bad_indices] = 3
        Whole_Data.append(cur)
        print("Data processed")
        return np.array(Whole_Data)