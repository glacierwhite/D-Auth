import random

import os
import soundfile as sf
import numpy as np
import time


def read_anno(anno_path):
    fi = open(anno_path)
    lines = fi.readlines()
    fi.close()
    file_ids, annos = [], []
    for line in lines:
        id, txt = line.split('\t')
        file_ids.append(id)
        annos.append(txt)
    return file_ids, annos


def keep_and_drop(conditions, keep_all_prob, drop_all_prob, drop_each_prob):
    results = []
    condition_weights = []
    seed = random.random()
    if seed < keep_all_prob:
        results = conditions
        for condition in conditions:
            condition_weights.append(np.ones(1))
    elif seed < keep_all_prob + drop_all_prob:
        for condition in conditions:
            results.append(np.zeros(condition.shape))
            condition_weights.append(np.zeros(1))
    else:
        for i in range(len(conditions)):
            if random.random() < drop_each_prob[i]:
                results.append(np.zeros(conditions[i].shape))
                condition_weights.append(np.zeros(1))
            else:
                results.append(conditions[i])
                condition_weights.append(np.ones(1))
    return results, condition_weights

def save_wave(waveform, savepath, name="outwav"):
    if type(name) is not list:
        name = [name] * waveform.shape[0]

    for i in range(waveform.shape[0]):
        path = os.path.join(
            savepath,
            "%s_%s.wav"
            % (
                os.path.basename(name[i])
                if (not ".wav" in name[i])
                else os.path.basename(name[i]).split(".")[0],
                i,
            ),
        )
        print("Save audio to %s" % path)
        sf.write(path, waveform[i, 0], samplerate=16000)
        
def get_time():
    t = time.localtime()
    return time.strftime("%d_%m_%Y_%H_%M_%S", t)