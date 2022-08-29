##TODO:
# Possible rounding error??

import glob
import itertools
import numpy as np
import os
import pickle
import re
import soundfile as sf
#import torch

# path = '/home/icalloway/Side Projects/AutoTranscribe/Data/'
path = "/outside/Data/"
CHUNK_LENGTH = 1
STRIDE_LENGTH = 2
OUTPUT_LENGTH = 0.05
segment_list = []
in_ = None

wav_files = glob.glob("/home/icalloway/TIMIT/TRAIN/*/*/*.WAV")

for wav in wav_files[:50]:
    audio_list = []
    print("Processing... %s" % format(wav))
    phones = os.path.join(wav, re.sub(".WAV$", ".PHN", wav))
    print(phones)
    if os.path.isfile(phones):
        with open(phones, "r") as f:
            processed = [a.strip().split(' ') for a in f.readlines()]
    sound, sr = sf.read(wav)
    print(sound)
    print(sr)
    print(round(sr*CHUNK_LENGTH))
    print(len(sound))
    chunked = np.lib.stride_tricks.sliding_window_view(sound, round(CHUNK_LENGTH * sr))[
        :: round(STRIDE_LENGTH * sr)
    ]
    for a in range(chunked.shape[0]):
        segs = []
        start = round(STRIDE_LENGTH * a)
        end = start + round(CHUNK_LENGTH)
        chunk = [
            b
            for b in range(len(processed))
            if (float(processed[b][0]) > start and float(processed[b][0]) <= end)
        ]
        print(chunk)
        exit()
        if len(chunk) < 1:
            continue
        else:
            last = max(chunk)
        if last < len(processed) - 1:
            chunk.append(last + 1)
        unit = start
exit()

'''
        for i, segment in enumerate(chunk):
            if len(processed[segment]) > 2:
                segment_id = processed[segment][2].replace(";", "")
                if segment_id in acceptable_segments:
                    seg_start = start if i == 0 else float(processed[chunk[i - 1]][0])
                    seg_end = min(end, float(processed[segment][0]))
                    while unit <= seg_end:
                        segs.append(segment_id)
                        unit += OUTPUT_LENGTH
                else:
                    segs = []
                    break
            else:
                segs = []
                break
        if len(segs) == round(CHUNK_LENGTH / OUTPUT_LENGTH):
            audio_list.append(chunked[a, :])
            segment_list.append(segs)

    if in_ == None:
        in_ = torch.tensor(np.array(audio_list))
    else:
        in_ = torch.concat((in_, torch.tensor(np.array(audio_list))))


in_ = torch.tensor(np.array(audio_list))


def convert_to_tensor(l):
    unique_segments = set(list(itertools.chain(*l)))
    mapping = dict(zip(unique_segments, range(len(unique_segments))))
    if len(l) > 0:
        out = torch.zeros((len(l), len(l[0])))
        for slice in range(out.shape[0]):
            for segment in range(out.shape[1]):
                out[slice, segment] = mapping[l[slice][segment]]
        print(out)
        return out, mapping
    else:
        raise RuntimeError("Segment list must be non-empty")


"""
def convert_to_tensor(l):
    unique_segments = set(list(itertools.chain(*l)))
    mapping = dict(zip(unique_segments, range(len(unique_segments))))
    if len(l) > 0:
        out = torch.zeros((len(l), len(l[0]), len(unique_segments)))
        for slice in range(out.shape[0]):
            for segment in range(out.shape[1]):
                out[slice,segment,mapping[l[slice][segment]]] = 1
        return out, mapping
    else:
        raise RuntimeError("Segment list must be non-empty")
"""

out, d = convert_to_tensor(segment_list)

with open("./mapping.pkl", "wb") as f:
    pickle.dump(d, f)

with open("./input.pkl", "wb") as g:
    pickle.dump(in_, g)

with open("./output.pkl", "wb") as h:
    pickle.dump(out, h)
'''
