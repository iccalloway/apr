##TODO:
# Possible rounding error??

import glob
import itertools
import json
import logging
import numpy as np
import os
import pickle
import re
import soundfile as sf
import torch
import tqdm

logging.basicConfig(
    filename="preprocess.log",
    level=logging.DEBUG,
    format="[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)

path = "/outside/Buckeyes/"
CHUNK_LENGTH = 5
STRIDE_LENGTH = 2
OUTPUT_LENGTH = 0.05
SR = 16000

acceptable_segments = [
    "aa",
    "ae",
    "ao",
    "aw",
    "ay",
    "eh",
    "ey",
    "ih",
    "iy",
    "oh",
    "ow",
    "oy",
    "uw",
    "uh",
    "ah",
    "er",
    "em",
    "el",
    "eng",
    "en",
    "aan",
    "aen",
    "ahn",
    "aon",
    "awn",
    "ayn",
    "ehn",
    "eyn",
    "ihn",
    "iyn",
    "ohn",
    "own",
    "uwn",
    "uhn",
    "p",
    "t",
    "k",
    "f",
    "th",
    "s",
    "sh",
    "hh",
    "ch",
    "tq",
    "b",
    "d",
    "g",
    "v",
    "dh",
    "z",
    "zh",
    "jh",
    "m",
    "n",
    "ng",
    "nx",
    "l",
    "r",
    "w",
    "y",
    "dx",
    "SIL",
    "VOCNOISE",
    "NOISE",
    "LAUGH",
]

output_path = "./output.tsv"


def process():
    b = 0
    c = 0
    with open(output_path, "w") as f:
        for wav in tqdm.tqdm(glob.glob("{}*.wav".format(path))):
            audio_list = []
            logging.debug("Processing... %s" % format(wav))
            phones = os.path.join(path, re.sub(".wav$", ".phones", wav))
            if os.path.isfile(phones):
                with open(phones, "r") as h:
                    lines = h.readlines()
                    processed = [line.split() for line in lines[11:-1]]

            sound, sr = sf.read(os.path.join(path, wav))
            chunked = np.lib.stride_tricks.sliding_window_view(
                sound, round(CHUNK_LENGTH * sr)
            )[:: round(STRIDE_LENGTH * sr)]
            for a in range(chunked.shape[0]):
                segs = []
                start = round(STRIDE_LENGTH * a)
                end = start + round(CHUNK_LENGTH)
                t = start
                for t in np.linspace(
                    start, end, round(CHUNK_LENGTH / OUTPUT_LENGTH), endpoint=False
                ):
                    try:
                        seg_id = min(
                            i for i, p in enumerate(processed) if float(p[0]) >= t
                        )
                    except:
                        segs = None
                        break
                    if len(processed[seg_id]) > 2:
                        segment_id = processed[seg_id][2].replace(";", "")
                        if segment_id in acceptable_segments:
                            segs.append(segment_id)
                        else:
                            logging.debug(
                                "{} not in acceptable segments".format(segment_id)
                            )
                            segs = None
                            break
                    else:
                        logging.debug("ending early because length off")
                        segs = None
                        break

                    t += OUTPUT_LENGTH

                if segs:
                    # logging.debug(start, end)
                    # logging.debug(len(segs))
                    # logging.debug(segs)
                    if len(segs) == round(CHUNK_LENGTH / OUTPUT_LENGTH):
                        f.write("{}\n".format(json.dumps(segs)))
                        audio_list.append(torch.tensor(chunked[a, :]))
                        # sf.write('./{}.wav'.format(c),chunked[a,:],sr)
                        c += 1

            if len(audio_list) > 0:
                final_tensor = torch.stack(audio_list)
                with open("input/{}.pkl".format(b), "wb") as g:
                    pickle.dump(final_tensor, g)
                    b += 1


def join_tensors():
    num_lines = sum(1 for line in open(output_path))
    print(num_lines)
    fp = np.memmap(
        "./input.bin",
        dtype="float64",
        mode="w+",
        shape=(num_lines, round(CHUNK_LENGTH * SR)),
    )
    ##Combine Individual Tensors
    a = 0
    for t in sorted(
        glob.glob("./input/*.pkl"),
        key=lambda x: int("".join([a for a in x if a.isdigit()])),
    ):
        print(t)
        logging.debug("Concatenating {}...".format(t))
        with open(t, "rb") as f:
            tens = pickle.load(f)
            logging.debug(tens)
            for b in range(tens.shape[0]):
                fp[a, :] = tens[b, :]
                a += 1


if __name__ == "__main__":
    # process()
    join_tensors()
