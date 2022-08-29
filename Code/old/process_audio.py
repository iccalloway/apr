import os
import re

path = '/home/icalloway/Side Projects/AutoTranscribe/Data/'
CHUNK_LENGTH = 1 
acceptable_segments = [
            'aa', 'ae', 'ao','aw', 'ay', 'eh', 'ey', 'ih', 'iy', 'oh', 'ow','oy', 'uw', 'uh', 'ah',
            'er','em','el','eng','en','aan','aen', 'ahn','aon', 'awn','ayn','ehn','eyn','ihn','iyn','ohn','own','uwn','uhn',
            'p', 't', 'k','f','th','s','sh','hh','ch','tq',
            'b','d','g','v','dh','z','zh','jh','m','n','ng','nx','l', 'r', 'w','y','dx',
            'SIL','VOCNOISE', 'NOISE','LAUGH']

wav_files = [f for f in os.listdir(path) if f.endswith('.wav')]
missing=set([])
for wav in wav_files[:2]:
    print("Processing... %s" % format(wav))
    phones = os.path.join(path,re.sub('.wav$', '.phones', wav))
    if os.path.isfile(phones):
        with open(phones, 'r') as f:
            lines = f.readlines()
            processed = [line.split() for line in lines[11:-1]]
    file_length = float(processed[-1][0])
    time = 0
    max_ind = -1 
    final_phones = []
    while time + CHUNK_LENGTH < file_length:
        chunk = [a for a in range(len(processed)) if (float(processed[a][0]) > time and float(processed[a][0]) <= (time+CHUNK_LENGTH))]
        
        if len(chunk) > 0:
            max_ind = chunk[-1]
        if CHUNK_LENGTH + time < file_length:
            if len(chunk) == 0 or float(processed[chunk[-1]][0]) < CHUNK_LENGTH + time:
                chunk.append(max_ind+1)
            stretch = [] 
            acceptable = True
            for a in range(len(chunk)):
                if len(processed[chunk[a]]) > 2:
                    segment_id = processed[chunk[a]][2].replace(';','')
                    if segment_id in acceptable_segments:
                        start = time if a==0 else float(processed[chunk[a-1]][0])
                        end = min(time+CHUNK_LENGTH, float(processed[chunk[a]][0]))
                        stretch.append([segment_id,start,end])
                    else:
                        missing.add(segment_id)
                        acceptable=False
                        break
                else:
                    acceptable=False
                    break
            if acceptable:
                print(phones)
                print(stretch)
                ##process wav_file
                ##create transcription_info
                #print(stretch)
                pass
        time = time+CHUNK_LENGTH

print(missing)

