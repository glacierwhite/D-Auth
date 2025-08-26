import os
import pandas as pd
import subprocess

metadata = pd.read_csv('./data/train.csv')

format = 'wav'
quality = 10
root_path = './data/AudioCaps/'

for i in range(0, metadata.shape[0]):
    audiocap_id = metadata['audiocap_id'][i]
    first_display_label = metadata['caption'][i]
    ytid = metadata['youtube_id'][i]
    start_seconds = metadata['start_time'][i]
    end_seconds = start_seconds + 10
    
#     subprocess.call(f'yt-dlp -x --audio-format {format} --audio-quality {quality} --output "{os.path.join(root_path, str(audiocap_id), first_display_label, ytid)}_{start_seconds}-{end_seconds}.%(ext)s" --postprocessor-args "-ss {start_seconds} -to {end_seconds}" https://www.youtube.com/watch?v={ytid}')
    os.system(f'yt-dlp -x --audio-format {format} --audio-quality {quality} --output "{os.path.join(root_path, str(audiocap_id), first_display_label, ytid)}_{start_seconds}-{end_seconds}.%(ext)s" --postprocessor-args "-ss {start_seconds} -to {end_seconds}" https://www.youtube.com/watch?v={ytid}')