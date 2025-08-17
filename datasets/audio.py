import numpy as np
import pandas as pd

from glob import glob
from pydub import AudioSegment
from typing import List, Tuple, Dict

def insert_ids(data, name:str):
    df = data.copy()
    id_col = df['audio_name']

    ids = f'{name}_'+id_col+'_'+df.groupby(['audio_name'], as_index=False).cumcount().astype(str)
    df.insert(0, 'ID', ids)

    return df

def read_audio_file(audio_path):
    if '.wav' in audio_path:
        return AudioSegment.from_wav(audio_path)

    return AudioSegment.from_mp3(audio_path)

def get_audio_ids(df, audio_paths, format):
    """Filter out annotated audio IDS"""
    files = set((df['audio_name']+f'.{format}').unique())\
        .intersection(set([pth.split('/')[-1] 
                           for pth in glob(audio_paths, recursive=True)]))

    id_dict = {f:df[df['ID'].str.contains(f.strip(f'.{format}'))]["ID"].values.tolist() for f in files}
    ids = [[(id,pth) 
            for id in id_dict[pth.split('/')[-1]]] 
            for pth in glob(audio_paths, recursive=True) 
            if pth.split('/')[-1] in id_dict.keys() 
            if any(f in pth for f in files)]

    return [i for id in ids for i in id]

def split_audio(
    data:pd.DataFrame, 
    audio_ids:List[Tuple[str, str]], 
    audio_files:Dict[str, AudioSegment], 
    output_path:str
    ):
    """
    Split audio files into segments

    Args:
    data        - Transcriptions dataframe
    audio_ids   - audio IDs for annotated files
    audio_files - Dictionary with path to original files and corresponding AudioSegment
    output_path - path to store processed audio files
    """
    df = data.copy()

    for id,pth in audio_ids:
        t1 = df[df['ID']==id]['Begin Time - ss.msec'].astype(float).values.item() * 1000 #Works in milliseconds
        t2 = df[df['ID']==id]['End Time - ss.msec'].astype(float).values.item() * 1000

        newAudio = audio_files[pth][t1:t2]
        newAudio.export(f'{output_path}/clips/{id}.wav', format="wav")

def prepare_audio_files(
    dfs:List[pd.DataFrame], 
    paths:List[str], 
    output_path:str, 
    format:str
    ):
    """
    Convert audio files to .wav foramt and split into annotataed segments

    Args:
    dfs         - List of transcription dataframes
    paths       - Path to orginal audio files
    output_path - Path to store processed audio files
    format      - audio file format of inputs (.wav or .mp3)
    """
    for df,data_path in zip(dfs, paths):
        audio_ids = get_audio_ids(df, data_path, format)
        audio_files = {audio_path:read_audio_file(audio_path) 
                       for audio_path in set([ids[1] for ids in audio_ids])}

        print(len(audio_ids), df.shape)
        split_audio(df, audio_ids, audio_files, output_path)