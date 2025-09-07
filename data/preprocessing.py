import os
import sox
import json
import pandas as pd
from sox import Transformer
from typing import Tuple, Optional
from glob import glob

from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from .utils import (tsv_to_json, remove_oov_characters, apply_preprocessors, 
                   remove_special_characters, replace_diacritics, write_processed_manifest)
SEED = 935

def load_transcript_data(path):
    dfs = {pth:pd.read_csv(pth) 
    for pth in glob(f"{path}/*csv", recursive=True)}

    return list(dfs.values())

def prepare_transcript_data(transcript_path, drop_cols):
    dfs = load_transcript_data(transcript_path)
    transcript = pd.concat(dfs).reset_index(drop=True)

    transcript.rename(columns={'text':'sentence'}, inplace=True)
    transcript['path'] = transcript['ID']+'.wav'

    transcript = transcript.drop(columns=[c for c in drop_cols if c in transcript.columns])

    #drop samples with overlapping speakers
    transcript = transcript[~transcript['sentence'].str.contains('--')].reset_index(drop=True)
    transcript['sentence'] = transcript['sentence'].str.replace('\[a\d+\]', '', regex=True)
    
    return transcript

def process(x, destination_folder):
    if not isinstance(x['text'], str):
        x['text'] = ''
    else:
        x['text'] = x['text'].lower().strip()
    _, file_with_ext = os.path.split(x['audio_filepath'])
    name, ext = os.path.splitext(file_with_ext)
    output_wav_path = destination_folder +'/'+ name + '.wav'
    if not os.path.exists(output_wav_path):
        tfm = Transformer()
        tfm.rate(samplerate=16000)
        tfm.channels(n_channels=1)
        tfm.build(input_filepath=x['audio_filepath'],
                output_filepath=output_wav_path)
    x['duration'] = sox.file_info.duration(output_wav_path)
    x['audio_filepath'] = output_wav_path
    return x


def load_data(manifest):
    data = []
    with open(manifest, 'r') as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    return data


def decode_resample(
    manifest:str, 
    destination_folder:str
    ) -> None:
    """
    Convert .mp3 files to .wav with sample rate of 16000

    Args: 
    manifest            - path to the original manifest
    num_workers         - Workers to process dataset
    destination_folder  - Destination folder where audio files will be stored

    Returns None
    """
    data = load_data(manifest)

    data_new = [process(x, destination_folder) for x in tqdm(data)]

    with open(manifest.replace('.json', '_decoded.json'), 'w') as f:
        for item in data_new:
            f.write(json.dumps(item) + '\n')


def dataset_split(
    data:pd.DataFrame, 
    transcript_col:str, 
    audio_files_dir:str,
    save_path:str,
    return_splits:bool=False
    ) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:

    #split data into features and target
    df = data.copy()
    features = df.drop(columns=[transcript_col])
    target = df[transcript_col].to_frame()

    X_train, X_test, y_train, y_test = train_test_split(features, target,
                                                        # stratify=target,
                                                        test_size=0.2,
                                                        random_state=SEED)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                        # stratify=target,
                                                        test_size=0.25,
                                                        random_state=SEED)

    #merge feature and target dataframes
    train = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
    test = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)
    val = pd.concat([X_val, y_val], axis=1).reset_index(drop=True)

    fls = [pth.split(f'{audio_files_dir}/')[1] for pth in glob(f'{audio_files_dir}/*.wav', recursive=True)]

    #save datasets to TSV files
    train[train['path'].isin(fls)].to_csv(f'{save_path}/train.tsv', sep="\t", index=False)
    test[test['path'].isin(fls)].to_csv(f'{save_path}/test.tsv', sep="\t", index=False)
    val[val['path'].isin(fls)].to_csv(f'{save_path}/dev.tsv', sep="\t", index=False)

    if return_splits:
        return train[train['path'].isin(fls)], val[val['path'].isin(fls)], test[test['path'].isin(fls)]

def process_data(
    tsv_file:str, 
    origin_folder:str,
    destination_folder:str, 
    sampling_count:int = -1, 
    return_manifest:bool = False
    ):

    """
    Prepare manifests and process data for Nvidia NeMo

    Args:
    tsv_file            - Input TSV file
    sampling_count      - Number of examples, you want, use -1 for all examples
    origin_folder     ` - Folder where original audio files will be stored
    destination_folder  - Destination folder where audio files will be stored
    return_manifest     - Return processed manifest

    Returns NeMo Manifest if return_manifest is True, otherwise None
    """

    #prepare manifest and convert audio files
    tsv_to_json(tsv_file, sampling_count, origin_folder)
    decode_resample(tsv_file.replace('.tsv', '.json'), destination_folder)
    
    # List of pre-processing functions
    PREPROCESSORS = [
        remove_special_characters,
        replace_diacritics,
        remove_oov_characters,
    ]

    manifest = tsv_file.replace('.tsv', '_decoded.json')
    data_processed = write_processed_manifest(
        apply_preprocessors(
            read_manifest(manifest), 
            PREPROCESSORS), 
            manifest
        ) 


    if return_manifest:
        return data_processed