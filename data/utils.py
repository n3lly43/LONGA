import io
import os
import re
import wget
import json
import tarfile
import logging
import torch
import pandas as pd
from collections import defaultdict
from nemo.collections.asr.parts.utils.manifest_utils import write_manifest

from dataclasses import dataclass
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, 
                 features: List[Dict[str, Union[List[int], torch.Tensor]]]
                 ) -> Dict[str, torch.Tensor]:
        """
        split inputs and labels since they have to be of different 
        lengths and need different padding methods
        """
        
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:448]

        batch["labels"] = labels

        return batch

def get_file_progress_file_object_class(on_progress):
    class FileProgressFileObject(tarfile.ExFileObject):
        def read(self, size, *args):
            on_progress(self.name, self.position, self.size)
            return tarfile.ExFileObject.read(self, size, *args)
    return FileProgressFileObject

class TestFileProgressFileObject(tarfile.ExFileObject):
    def read(self, size, *args):
        on_progress(self.name, self.position, self.size)
        return tarfile.ExFileObject.read(self, size, *args)

class ProgressFileObject(io.FileIO):
    def __init__(self, path, *args, **kwargs):
        self._total_size = os.path.getsize(path)
        io.FileIO.__init__(self, path, *args, **kwargs)

    def read(self, size):
        logger.debug("Overall process: %d of %d (%d) percent" %(self.tell(), self._total_size, (self.tell()/self._total_size)*100))
        return io.FileIO.read(self, size)

def on_progress(filename, position, total_size):
    print("%s: %d of %s" %(filename, position, total_size))



def dataset_download(data_url: str, filename: str) -> None:    
    """
    Helper function to download files from link

    Args:
    data_url - link to files
    filename - save name for files downloaded from URL

    Returns:
    None
    """
    logger.debug("******")
    if not os.path.exists(f"{DATA_DIR}/{filename}"):
        data_path = wget.download(data_url, DATA_DIR)
        logger.info(f"Dataset downloaded at: {data_path}")
    else:
        logger.debug("Tarfile already exists.")
        data_path = f'{DATA_DIR}/{filename}'

def read_transcript(txt_filepath: str) -> str:
    """
    Helper function to read text from txt files

    Args:
    txt_filepath - path to txt file

    Return:
    String containg text from the txt file
    """
    path = f'{DATA_DIR}/{txt_filepath}'
    try:
        with open(path, 'rb') as f:
            text = f.read().decode(errors='replace')
            return text

    except FileNotFoundError:
        logger.debug(f'{txt_filepath} not found')
        return ''
    
def tsv_to_json(
    tsv_file:str,
    sampling_count:int,
    folder:str,
    audio_path_col:str = 'path',
    transcript_col:str = 'sentence') -> None:
    """
    Helper function to convert tsv files to json manifests for Nvidia NeMo

    Args:
    tsv_file        - Input TSV file
    sampling_count  - Number of examples, you want, use -1 for all examples
    folder          - Relative path to folder with audio files
    audio_path_col  - column name with audio file paths
    transcript_col  - transcripts column name 

    Return None
    """
    
    df = pd.read_csv(tsv_file, sep='\t')
    with open(tsv_file.replace('.tsv', '.json'), 'w') as fo:
        mod = 1
        if sampling_count > 0:
            mod = len(df) // sampling_count
        for idx in range(len(df)):
            if idx % mod != 0:
                continue
            item = {
                'audio_filepath': folder + "/" + df[audio_path_col][idx],
                'text': df[transcript_col][idx]
            }
            fo.write(json.dumps(item) + "\n")


def write_processed_manifest(data, original_path):
    original_manifest_name = os.path.basename(original_path)
    new_manifest_name = original_manifest_name.replace(".json", "_processed.json")

    manifest_dir = os.path.split(original_path)[0]
    filepath = os.path.join(manifest_dir, new_manifest_name)
    write_manifest(filepath, data)
    print(f"Finished writing manifest: {filepath}")
    return filepath


# calculate the character set
def get_charset(manifest_data):
    charset = defaultdict(int)
    logger.info("Computing character set")

    for row in manifest_data:
        text = row['text']
        for character in text:
            charset[character] += 1
    return charset

# Preprocessing steps
def remove_special_characters(data):
    chars_to_ignore_regex = "[\.\,\?\:\-!;()«»…\]\[/\*–‽+&_\\½√>€™$•¼}{~—=“\"”″‟„]"
    apostrophes_regex = "[’'‘`ʽ']"
    data["text"] = re.sub(chars_to_ignore_regex, " ", data["text"])  # replace punctuation by space
    data["text"] = re.sub(apostrophes_regex, "'", data["text"])  # replace different apostrophes by one
    data["text"] = re.sub(r"'+", "'", data["text"])  # merge multiple apostrophes

    # remove spaces where apostrophe marks a deleted vowel
    # this rule is taken from https://huggingface.co/lucio/wav2vec2-large-xlsr-kinyarwanda-apostrophied
    # data["text"] = re.sub(r"([b-df-hj-np-tv-z])' ([aeiou])", r"\1'\2", data["text"])

    # data["text"] = re.sub(r" '", " ", data["text"])  # delete apostrophes at the beginning of word
    # data["text"] = re.sub(r"' ", " ", data["text"])  # delete apostrophes at the end of word
    data["text"] = re.sub(r" +", " ", data["text"])  # merge multiple spaces
    return data


def replace_diacritics(data):
    data["text"] = re.sub(r"[éèëēê]", "e", data["text"])
    data["text"] = re.sub(r"[ãâāá]", "a", data["text"])
    data["text"] = re.sub(r"[úūü]", "u", data["text"])
    data["text"] = re.sub(r"[ôōó]", "o", data["text"])
    data["text"] = re.sub(r"[ćç]", "c", data["text"])
    data["text"] = re.sub(r"[ïī]", "i", data["text"])
    data["text"] = re.sub(r"[ñ]", "n", data["text"])
    return data


def remove_oov_characters(data):
    oov_regex = "[^ 'aiuenrbomkygwthszdcjfvplxq]"
    data["text"] = re.sub(oov_regex, "", data["text"])  # delete oov characters
    data["text"] = data["text"].strip()
    return data


# Processing pipeline
def apply_preprocessors(manifest, preprocessors):
    for processor in preprocessors:
        logger.debug(f"Applying {processor.__name__}")

        for idx in range(len(manifest)):
            manifest[idx] = processor(manifest[idx])

    logger.info("Finished processing manifest !")
    return manifest

