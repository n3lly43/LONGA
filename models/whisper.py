import evaluate
import pandas as pd
from typing import Optional

from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import WhisperFeatureExtractor
from transformers import WhisperForConditionalGeneration
from datasets import Audio, Dataset, DatasetDict, load_from_disk, load_dataset

from data.utils import DataCollatorSpeechSeq2SeqWithPadding

def get_dataset_from_manifest(audio_data:str):
    df = pd.read_json(audio_data.replace('\ ', ' '), lines=True)
    df = df[df['text'].str.len()>0]

    data = {
        "audio": df['audio_filepath'].values.tolist(),
        'sentence': df['text'].values.tolist()
        }

    return Dataset.from_dict(data).cast_column("audio", Audio())

def encode_dataset(batch, feature_extractor, tokenizer):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

def prepare_dataset_from_manifest(
        train_manifest:str,
        val_manifest:Optional[str]=None,
        test_manifest:Optional[str]=None,
        ) -> Dataset|DatasetDict:
    """
    Prepare dataset from json manifests
    """
    dataset = DatasetDict({
        'train':get_dataset_from_manifest(train_manifest)}
        )
    if val_manifest is not None:
        dataset = DatasetDict({
            'train':get_dataset_from_manifest(train_manifest),
            'val':get_dataset_from_manifest(val_manifest)}
            )
    if test_manifest is not None:
        dataset = DatasetDict({
            'train':get_dataset_from_manifest(train_manifest),
            'test':get_dataset_from_manifest(test_manifest)}
            )
    if val_manifest is not None and test_manifest is not None:
        dataset = DatasetDict({
            'train':get_dataset_from_manifest(train_manifest),
            'val':get_dataset_from_manifest(val_manifest),
            'test':get_dataset_from_manifest(test_manifest)}
            )
        
    return dataset            

def prepare_dataset(cfg, feature_extractor, tokenizer) -> DatasetDict|Dataset:
    """
    Prepare train, validation, and test datasets
    """
    if cfg.load_from_hub:
        assert cfg.repo_id is not None, "Specify Hugging Face repo with dataset"
        return  load_dataset(cfg.repo_id)
    
    if cfg.load_path is not None:
        dataset = load_from_disk(cfg.load_path)

    dataset = prepare_dataset_from_manifest(cfg.train_manifest, 
                                            cfg.val_manifest, 
                                            cfg.test_manifest)

    cols = dataset.column_names
    dataset = dataset.map(lambda batch: encode_dataset(batch, 
                                                       feature_extractor, 
                                                       tokenizer), 
                          remove_columns=cols["train"])

    return dataset

def compute_metrics(pred, tokenizer):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    metric = evaluate.load("wer")
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

def load_pretrained_model(
        pretrained_model:str,
        language:str
    ):

    model = WhisperForConditionalGeneration.from_pretrained(pretrained_model)
    model.generation_config.language = language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    return model

def prepare_model(cfg):
    model = load_pretrained_model(cfg.pretrained_model, cfg.language)

    feature_extractor = WhisperFeatureExtractor.from_pretrained(cfg.pretrained_model)
    tokenizer = WhisperTokenizer.from_pretrained(cfg.pretrained_model)

    processor = WhisperProcessor.from_pretrained(cfg.pretrained_model)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    
    return model, data_collator, processor, feature_extractor, tokenizer