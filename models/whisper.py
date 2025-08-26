import evaluate
import pandas as pd
from typing import Optional, List, Tuple

from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import WhisperFeatureExtractor
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Audio, Dataset, DatasetDict, load_from_disk

from utils import push_to_hugging_face
from data.utils import DataCollatorSpeechSeq2SeqWithPadding
from models import training_args

def get_dataset_from_manifest(audio_data:str):
    df = pd.read_json(audio_data.replace('\ ', ' '), lines=True)
    df = df[df['text'].str.len()>0]

    data = {
        "audio": df['audio_filepath'].values.tolist(),
        'sentence': df['text'].values.tolist()
        }

    return Dataset.from_dict(data).cast_column("audio", Audio())

def encode_dataset(batch):
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

def prepare_dataset(
        train_manifest:Optional[str]=None,
        val_manifest:Optional[str]=None,
        test_manifest:Optional[str]=None,
        save_path:Optional[str]=None,
        load_path:Optional[str]=None,
        dataset:Optional[DatasetDict|Dataset]=None,
        repo_id:Optional[str]=None,
        push_to_hf:bool=False,
        ) -> DatasetDict|Dataset:
    """
    Prepare train, validation, and test datasets

    Args:    
    train_manifest  - Path to training data manifest
    val_manifest    - Path to validation data manifest
    test_manifest   - Path to test data manifest
    save_path       - Path to save Dataset object
    load_path       - Path to dataset
    dataset         - Dataset object
    repo_id         - Hugging Face repository id
    push_to_hf      - Push dataset to Hugging Face

    """
    if load_path is not None:
        dataset = load_from_disk(load_path)

    if dataset is None:
        assert train_manifest is not None, "Specify dataset"
        dataset = prepare_dataset_from_manifest(train_manifest, 
                                                val_manifest, 
                                                test_manifest)

    cols = dataset.column_names
    fri_dataset = dataset.map(encode_dataset, 
                              remove_columns=cols["train"])

    if save_path is not None:
        fri_dataset.save_to_disk(save_path)

    if push_to_hf:
        assert repo_id is not None, "Specify HF Repo"
        fri_dataset.push_to_hub(repo_id=repo_id)
    
    return dataset

def compute_metrics(pred, pretrained_model):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    tokenizer = WhisperTokenizer.from_pretrained(pretrained_model)
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

def prepare_trainer(
        pretrained_model:str,
        language:str,
        training_args:Seq2SeqTrainingArguments,
        pretrained:bool=True,
        model:Optional[WhisperForConditionalGeneration]=None,
        train_manifest:Optional[str]=None,
        val_manifest:Optional[str]=None,
        test_manifest:Optional[str]=None,
        save_path:Optional[str]=None,
        load_path:Optional[str]=None,
        dataset:Optional[DatasetDict|Dataset]=None,
        repo_id:Optional[str]=None,
        push_to_hf:bool=False,
        eval_set:str='test'
        ):
    if pretrained:
        model = load_pretrained_model(pretrained_model, language)

    assert model is not None, "Specify model to use" 

    if dataset is None:
        assert not any(m is None for m in [train_manifest, val_manifest, test_manifest]), \
        "Specify manifest files to build dataset"
        dataset = prepare_dataset(train_manifest, 
                                    val_manifest, 
                                    test_manifest, 
                                    save_path, 
                                    load_path, 
                                    dataset, 
                                    repo_id, 
                                    push_to_hf
                                    )
    # feature_extractor = WhisperFeatureExtractor.from_pretrained(pretrained_model)
    processor = WhisperProcessor.from_pretrained(pretrained_model)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    
    return dataset, Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset[eval_set],
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, pretrained_model),
        processing_class=processor.feature_extractor,
    )
