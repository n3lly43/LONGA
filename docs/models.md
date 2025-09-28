This work fine-tuned a Whisper model using the transformers library along with a transducer model from the Nvidia NeMo toolkit. To ease the training and finetuning of the models, we make use of configuration (.yaml) files created using the [OmegaConf](https://omegaconf.readthedocs.io/) library. Configurations for the NeMo models are downloaded from the [GitHub repo](https://github.com/NVIDIA-NeMo/NeMo/tree/main/examples/asr/conf) using the `wget` package

```Bash
wget -P configs/ https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/asr/conf/conformer/conformer_transducer_bpe.yaml
```
The Whisper config can be created using the code below

```Python
from omegaconf import OmegaConf
from omegaconf import MISSING

dataset_config = OmegaConf.create({
    'train_manifest':MISSING,
    'val_manifest'  :MISSING,
    'test_manifest' :MISSING,
    'eval_set'      :'val',
    'load_path'     :None,
    'load_from_hub' :False,
    'repo_id'       :None
})

pretrained_model_config = OmegaConf.create({
    'pretrained_model':MISSING,
    'language':MISSING,
})

# from https://huggingface.co/blog/fine-tune-whisper
train_args_config = OmegaConf.create({
    'output_dir':MISSING,
    'per_device_train_batch_size':16,
    'gradient_accumulation_steps':1,  # increase by 2x for every 2x decrease in batch size
    'learning_rate':1e-5,
    'warmup_steps':500,
    'max_steps':500000,
    'gradient_checkpointing':True,
    'fp16':True,
    'eval_strategy':"steps",
    'per_device_eval_batch_size':8,
    'predict_with_generate':True,
    'generation_max_length':225,
    'save_steps':1000,
    'eval_steps':1000,
    'logging_steps':25,
    'report_to':"tensorboard",
    'load_best_model_at_end':True,
    'metric_for_best_model':"wer",
    'greater_is_better':False,
    'push_to_hub':False,
    'hub_model_id':None
})

# final model config
model_config = OmegaConf.create({
    'name':'Whisper Longa OpenAI',
    'dataset':dataset_config,
    'model':pretrained_model_config,
    'training_args':train_args_config
})

# save model config
OmegaConf.save(config=model_config, f="configs/luganda_whisper.yaml")
```

The models can then be fine-tuned using the configs as illustrated below

```Bash
# Whisper model fine-tuning
python train.py \
    --config-path=configs \
    --config-name=luganda_whisper \
    model.pretrained_model="Sunbird/asr-whisper-large-v3-salt" \
    model.language="swahili" \
    dataset.train_manifest=<path to train manifest> \
    dataset.val_manifest=<path to validation manifest> \
    dataset.test_manifest=<path to test manifest> \
    training_args.output_dir=<path to experiments directory>
```

```Bash
# NeMo Conformer model fine-tuning
python train.py \
    --config-path=configs/ \
    --config-name=conformer_transducer_bpe \
    trainer.max_epochs=100 \
    trainer.check_val_every_n_epoch=5 \
    exp_manager.name="longa-multilingual-luganda" \
    exp_manager.resume_if_exists=true \
    exp_manager.resume_ignore_no_checkpoint=true \
    exp_manager.exp_dir=<path to experiments folder> \
    model.tokenizer.dir=<path to tokenizer> \
    model.train_ds.is_tarred=false \
    model.train_ds.manifest_filepath=<path to training dataset> \
    model.validation_ds.manifest_filepath=<path to validation dataset> \
    model.test_ds.manifest_filepath=<path to test dataset> \
    +model.freeze_updates.enabled=true \
    +model.freeze_updates.modules.encoder=50
```