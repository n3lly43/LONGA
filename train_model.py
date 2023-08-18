import os
import gc
import torch
import argparse
import nemo.collections.asr as nemo_asr

from omegaconf import OmegaConf, open_dict
from config import accelerator, EPOCHS, TOKENIZER, TOKENIZER_TYPE_CFG
from pytorch_lightning import Trainer
from nemo.utils import exp_manager

parser = argparse.ArgumentParser()
parser.add_argument(
    '--config_path',
     required=True,
      type=str,
       help='config file path')
parser.add_argument(
    '--pretrained_model',
     required=True,
      type=bool,
       help='pretrained model from huggingface')
parser.add_argument(
    '--weights',
     default='pretrained',
       required=True,
       type=str,
       help='''
       pretrained model weights to load: 
       pretrained=full pretrained model, 
       encoder=encoder weights
       checkpoint=model checkpoint to resume training''')
parser.add_argument(
    "--experiment_folder",
     required=True,
      type=str,
       help="Destination folder where checkpoints and experiment logs will be saved")
parser.add_argument(
    "--experiment_name",
     required=True,
      type=str,
       help="Current experiment name")
# parser.add_argument(
#     "--train_model",
#     default=True,
#      required=False,
#       type=str,
#        help="train or test the model on the data")
args = parser.parse_args()

config = OmegaConf.load(args.config_path)

# Initialize a Trainer for the Transducer model
trainer = Trainer(devices=1,
                accelerator=accelerator,
                max_epochs=EPOCHS,
                enable_checkpointing=False,
                logger=False,
                log_every_n_steps=5,
                check_val_every_n_epoch=2)

model = nemo_asr.models.EncDecRNNTBPEModel(cfg=config.model, trainer=trainer)
pretrained_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
   args.pretrained_model, trainer=trainer)

if args.weights=='encoder':
    model.encoder.load_state_dict(pretrained_model.encoder.state_dict(), strict=True)

elif args.weights=='pretrained':
    model = pretrained_model
    model.cfg.train_ds.sample_rate = 16000
    model.cfg.validation_ds.sample_rate = 16000
    model.cfg.test_ds.sample_rate = 16000

    model.set_trainer(trainer)
    model.setup_training_data(config.model.train_ds)
    model.setup_validation_data(config.model.validation_ds)
    model.setup_test_data(config.model.test_ds)

    model.cfg.tokenizer.dir = TOKENIZER
    model.cfg.tokenizer.type = TOKENIZER_TYPE_CFG

elif args.weighs=='checkpoint':
    model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(
      restore_path=args.pretrained_model, 
      trainer=trainer)
    
    config.model.train_ds.sample_rate = 16000
    config.model.validation_ds.sample_rate = 16000
    config.model.test_ds.sample_rate = 16000

    model.set_trainer(trainer)
    model.setup_training_data(config.model.train_ds)
    model.setup_validation_data(config.model.validation_ds)
    model.setup_test_data(config.model.test_ds)

model.summarize()
# Prepare NeMo's Experiment manager to handle checkpoint saving and logging for us

# Environment variable generally used for multi-node multi-gpu training.
# In notebook environments, this flag is unnecessary and can cause logs of multiple training runs to overwrite each other.
os.environ.pop('NEMO_EXPM_VERSION', None)

exp_config = exp_manager.ExpManagerConfig(
    exp_dir=args.experiment_folder,
    name=args.experiment_name,
    resume_if_exists=True,
    resume_ignore_no_checkpoint=True,
    checkpoint_callback_params=exp_manager.CallbackParams(
        monitor="val_wer",
        mode="min",
        always_save_nemo=True,
        save_best_model=True,
    ),
)

exp_config = OmegaConf.structured(exp_config)

logdir = exp_manager.exp_manager(trainer, exp_config)

# try:
#   from google import colab
#   COLAB_ENV = True
# except (ImportError, ModuleNotFoundError):
#   COLAB_ENV = False

# # Load the TensorBoard notebook extension
# if COLAB_ENV:
#     %reload_ext tensorboard
#     %tensorboard --logdir /content/drive/MyDrive/CGIAR/Speech\ Recognition/experiments/LG-Conformer-Char-Model/
# else:
#     print("To use TensorBoard, please use this notebook in a Google Colab environment.")

# Release resources prior to training
gc.collect()

if accelerator == 'gpu':
  torch.cuda.empty_cache()

trainer.fit(model)
# if args.train_model:
# else:
#     trainer.test(model)
