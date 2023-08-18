import argparse
from omegaconf import OmegaConf, open_dict

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_path", 
    required=True, 
    type=str, 
    help="yaml file with model config")

parser.add_argument(
    "--train_manifest", 
    required=True, 
    type=str, 
    help="Train set manifest")
parser.add_argument(
    "--tarred_audio_filepaths", 
    required=True, 
    type=str, 
    help="Tarred audio filepaths")

parser.add_argument(
    "--valid_manifest", 
    required=True, 
    type=str, 
    help="Validation set manifest")
parser.add_argument(
    "--test_manifest", 
    required=True, 
    type=str, 
    help="Test set manifest")

parser.add_argument(
    "--tokenizer", 
    required=True, 
    type=str, 
    help="Tokenizer path")
parser.add_argument(
    "--tokenizer_type", 
    required=True, 
    type=str, 
    help="tokenizer type: spe, bpe")

parser.add_argument(
    "--fuse_loss_wer", 
    required=False, 
    type=bool, 
    help="WER loss for fused training")
parser.add_argument(
    "--fused_batch_size", 
    required=False, 
    type=int, 
    help="batch size for fused training")
args = parser.parse_args()

config = OmegaConf.load(args.config_path)

# config.model.train_ds.is_tarred = True
config.model.train_ds.manifest_filepath = args.train_manifest
config.model.train_ds.tarred_audio_filepaths = args.tarred_audio_filepaths
# config.model.train_ds.batch_size = 8

config.model.validation_ds.manifest_filepath = args.valid_manifest
config.model.test_ds.manifest_filepath = args.test_manifest

config.model.tokenizer.dir = args.tokenizer
config.model.tokenizer.type = args.tokenizer_type

# Two lines to enable the fused batch step for transducer model
config.model.joint.fuse_loss_wer = args.fuse_loss_wer
config.model.joint.fused_batch_size = 4  # this can be any value (preferably less than model.*_ds.batch_size)

OmegaConf.save(args.config_path)