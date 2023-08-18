import os
import torch 

# Manifest filepaths
TOKENIZER_PATH = '/content/drive/MyDrive/CGIAR/Speech Recognition/Data/Raw/Luganda/makerere_radio_dataset/transcribed/tokenizer_bpe_maxlen_4'
TRAIN_MANIFEST = '/content/drive/MyDrive/CGIAR/Speech Recognition/Data/Raw/Luganda/makerere_radio_dataset/transcribed/train_tarred_1bk/tarred_audio_manifest.json'
TRAIN_FILEPATHS = '/content/drive/MyDrive/CGIAR/Speech Recognition/Data/Raw/Luganda/makerere_radio_dataset/transcribed/train_tarred_1bk/audio_0..1023.tar'
VAL_MANIFEST = '/content/drive/MyDrive/CGIAR/Speech Recognition/Data/Raw/Luganda/makerere_radio_dataset/transcribed/dev_decoded_processed.json'
TEST_MANIFEST = '/content/drive/MyDrive/CGIAR/Speech Recognition/Data/Raw/Luganda/makerere_radio_dataset/transcribed/test_decoded_processed.json'
TRANSDUCER_CONFIG_PATH = "/content/configs/conformer_transducer_bpe.yaml"

VOCAB_SIZE = 1024  # can be any value above 29
TOKENIZER_TYPE = "spe"  # can be wpe or spe
SPE_TYPE = "bpe"  # can be bpe or unigram
MAX_LEN = 4
TOKENIZER_TYPE == 'spe'

TOKENIZER = os.path.join(TOKENIZER_PATH, 
                         f"tokenizer_spe_{SPE_TYPE}_v{VOCAB_SIZE}_max_{MAX_LEN}")
TOKENIZER_TYPE_CFG = "bpe"


if torch.cuda.is_available():
  accelerator = 'gpu'
else:
  accelerator = 'gpu'

EPOCHS = 1000
