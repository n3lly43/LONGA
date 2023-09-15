import subprocess
import streamlit as st
from omegaconf import OmegaConf
# from scripts.speech_to_text_rnnt_bpe import main

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='The Machine Learning App',
    layout='wide')


#---------------------------------#
# Model training
def train_model(from_pretrained):
    if from_pretrained:
        subprocess.Popen([
        'python',
        './scripts/speech_to_text_rnnt_bpe.py',
        '--config-path','../configs/',
        '--config-name','conformer_transducer_bpe',
        'trainer.max_epochs',f'{max_epochs}',
        'exp_manager.name',f'{experiment_name}',
        'exp_manager.resume_if_exists',f'{resume_if_exists}',
        'exp_manager.resume_ignore_no_checkpoint',f'{resume_ignore_no_checkpoint}',
        'exp_manager.exp_dir','results/',
        'model.tokenizer.dir',f'{tokenizer_dir}',
        'model.train_ds.is_tarred',f'{is_tarred}',
        'model.train_ds.tarred_audio_filepaths',f'{train_filepaths}',
        'model.train_ds.manifest_filepath',f'{train_manifest}',
        'model.validation_ds.manifest_filepath',f'{val_manifest}',
        'model.test_ds.manifest_filepath',f'{test_manifest}',
        '+init_from_pretrained_model',f'{INIT_MODEL}'
        ])

    else:
        subprocess.Popen([
        'python',
        './scripts/speech_to_text_rnnt_bpe.py',
        '--config-path','../configs/',
        '--config-name','conformer_transducer_bpe',

        'exp_manager.name',f'{experiment_name}',
        'exp_manager.resume_if_exists',f'{resume_if_exists}',
        'exp_manager.resume_ignore_no_checkpoint',f'{resume_ignore_no_checkpoint}',
        'exp_manager.exp_dir','results/',

        'model.tokenizer.dir',f'{tokenizer_dir}',
        'model.train_ds.is_tarred',f'{is_tarred}',
        'model.train_ds.tarred_audio_filepaths',f'{train_filepaths}',
        'model.train_ds.manifest_filepath',f'{train_manifest}',
        
        'model.validation_ds.manifest_filepath',f'{val_manifest}',
        'model.test_ds.manifest_filepath',f'{test_manifest}',
        # '+init_from_pretrained_model',f'{INIT_MODEL}'
        ])
        
#---------------------------------#
st.write("""
# The Machine Learning App
""")

#---------------------------------#
# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Specify manifest files'):
    train_manifest = st.sidebar.file_uploader("Specify train manifest", type=["json"])
    val_manifest = st.sidebar.file_uploader("Specify validation manifest", type=["json"])
    test_manifest = st.sidebar.file_uploader("Specify test manifest", type=["json"])
    st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
""")

# Sidebar - Specify parameter settings
with st.sidebar.header('2. Set Parameters'):
    use_pretrained = st.sidebar.checkbox('Initialize model from pretrained weights')
    if use_pretrained:
        INIT_MODEL = st.sidebar.text_input('Please specify pretrained model', '')

with st.sidebar.subheader('2.1. Experiment Parameters'):
    experiment_name = st.sidebar.text_input('Please specify experiment name', '')
    max_epochs = st.sidebar.number_input('Please specify number of epochs')
    resume_if_exists = st.sidebar.checkbox('Resume training if checkpoint exists')
    resume_ignore_no_checkpoint = st.sidebar.checkbox('Start training with no existing checkpoints')

with st.sidebar.subheader('2.2. Data Parameters'):
    tokenizer_dir = st.sidebar.text_input('Specify path to tokenizer')
    is_tarred = st.sidebar.checkbox('Train set is tarred dataset')
    train_filepaths = st.sidebar.text_input('Specify path to train files')


#---------------------------------#
# Main panel

# Displays the dataset
st.subheader('1. Dataset')

st.markdown('**1.1. Manifest Files**')
st.write('Training set')
st.info(train_manifest)
st.write('Validation set')
st.info(val_manifest)
st.write('Test set')
st.info(test_manifest)

if tokenizer_dir is not None:
    st.markdown('**1.2. Tokenizer**')
    st.write('Tokenizer Path')
    st.info(tokenizer_dir)


st.subheader('2. Model Training')
if use_pretrained:
    st.write(f'Training from pretrained model {INIT_MODEL}')

if st.button('Train Model'):
    train_model(use_pretrained)