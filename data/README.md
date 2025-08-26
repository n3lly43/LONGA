
## Data Cleaning
The *transcripts.py* script contains code used in processing transcriptions obtained from ELAN. The methods in the script can be used to:
1. Read and process csv files exported using ELAN
2. Summarize the audio and text information for the annotated data

The sample code block below can be used to process an exported csv file

```python
data_path = "path/to/transcriptions"
cols = ['Tier', 'Begin Time - hh:mm:ss.ms', 'Begin Time - ss.msec',
        'End Time - hh:mm:ss.ms', 'End Time - ss.msec', 'Duration - hh:mm:ss.ms',
        'Duration - ss.msec', 'text']

#optional
rename_cols = {"Temps de départ - hh:mm:ss.ms":'Begin Time - hh:mm:ss.ms',
        "Temps de départ - ss.msec":'Begin Time - ss.msec',
        "temps de fin - hh:mm:ss.ms":'End Time - hh:mm:ss.ms',
        "temps de fin - ss.msec":'End Time - ss.msec',
        "Durée - hh:mm:ss.ms":'Duration - hh:mm:ss.ms',
        "Durée - ss.msec":'Duration - ss.msec',
        "default":'text'}
drop_cols = ['Temps de départ - msec', 'temps de fin - msec', 'Durée - msec', 
"Temps de départ - PAL", "temps de fin - PAL", "Durée - PAL"]

annotator1 = read_transcriptions(
        data_path = data_path,
        cols = cols,     
        sample_col = 'Duration - ss.msec',
        replace_header = True, #must specify replace_cols
        rename_cols = rename_cols, #optional unless replace_header is True
        drop_cols = drop_cols #optional
        ):
```

The transcriptions output should be a dataframe with a structure similar to the one below

| Begin Time - hh:mm:ss.ms | Begin Time - ss.msec | ... | text | audio_name |
| -------------------------| -------------------- | --- | ---- | ---------- |
| 00:00:01.400 | 1.40 | ... | Ne tɔgɔ ye  | 60c8bd2911de30 |
| 00:06:04.080 | 364.080 | ... | kyenda mu musanvu akasirise satu ffe mwe mwe f... | R4OSP |

The transcription summary can be obtained using the following code block and should output the table below

```python
annotator1_summary = pd.DataFrame(
    build_summary_dict(
        data=annotator1,
        audio_data_path="path/to/audio/data",
        annotator="annotator's name or ID",
    ))
```

| Annotator | Total Transcribed Audio (hours) | ... | Total Transcription (CSV) Files Submitted |
| --------- | ------------------------------- | --- | ----------------------------------------- |
| Annotator1 | 5.76 | ... | 280 |
| Annotator2 | 7.41 | ... | 886 |

## Audio Data Preparation
Once the transcriptions obtained from ELAN are cleaned and in a uniform format, the merged datasets can then used to prepare audio files by splitting the recordings according to segments identified using “Begin Time - ss.msec” and “End Time - ss.msec” from the transcription data. These segments, as emphasized in the guidelines, ensure audio clips are on average no longer than 30 seconds. Following the split, the clips can then be resampled to 16kHz and converted into the .wav format as required by speech recognition models, particularly those used in this work.

Code in *audio.py* can be used to segment and process audio files as illustrated below

```python
#saves the segmented audio files to the output directory
prepare_audio_files(
    dfs=[annotator1, annotator2, annotator3],
    paths="path/to/audio/data",
    output_path="path/to/output/directory"
    format="wav" #or mp3, run twice changing the parameter if folder contains both file formats
)
```

## Train-Test Split
The code in the *preprocessing.py* script can be used to split the data into the train, validation, and test sets which are used in training and evaluating the ASR models. The code can also be used to further create manifest files for [Nvidia NeMo models](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/intro.html) as well as decode and resample the audio files.

```python
#read the saved transcription files
data_path = "path/to/saved/transcription/files"
transcriptions = pd.concat([pd.read_csv(pth) for pth in glob(data_path)])

#saves the split transcriptions to folder
train, val, test = dataset_split(
        data=transcriptions,
        transcript_col="sentence",
        audio_files_dir="path/to/audio/files",
        save_path="path/to/save/transcription?splits",
        return_splits=True
    )

#create manifest
tsv_to_json(
    tsv_file="path/to/saved/transcription/split,
    sampling_count=-1,
    destination_folder="path/to/save/manifest
)

#decode audio files not in .wav format and resample to 16kHz
decode_resample(
    manifest="path/to/json/file", #usually tsv_file.replace('.tsv', '.json'), 
    destination_folder="path/to/save/decoded/audio/files")
```
