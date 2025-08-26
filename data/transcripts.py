import pandas as pd
from glob import glob
from typing import List, Optional, Dict

def build_summary_dict(
        data:pd.DataFrame, 
        audio_data_path:str,
        annotator:str,
        duration_col:str = 'Duration - ss.msec',
        ):
    """
    Get dictionary with summary of transcriptions 

    Args:
    data            - Transcriptions dataframe from ELAN
    audio_data_path - path to audio files
    annotator       - Annotator name
    duration_col    - Column indicating duration of segemnts
    """
    df = data.copy()

    return {
    'Annotator': [annotator],
    'Total Transcribed Audio (hours)':[ round(df[duration_col].sum()/3600, 2)],
    'Longest Clip (secs)':[df[duration_col].max()],
    'Shortest Clip (secs)':[df[duration_col].min()],
    'Average Clip Length (secs)':[ round(df[duration_col].mean(), 2)],
    'Total Audio Files Assigned':[len(glob(f'{audio_data_path}/*/*[0-9].wav'))],
    'Total Annotation (EAF) Files Submitted':[len(glob(f'{audio_data_path}/*/*[0-9].eaf'))],
    'Total Transcription (CSV) Files Submitted':[len(glob(f'{audio_data_path}/*/*[0-9].csv'))],
    }

def read_transcriptions(
        data_path:str,
        cols:List[str],      
        rename_cols:Optional[Dict],
        replace_header:bool = False,
        sample_col:str = 'Duration - ss.msec',
        drop_cols:Optional[List[str]] = None
        ):
    """
    Read transcriptions from ELAN

    Args:
    data_path - path to directory with transcription data
    """
    # cols = ['Tier', 'Begin Time - hh:mm:ss.ms', 'Begin Time - ss.msec',
    #         'End Time - hh:mm:ss.ms', 'End Time - ss.msec', 'Duration - hh:mm:ss.ms',
    #         'Duration - ss.msec', 'text']
    dfs = check_tiers(data_path, cols, sample_col, 
                      rename_cols, replace_header, drop_cols)
    return pd.concat(dfs).reset_index(drop=True)

def check_tiers(
        data_path:str, 
        cols:List[str],
        sample_col:str,        
        rename_cols:Optional[Dict],
        replace_header:bool,
        drop_cols:Optional[List[str]]
        ):
    """Check if data was annotated with tiers and read accordingly"""
    
    use_cols = list(range(len(cols)+1))
    file_list = glob(f'{data_path}/**/*[0-9].csv')
    dfs = [pd.read_csv(pth) for pth in file_list]

    if any(sample_col in df.columns for df in dfs):
        if any(c in df.columns for df in dfs for c in ['Unnamed','examiner']):        
            return [read_dfs(pth, rename_cols) for pth in file_list]
          
        dfs = [pd.read_csv(pth, on_bad_lines='skip', header=1)\
            .dropna(axis=1, how='all')\
                .assign(audio_name=pth\
                        .split('/')[-1]\
                            .replace('.csv', ''))
                if pd.read_csv(pth, on_bad_lines='skip')\
                    .dropna(axis=1, how='all').shape[1]<3
                    else pd.read_csv(pth, on_bad_lines='skip')\
                        .dropna(axis=1, how='all')\
                            .assign(audio_name=pth.split('/')[-1]\
                                    .replace('.csv', '')) for pth in file_list]
          
        dfs = [df for df in dfs if df.shape[1]>2]

        return [df.dropna(
            subset=df.columns[list(df.columns).index('Duration - ss.msec')+1:], 
                how='all') for df in dfs]
    
    if replace_header:
        return [read_dfs(pth, rename_cols, replace_header, drop_cols) 
                for pth in file_list]
    
    if use_cols==9:
        use_cols.remove(1)

    #read transcription data and add audio file names
    return [pd.read_csv(pth, header=None, usecols=use_cols, names=cols)\
                    .dropna(subset=['text'])\
                        .assign(audio_name=pth.split('/')[-1]\
                                .replace('csv', ''))
                    for pth in file_list]

def read_dfs(
        pth:str, 
        rename_cols:Optional[Dict],
        replace_header:bool = False,
        drop_cols:Optional[List[str]] = None):
    """
    Helper function to process transcriptions with tiers

    Args:
    pth             - path to transcription dataframe
    replace_header  - flag to translate non-English (French) column names
    drop_cols       - redundant columns to drop from dataset

    Returns dataframe with saved audio file names
    """
    #non-English headers
    if replace_header:
        assert rename_cols is not None, "Specify columns to replace"
        df = pd.read_csv(pth, header=1) 

        if len(df.columns)<len(rename_cols):
            return pd.read_csv(pth, 
                            header=None, 
                            usecols=list(range(2, len(rename_cols)+2)), 
                            names=list(rename_cols.values()))
        
        if drop_cols is not None:
            return df.rename(columns=rename_cols).drop(columns=drop_cols)
        
        return df.rename(columns=rename_cols)
    

    df = pd.read_csv(pth) 
    #exclude empty columns
    cols = [c for c in df.columns 
            if all(d not in c for d in ['Unnamed','examiner'])]
    df = df[cols]

    try:
        return df.dropna(subset=['participant'])\
            .assign(audio_name=pth.split('/')[-1]\
                    .replace('csv', 'wav'))
    
    except KeyError:
        #rename participant column and return the data
        return df.rename(columns=rename_cols)\
            .dropna(subset=['participant'])\
                .assign(audio_name=pth.split('/')[-1]\
                        .replace('csv', 'wav'))