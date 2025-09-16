import os
from huggingface_hub import HfApi

def push_to_hugging_face(
        token:str,
        folder_path:str,
        repo_id:str,
        repo_type:str
        ):
    api = HfApi(token=token)
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type=repo_type,
    )
