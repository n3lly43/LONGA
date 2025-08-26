import os
from huggingface_hub import HfApi

def push_to_hugging_face(
        folder_path:str,
        repo_id:str,
        repo_type:str
        ):
    api = HfApi(token=os.getenv("HF_TOKEN"))
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type=repo_type,
    )
