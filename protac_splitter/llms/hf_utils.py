from typing import Optional

import huggingface_hub as hf
from huggingface_hub import get_hf_file_metadata, hf_hub_url, repo_info
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError


def repo_exists(repo_id: str, token: Optional[str] = None) -> bool:
    try:
        print(repo_info(repo_id, token=token))
        return True
    except RepositoryNotFoundError:
        return False

def create_hf_repository(**kwargs):
    """Creates a new Hugging Face repository."""
    api = hf.HfApi()
    return api.create_repo(**kwargs)


def delete_hf_repository(**kwargs):
    """Creates a new Hugging Face repository."""
    print(f'Deleting repository {kwargs["repo_id"]}.')
    api = hf.HfApi()
    return api.delete_repo(**kwargs)