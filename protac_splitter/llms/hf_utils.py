import huggingface_hub as hf

def create_hf_repository(**kwargs):
  """Creates a new Hugging Face repository."""
  api = hf.HfApi()
  return api.create_repo(**kwargs)


def delete_hf_repository(**kwargs):
  """Creates a new Hugging Face repository."""
  print(f'Deleting repository {kwargs["repo_id"]}.')
  api = hf.HfApi()
  return api.delete_repo(**kwargs)