from huggingface_hub import snapshot_download
import os
def download_bert_base_chinese():
    local_dir=os.getcwd()+"/bert-base-chinese"
    snapshot_download(repo_id="bert-base-chinese",local_dir=local_dir,local_dir_use_symlinks=False,ignore_patterns=["*.msgpack","*.safetensors","*.h5"])