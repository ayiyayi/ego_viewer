import os
from huggingface_hub import snapshot_download
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# snapshot_download(
#     repo_id="IDEA-Research/grounding-dino-tiny",
#     local_dir="/mnt/data/liuyu/project/HumanEgo/models/grounding-dino-tiny",
#     repo_type="model"
# )


snapshot_download(
    repo_id="facebook/sam2-hiera-tiny",
    repo_type="model",
    local_dir="/mnt/data/liuyu/project/HumanEgo/models/sam2-hiera-tiny"
)