from huggingface_hub import snapshot_download

# 数据集名称
# dataset_name = "Deep1994/ReNeLLM-Jailbreak"
dataset_name = "avery-ma/ManyHarm"

# 指定保存路径
local_dir = "/root/Safety-Refine/data"

# 下载 snapshot（保留原始结构）
snapshot_download(
    repo_id=dataset_name,
    repo_type="dataset",  # 注意这里要指定是数据集
    local_dir=local_dir,
    local_dir_use_symlinks=False  # 直接复制文件而不是软链接
)

print(f"数据集已下载到: {local_dir}")
