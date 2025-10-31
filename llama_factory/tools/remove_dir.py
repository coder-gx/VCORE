import os
import shutil

def clean_subdirs(root_dir, keep_names):
    """
    删除 root_dir 下的子文件夹，保留 keep_names 中的那些。
    
    :param root_dir: 目标文件夹路径
    :param keep_names: 需要保留的子文件夹名（list[str]）
    """
    for name in os.listdir(root_dir):
        full_path = os.path.join(root_dir, name)
        if os.path.isdir(full_path) and name not in keep_names:
            print(f"Deleting: {full_path}")
            shutil.rmtree(full_path)

# 使用示例
root = "/gemini/data-1/margin_sft/llama_factory/saves_final/chemistry/qwen3_32b_random/lora/sft"
keep = [f""]
clean_subdirs(root, keep)
