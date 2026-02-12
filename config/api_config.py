"""
API配置文件 - 统一配置
所有路径相对于 BASE_DIR (data_prepare目录)
"""

import os

# ==================== 基础路径 ====================
# data_prepare 目录（full_process_m3/config/api_config.py → 上两级）
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# 外部模型根目录 - 修正路径指向正确的 models 目录
MODELS_DIR = os.environ.get('MODELS_DIR', '/root/siton-tmp/sx/xks/models')

# ==================== 路径配置 ====================
PATHS = {
    # --- data_prepare 内部路径 ---
    'osediff_model': os.path.join(BASE_DIR, 'OSEDiff', 'preset', 'models', 'osediff.pkl'),
    'ram_model': os.path.join(BASE_DIR, 'OSEDiff', 'preset', 'models', 'ram_swin_large_14m.pth'),
    'ram_ft_model': os.path.join(BASE_DIR, 'OSEDiff', 'preset', 'models', 'DAPE.pth'),
    'quality_prompt': os.path.join(BASE_DIR, 'qwen_filter', 'image_pair_quality_prompt.txt'),
    'precisecam': os.path.join(BASE_DIR, 'PreciseCam'),
    'osediff_repo': os.path.join(BASE_DIR, 'OSEDiff'),
    'qwen_filter': os.path.join(BASE_DIR, 'qwen_filter'),

    # --- 外部模型路径（通过 MODELS_DIR 环境变量控制）---
    'pretrained_model': os.path.join(MODELS_DIR, 'AI-ModelScope', 'stable-diffusion-2-1-base'),
    'qwen_model': os.path.join(MODELS_DIR, 'Qwen2.5-VL-7B-Instruct'),
}

def get_path(key):
    """获取指定的路径"""
    if key not in PATHS:
        raise ValueError(f"未知的路径键: {key}，可选: {list(PATHS.keys())}")
    return PATHS[key]

# ==================== API版本配置 ====================
API_VERSIONS = {
    'original': {
        'ports': {'preprocess': 5000, 'osediff': 5001, 'quality': 5002},
        'gpu': '0',
    },
    'shared_left': {
        'ports': {'preprocess': 5010, 'osediff': 5011, 'quality': 5012},
        'gpu': '1',
    }
}

API_BASE_URL = "http://localhost"

def get_api_ports(version='original'):
    """获取指定版本的API端口配置"""
    if version not in API_VERSIONS:
        raise ValueError(f"未知的API版本: {version}，可选: {list(API_VERSIONS.keys())}")
    return API_VERSIONS[version]['ports']

def get_gpu_id(version='original'):
    """获取指定版本的GPU ID"""
    if version not in API_VERSIONS:
        raise ValueError(f"未知的API版本: {version}，可选: {list(API_VERSIONS.keys())}")
    return API_VERSIONS[version]['gpu']