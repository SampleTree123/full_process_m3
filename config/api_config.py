"""
API配置文件 - 统一配置，支持两个版本
"""

# API版本配置
API_VERSIONS = {
    'original': {
        'ports': {
            'preprocess': 5000,
            'osediff': 5001,
            'quality': 5002
        },
        'gpu': '0',
        'description': '原版API (GPU 0)'
    },
    'shared_left': {
        'ports': {
            'preprocess': 5010,
            'osediff': 5011,
            'quality': 5012
        },
        'gpu': '1',
        'description': '共享左图版本 (GPU 1)'
    }
}

# API服务基础URL
API_BASE_URL = "http://localhost"

# 获取指定版本的端口配置
def get_api_ports(version='original'):
    """获取指定版本的API端口配置
    
    Args:
        version: API版本 ('original' 或 'shared_left')
    
    Returns:
        端口配置字典
    """
    if version not in API_VERSIONS:
        raise ValueError(f"未知的API版本: {version}，可选: {list(API_VERSIONS.keys())}")
    return API_VERSIONS[version]['ports']

# 获取指定版本的GPU配置
def get_gpu_id(version='original'):
    """获取指定版本的GPU ID
    
    Args:
        version: API版本 ('original' 或 'shared_left')
    
    Returns:
        GPU ID字符串
    """
    if version not in API_VERSIONS:
        raise ValueError(f"未知的API版本: {version}，可选: {list(API_VERSIONS.keys())}")
    return API_VERSIONS[version]['gpu']

# 向后兼容：默认使用原版配置
API_PORTS = API_VERSIONS['original']['ports'] 