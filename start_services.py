"""
启动所有API服务的脚本（统一版本）
支持两个API版本：original (GPU 0) 和 shared_left (GPU 1)
"""

import os
import sys
import subprocess
import time
import logging
import argparse
import requests
from typing import Dict, List

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_services_config(api_version='original'):
    """根据API版本获取服务配置
    
    Args:
        api_version: 'original' (端口5000-5002, GPU 0) 或 'shared_left' (端口5010-5012, GPU 1)
    
    Returns:
        服务配置字典
    """
    # 基础端口和GPU配置
    if api_version == 'shared_left':
        base_port = 5010
        gpu_id = '1'
        log_suffix = '_shared_left'
    else:  # original
        base_port = 5000
        gpu_id = '0'
        log_suffix = ''
    
    return {
        'preprocess': {
            'script': 'api_services/preprocess.py',
            'port': base_port,
            'env': 'xks_precisecam',
            'args': ['--port', str(base_port)],
            'log_suffix': log_suffix
        },
        'osediff': {
            'script': 'api_services/osediff_api.py',
            'port': base_port + 1,
            'env': 'xks_OSEDiff',
            'env_vars': {'CUDA_VISIBLE_DEVICES': gpu_id, 'HF_ENDPOINT': 'https://hf-mirror.com'},
            'args': [
                '--port', str(base_port + 1),
                '--osediff_path', '/root/siton-tmp/sx/xks/51_code/data_prepare/OSEDiff/preset/models/osediff.pkl',
                '--pretrained_model_name_or_path', '/root/siton-tmp/sx/xks/models/AI-ModelScope/stable-diffusion-2-1-base',
                '--ram_path', '/root/siton-tmp/sx/xks/51_code/data_prepare/OSEDiff/preset/models/ram_swin_large_14m.pth',
                '--ram_ft_path', '/root/siton-tmp/sx/xks/51_code/data_prepare/OSEDiff/preset/models/DAPE.pth',
                '--device', 'cuda',
                '--upscale', '2',
                '--process_size', '512',
                '--mixed_precision', 'fp16'
            ],
            'log_suffix': log_suffix
        },
        'quality': {
            'script': 'api_services/quality_api.py',
            'port': base_port + 2,
            'env': 'xks_qwen',
            'env_vars': {'CUDA_VISIBLE_DEVICES': gpu_id},
            'args': [
                '--port', str(base_port + 2),
                '--model_dir', '/root/siton-tmp/sx/xks/models/Qwen2.5-VL-7B-Instruct',
                '--prompt_path', '/root/siton-tmp/sx/xks/51_code/data_prepare/qwen_filter/image_pair_quality_prompt.txt'
            ],
            'log_suffix': log_suffix
        }
    }

class ServiceManager:
    """服务管理器"""
    
    def __init__(self, base_dir: str, api_version: str = 'original'):
        self.base_dir = base_dir
        self.api_version = api_version
        self.services = get_services_config(api_version)
        self.processes: Dict[str, subprocess.Popen] = {}
        logger.info(f"初始化服务管理器 - API版本: {api_version}")
    
    def start_service(self, service_name: str, service_config: Dict, output_dir: str) -> bool:
        """启动单个服务"""
        try:
            script_path = os.path.join(self.base_dir, service_config['script'])
            port = service_config['port']
            env = service_config['env']
            args = service_config['args'].copy()  # 复制一份，避免修改原始配置
            
            # 添加输出目录参数
            args.extend(['--output_dir', output_dir])
            
            # 获取环境变量
            env_vars = service_config.get('env_vars', {})
            
            # 创建日志文件（使用log_suffix）
            log_suffix = service_config.get('log_suffix', '')
            log_file = os.path.join(self.base_dir, f"{service_name}{log_suffix}_log.txt")
            
            # 构建命令
            if env == 'base':
                cmd = ['python', script_path] + args
            else:
                cmd = ['conda', 'run', '-n', env, 'python', script_path] + args
            
            logger.info(f"启动 {service_name} 服务 (端口: {port})")
            logger.info(f"命令: {' '.join(cmd)}")
            if env_vars:
                logger.info(f"环境变量: {env_vars}")
            
            # 启动进程
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=self.base_dir,
                    env={**os.environ, **env_vars}  # 合并环境变量
                )
            
            self.processes[service_name] = process
            logger.info(f"✅ {service_name} 服务启动成功 (PID: {process.pid})")
            
            # 等待服务启动
            time.sleep(5)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 启动 {service_name} 服务失败: {e}")
            return False
    
    def start_all_services(self, output_dir: str = "output") -> Dict[str, bool]:
        """启动所有服务"""
        results = {}
        
        for service_name, service_config in self.services.items():
            success = self.start_service(service_name, service_config, output_dir)
            results[service_name] = success
            
            if not success:
                logger.error(f"❌ {service_name} 服务启动失败，停止后续服务")
                break
        
        return results
    
    def stop_service(self, service_name: str):
        """停止单个服务"""
        if service_name in self.processes:
            process = self.processes[service_name]
            try:
                process.terminate()
                process.wait(timeout=10)
                logger.info(f"✅ {service_name} 服务已停止")
            except subprocess.TimeoutExpired:
                process.kill()
                logger.warning(f"⚠️ {service_name} 服务被强制终止")
            except Exception as e:
                logger.error(f"❌ 停止 {service_name} 服务失败: {e}")
        else:
            logger.warning(f"⚠️ 服务 {service_name} 不在当前进程列表中")
    
    def stop_all_services(self):
        """停止所有服务"""
        for service_name in self.services:
            self.stop_service(service_name)
    
    def check_service_health(self, service_name: str, port: int) -> bool:
        """检查服务健康状态（单次检查）"""
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def wait_for_service_healthy(self, service_name: str, port: int, max_wait: int = 30, interval: int = 5) -> bool:
        """等待服务健康，带重试机制
        
        Args:
            service_name: 服务名称
            port: 服务端口
            max_wait: 最大等待时间（秒）
            interval: 检查间隔（秒）
        
        Returns:
            服务是否健康
        """
        waited = 0
        while waited < max_wait:
            if self.check_service_health(service_name, port):
                return True
            logger.info(f"⏳ 等待 {service_name} 服务就绪... ({waited}/{max_wait}秒)")
            time.sleep(interval)
            waited += interval
        return self.check_service_health(service_name, port)
    
    def check_all_services_health(self, with_wait: bool = False) -> Dict[str, bool]:
        """检查所有服务健康状态
        
        Args:
            with_wait: 是否等待服务就绪（启动后使用）
        """
        health_status = {}
        
        # 不同服务的最大等待时间配置
        max_wait_times = {
            'preprocess': 30,
            'osediff': 120,  # osediff 需要加载大模型，等待更长
            'quality': 60
        }
        
        for service_name, service_config in self.services.items():
            port = service_config['port']
            
            if with_wait:
                max_wait = max_wait_times.get(service_name, 30)
                is_healthy = self.wait_for_service_healthy(service_name, port, max_wait=max_wait)
            else:
                is_healthy = self.check_service_health(service_name, port)
            
            health_status[service_name] = is_healthy
            
            status = "✅ 健康" if is_healthy else "❌ 异常"
            logger.info(f"{service_name} 服务 ({port}): {status}")
        
        return health_status

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='API服务管理器（统一版本）')
    parser.add_argument('--action', choices=['start', 'stop', 'status'], 
                       default='start', help='操作类型')
    parser.add_argument('--output_dir', type=str, default='output', help='输出根目录')
    parser.add_argument('--api_version', type=str, choices=['original', 'shared_left'], 
                       default='original', help='API版本选择: original(端口5000-5002, GPU 0), shared_left(端口5010-5012, GPU 1)')
    args = parser.parse_args()
    
    # 获取当前目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    manager = ServiceManager(base_dir, api_version=args.api_version)
    
    if args.action == 'start':
        logger.info(f"启动所有API服务 (版本: {args.api_version})...")
        results = manager.start_all_services(output_dir=args.output_dir)
        
        # 等待所有服务健康（带重试机制）
        logger.info("等待所有服务就绪...")
        health_status = manager.check_all_services_health(with_wait=True)
        
        all_healthy = all(health_status.values())
        if all_healthy:
            logger.info("✅ 所有API服务启动成功并健康")
        else:
            logger.error("❌ 部分API服务异常")
            log_suffix = '_shared_left' if args.api_version == 'shared_left' else ''
            for service, is_healthy in health_status.items():
                if not is_healthy:
                    logger.error(f"  - {service} 服务异常，请查看日志: {service}{log_suffix}_log.txt")
    
    elif args.action == 'stop':
        logger.info(f"停止所有API服务 (版本: {args.api_version})...")
        manager.stop_all_services()
    
    elif args.action == 'status':
        logger.info(f"检查API服务状态 (版本: {args.api_version})...")
        health_status = manager.check_all_services_health()
        
        all_healthy = all(health_status.values())
        if all_healthy:
            logger.info("✅ 所有API服务都健康")
        else:
            logger.error("❌ 部分API服务异常")

if __name__ == '__main__':
    main()
