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
import socket
from typing import Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加配置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from config.api_config import get_path

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
                '--osediff_path', get_path('osediff_model'),
                '--pretrained_model_name_or_path', get_path('pretrained_model'),
                '--ram_path', get_path('ram_model'),
                '--ram_ft_path', get_path('ram_ft_model'),
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
                '--model_dir', get_path('qwen_model'),
                '--prompt_path', get_path('quality_prompt')
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
        
        # 创建logs/start目录
        self.logs_dir = os.path.join(base_dir, 'logs', 'start')
        os.makedirs(self.logs_dir, exist_ok=True)
        
        logger.info(f"初始化服务管理器 - API版本: {api_version}")
        logger.info(f"日志目录: {self.logs_dir}")
    
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
            
            # 创建日志文件（放在logs/start目录下）
            log_suffix = service_config.get('log_suffix', '')
            log_file = os.path.join(self.logs_dir, f"{service_name}{log_suffix}_log.txt")
            
            # 构建命令（conda run 需加 --no-capture-output，否则会吞掉子进程 stdout/stderr，导致 *_log.txt 为空）
            if env == 'base':
                cmd = ['python', script_path] + args
            else:
                cmd = ['conda', 'run', '-n', env, '--no-capture-output', 'python', script_path] + args
            
            logger.info(f"启动 {service_name} 服务 (端口: {port})")
            logger.info(f"命令: {' '.join(cmd)}")
            if env_vars:
                logger.info(f"环境变量: {env_vars}")
            
            # 启动进程：PYTHONUNBUFFERED=1 便于子进程日志实时写入；不能使用 with 关闭 log_fd，否则子进程无法写入
            log_fd = open(log_file, 'w')
            process = subprocess.Popen(
                cmd,
                stdout=log_fd,
                stderr=subprocess.STDOUT,
                cwd=self.base_dir,
                env={**os.environ, 'PYTHONUNBUFFERED': '1', **env_vars}
            )
            
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
        """停止单个服务（仅通过进程名+端口查找并强制停止）"""
        service_config = self.services.get(service_name)
        if not service_config:
            logger.warning(f"⚠️ 未知服务: {service_name}")
            return
        
        script_name = os.path.basename(service_config['script'])
        port = service_config['port']
        cmd = f"ps aux | grep '{script_name}' | grep 'port {port}' | grep -v grep | awk '{{print $2}}' | xargs -r kill -9"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"✅ {service_name} 服务已停止")
        elif result.returncode == 123:
            logger.info(f"ℹ️ {service_name} 服务未在运行")
        else:
            logger.info(f"✅ {service_name} 服务已停止")
    
    def stop_all_services(self):
        """停止所有服务"""
        logger.info("正在停止所有服务...")
        for service_name in self.services:
            self.stop_service(service_name)
    
    def is_port_open(self, port: int, timeout: float = 0.5) -> bool:
        """快速检查端口是否开放（避免HTTP超时）"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            return result == 0
        except Exception:
            return False
    
    def check_service_health(self, service_name: str, port: int) -> bool:
        """检查服务健康状态（单次检查）
        
        优化：
        1. 先快速检查端口是否开放（0.5秒）
        2. 如果端口未开放，立即返回 False
        3. 如果端口开放，再发送 HTTP 请求（1秒超时）
        """
        # 快速端口检查（避免5秒HTTP超时）
        if not self.is_port_open(port, timeout=0.5):
            return False
        
        # 端口开放，检查健康状态
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=1)
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
    
    def check_all_services_health(self, with_wait: bool = False, parallel: bool = True) -> Dict[str, bool]:
        """检查所有服务健康状态
        
        Args:
            with_wait: 是否等待服务就绪（启动后使用）
            parallel: 是否并发检查（默认True，加快速度）
        """
        health_status = {}
        
        # 不同服务的最大等待时间配置
        max_wait_times = {
            'preprocess': 30,
            'osediff': 120,  # osediff 需要加载大模型，等待更长
            'quality': 60
        }
        
        if parallel and not with_wait:
            # 并发检查所有服务（仅用于状态查询，不等待）
            def check_single_service(service_name, port):
                is_healthy = self.check_service_health(service_name, port)
                status = "✅ 健康" if is_healthy else "❌ 异常"
                logger.info(f"{service_name} 服务 (http://localhost:{port}) - {status}")
                return service_name, is_healthy
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {}
                for service_name, service_config in self.services.items():
                    port = service_config['port']
                    future = executor.submit(check_single_service, service_name, port)
                    futures[future] = service_name
                
                for future in as_completed(futures):
                    service_name, is_healthy = future.result()
                    health_status[service_name] = is_healthy
        else:
            # 串行检查（用于启动后等待）
            for service_name, service_config in self.services.items():
                port = service_config['port']
                
                if with_wait:
                    max_wait = max_wait_times.get(service_name, 30)
                    is_healthy = self.wait_for_service_healthy(service_name, port, max_wait=max_wait)
                else:
                    is_healthy = self.check_service_health(service_name, port)
                
                health_status[service_name] = is_healthy
                
                status = "✅ 健康" if is_healthy else "❌ 异常"
                logger.info(f"{service_name} 服务 (http://localhost:{port}) - {status}")
        
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
        manager.start_all_services(output_dir=args.output_dir)
        
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
                    logger.error(f"  - {service} 服务异常，请查看日志: logs/start/{service}{log_suffix}_log.txt")
    
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
