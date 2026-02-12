"""
图像质量评估API服务
基于batch_image_scoring.py
"""

import os
import sys
import argparse
import logging
import csv
import time
import json
import numpy as np
import torch
from flask import Flask, request, jsonify
from PIL import Image
from typing import List, Dict

# 添加项目路径（必须在 from config 之前）
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(current_dir, '..')
sys.path.insert(0, project_dir)

from config.api_config import get_path

# 添加qwen_filter路径
qwen_filter_path = get_path('qwen_filter')
sys.path.append(qwen_filter_path)

try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    logger = logging.getLogger(__name__)
    logger.info(f"✅ Qwen模块导入成功")
    logger.info(f"   qwen_filter路径: {qwen_filter_path}")
except ImportError as e:
    logging.error(f"无法导入Qwen模块: {e}")
    logging.error(f"当前路径: {os.getcwd()}")
    logging.error(f"qwen_filter路径: {qwen_filter_path}")
    logging.error(f"Python路径: {sys.path}")
    sys.exit(1)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class QwenQualityEvaluator:
    """基于Qwen2.5-VL的图片质量评估器 - 基于batch_image_scoring.py"""
    
    def __init__(self, model_dir, prompt_path, output_dir):
        self.model_dir = model_dir
        self.prompt_path = prompt_path
        self.output_dir = output_dir
        self.model = None
        self.processor = None
        self.system_prompt = None
        
        # 初始化模型和提示词
        self._init_model()
        self._load_prompt()
    
    def _init_model(self):
        """初始化Qwen模型"""
        logger.info("正在加载Qwen2.5-VL模型...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_dir, 
            torch_dtype="auto", 
            device_map="auto",
            local_files_only=True,
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_dir,
            local_files_only=True,
            trust_remote_code=True
        )
        logger.info("✅ Qwen2.5-VL模型加载完成")
    
    def _load_prompt(self):
        """加载质量评估提示词"""
        prompt_file = self.prompt_path
        if not os.path.isabs(self.prompt_path):
            prompt_file = os.path.join(qwen_filter_path, self.prompt_path)
        
        with open(prompt_file, "r", encoding='utf-8') as f:
            self.system_prompt = f.read()
        logger.info(f"✅ 质量评估提示词加载完成: {prompt_file}")
    
    def _parse_model_output(self, reply):
        """解析模型输出"""
        lines = reply.split("\n")
        scores = [None] * 5
        final_score = None
        
        for line in lines:
            line_lower = line.lower()
            if line_lower.startswith("correlation"):
                scores[0] = line.split(":")[1].split("Reason")[0].strip().strip("[] ")
            elif line_lower.startswith("information"):
                scores[1] = line.split(":")[1].split("Reason")[0].strip().strip("[] ")
            elif line_lower.startswith("subject"):
                scores[2] = line.split(":")[1].split("Reason")[0].strip().strip("[] ")
            elif line_lower.startswith("image clarity"):
                scores[3] = line.split(":")[1].split("Reason")[0].strip().strip("[] ")
            elif line_lower.startswith("content coherence"):
                scores[4] = line.split(":")[1].split("Reason")[0].strip().strip("[] ")
            elif line_lower.startswith("final score"):
                final_score = line.split(":")[1].strip().strip("[] ")
        
        return {
            'correlation': scores[0], 'balance': scores[1], 'richness': scores[2],
            'clarity': scores[3], 'coherence': scores[4],
            'final_score': final_score, 'raw_output': reply
        }
    
    def evaluate_single_image(self, image_path):
        """评估单张图像质量"""
        logger.info(f"开始评估图像质量: {image_path}")
        start_time = time.time()
        
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": [{"type": "image", "image": f"file://{os.path.abspath(image_path)}"}]}
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            reply = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        scores = self._parse_model_output(reply)
        scores['image_path'] = image_path
        scores['processing_time'] = time.time() - start_time
        
        logger.info(f"图像质量评估完成: {image_path} - 最终分数: {scores['final_score']}")
        return scores
    
    def evaluate_image_pairs(self, pairs_data: List[Dict]) -> List[Dict]:
        """评估图像对列表的质量
        
        只返回下游需要的字段：
        - pair_image: 图像路径
        - interval: 区间ID
        - final_score: 最终质量分数
        - main_params, rand_params: 参数信息
        - yaw_interval: yaw区间
        """
        results = []
        # csv_results = []  # 用于保存到CSV的完整结果 - 暂时不需要CSV
        
        for i, pair_data in enumerate(pairs_data):
            image_path = pair_data.get('pair_image') or pair_data.get('super_resolved')
            
            if not image_path:
                logger.warning(f"第 {i+1} 个图像对缺少图像路径，跳过。")
                continue
                
            try:
                scores = self.evaluate_single_image(image_path)
                
                # 用于返回给调用方的精简结果
                result_entry = {
                    'pair_image': image_path,
                    'interval': pair_data.get('interval'),
                    'final_score': scores.get('final_score'),
                    'main_params': pair_data.get('main_params'),
                    'rand_params': pair_data.get('rand_params'),
                    'yaw_interval': pair_data.get('yaw_interval')
                }
                results.append(result_entry)
                
                # [已注释] 用于保存到CSV的完整结果（包含所有维度分数）
                # csv_entry = {
                #     'image_path': image_path,
                #     'correlation': scores.get('correlation'),
                #     'balance': scores.get('balance'),
                #     'richness': scores.get('richness'),
                #     'clarity': scores.get('clarity'),
                #     'coherence': scores.get('coherence'),
                #     'final_score': scores.get('final_score')
                # }
                # csv_results.append(csv_entry)
                
            except Exception as e:
                logger.error(f"评估图像对失败: {image_path} - {e}")
                results.append({"error": str(e), "image_path": image_path})
                # csv_results.append({"error": str(e), "image_path": image_path})
        
        # [已注释] 保存CSV文件（使用完整的评估结果）
        # if csv_results:
        #     csv_path = os.path.join(self.output_dir, "quality_results.csv")
        #     self._save_results_to_csv(csv_results, csv_path)
            
        return results
    
    def _save_results_to_csv(self, results, csv_path):
        """保存结果到CSV文件"""
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        file_exists = os.path.exists(csv_path)
        
        score_fields = ['correlation', 'balance', 'richness', 'clarity', 'coherence', 'final_score']
        
        with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(["filename".ljust(30)] + [f.ljust(12) for f in score_fields])
            
            for result in results:
                filename = os.path.basename(result.get('image_path', 'unknown'))
                if 'error' not in result:
                    row = [filename.ljust(30)] + [str(result.get(f, '')).ljust(12) for f in score_fields]
                else:
                    row = [filename.ljust(30)] + ["error".ljust(12)] * 6
                writer.writerow(row)
        
        logger.info(f"质量评估结果已保存: {csv_path}")
    
    def generate_image_description(self, image_path: str) -> Dict:
        """为单张图像生成文本描述"""
        logger.info(f"生成图像描述: {image_path}")
        
        description_prompt = """Please provide a detailed description of this image. Include:
1. Main subjects and objects in the scene
2. Environment and setting (indoor/outdoor, location type)
3. Visual characteristics (colors, lighting, weather if applicable)
4. Spatial layout and composition
5. Any notable features or points of interest

Provide a concise but comprehensive description in 2-3 sentences."""
        
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": description_prompt}
        ]}]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=256)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            description = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        
        logger.info(f"描述生成完成: {image_path}")
        return {"image_path": image_path, "description": description.strip(), "timestamp": time.time()}
    
    def generate_batch_descriptions(self, image_paths: List[str]) -> List[Dict]:
        """批量生成图像描述"""
        logger.info(f"开始批量生成描述，共 {len(image_paths)} 张图像")
        
        results = []
        for i, image_path in enumerate(image_paths):
            logger.info(f"[{i+1}/{len(image_paths)}] 生成描述: {image_path}")
            result = self.generate_image_description(image_path)
            results.append(result)
        
        logger.info(f"批量描述生成完成")
        return results

# 全局评估器实例
evaluator = None

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({"status": "healthy", "service": "quality"})

@app.route('/evaluate_pairs', methods=['POST'])
def evaluate_pairs():
    """评估图像对质量"""
    try:
        data = request.get_json()
        pairs_data = data.get('pairs_data', [])
        
        if not pairs_data:
            return jsonify({"error": "缺少pairs_data参数"}), 400
        
        # --- [修改] 移除了多余的 output_dir 参数 ---
        results = evaluator.evaluate_image_pairs(pairs_data)
        return jsonify({"success": True, "results": results})
        
    except Exception as e:
        logger.error(f"图像对质量评估失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate_descriptions', methods=['POST'])
def generate_descriptions():
    """批量生成图像描述"""
    try:
        data = request.get_json()
        image_paths = data.get('image_paths', [])
        
        if not image_paths:
            return jsonify({"error": "缺少image_paths参数"}), 400
        
        results = evaluator.generate_batch_descriptions(image_paths)
        return jsonify({"success": True, "results": results})
        
    except Exception as e:
        logger.error(f"批量描述生成失败: {e}")
        return jsonify({"error": str(e)}), 500

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='图像质量评估API服务')
    parser.add_argument('--port', type=int, default=5002, help='服务端口')
    parser.add_argument('--output_dir', type=str, default='output/quality', help='输出目录')
    parser.add_argument('--model_dir', type=str, required=True, help='Qwen模型目录')
    parser.add_argument('--prompt_path', type=str, required=True, help='提示词文件路径')
    args = parser.parse_args()
    
    # 初始化评估器
    global evaluator
    evaluator = QwenQualityEvaluator(args.model_dir, args.prompt_path, args.output_dir)
    
    logger.info(f"图像质量评估API服务启动 - 端口: {args.port}")
    app.run(host='0.0.0.0', port=args.port, debug=False)

if __name__ == '__main__':
    main() 