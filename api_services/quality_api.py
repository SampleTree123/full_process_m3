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
from flask import Flask, request, jsonify
from PIL import Image
import json
from typing import List, Dict
import numpy as np
import torch

# 添加qwen_filter路径 - 修复路径问题
current_dir = os.path.dirname(os.path.abspath(__file__))
qwen_filter_path = os.path.join(current_dir, '..', '..', 'qwen_filter')
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
        """初始化Qwen模型 - 基于batch_image_scoring.py"""
        try:
            logger.info("正在加载Qwen2.5-VL模型...")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_dir, 
                torch_dtype="auto", 
                device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(self.model_dir)
            logger.info("✅ Qwen2.5-VL模型加载完成")
            
        except Exception as e:
            logger.error(f"❌ 模型初始化失败: {str(e)}")
            raise
    
    def _load_prompt(self):
        """加载质量评估提示词"""
        try:
            # 修复提示词文件路径
            if not os.path.isabs(self.prompt_path):
                # 如果是相对路径，从qwen_filter目录开始
                prompt_file = os.path.join(qwen_filter_path, self.prompt_path)
            else:
                prompt_file = self.prompt_path
                
            logger.info(f"加载提示词文件: {prompt_file}")
            
            with open(prompt_file, "r", encoding='utf-8') as f:
                self.system_prompt = f.read()
            logger.info("✅ 质量评估提示词加载完成")
            
        except Exception as e:
            logger.error(f"❌ 提示词加载失败: {str(e)}")
            logger.error(f"尝试的路径: {prompt_file}")
            raise
    
    def _parse_model_output(self, reply):
        """解析模型输出 - 基于batch_image_scoring.py"""
        try:
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
                'correlation': scores[0],
                'balance': scores[1],
                'richness': scores[2],
                'clarity': scores[3],
                'coherence': scores[4],
                'final_score': final_score,
                'raw_output': reply
            }
            
        except Exception as e:
            logger.error(f"解析模型输出失败: {e}")
            return {
                'correlation': None,
                'information_balance': None,
                'subject_richness': None,
                'image_clarity': None,
                'content_coherence': None,
                'final_score': None,
                'raw_output': reply
            }
    
    def evaluate_single_image(self, image_path):
        """评估单张图像质量 - 基于batch_image_scoring.py"""
        try:
            if not os.path.exists(image_path):
                raise ValueError(f"图像不存在: {image_path}")
            
            logger.info(f"开始评估图像质量: {image_path}")
            start_time = time.time()
            
            # 构建消息
            messages = [
                {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
                {"role": "user", "content": [{"type": "image", "image": f"file://{os.path.abspath(image_path)}"}]}
            ]
            
            # 处理输入
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)
            
            # 生成评估结果
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=512)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
                ]
                reply = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            
            # 解析结果
            scores = self._parse_model_output(reply)
            scores['image_path'] = image_path
            scores['processing_time'] = time.time() - start_time
            
            logger.info(f"图像质量评估完成: {image_path} - 最终分数: {scores['final_score']}")
            return scores
            
        except Exception as e:
            logger.error(f"评估图像质量失败: {e}")
            raise
    
    def evaluate_image_pairs(self, pairs_data: List[Dict]) -> List[Dict]:
        """评估图像对列表的质量"""
        results = []
        for i, pair_data in enumerate(pairs_data):
            image_path = pair_data.get('pair_image') or pair_data.get('super_resolved')
            
            if not image_path:
                logger.warning(f"第 {i+1} 个图像对缺少图像路径，跳过。")
                continue
                
            try:
                scores = self.evaluate_single_image(image_path)
                result_entry = pair_data.copy()
                result_entry.update(scores)
                results.append(result_entry)
            except Exception as e:
                logger.error(f"评估图像对失败: {image_path} - {e}")
                results.append({"error": str(e), "image_path": image_path})
        
        # --- [修改] 使用 self.output_dir 来构建保存路径 ---
        if results:
            # os.path.join 会正确处理路径，即使 self.output_dir 为空
            csv_path = os.path.join(self.output_dir, "quality_results.csv")
            self._save_results_to_csv(results, csv_path)
            
        return results
    
    def _save_results_to_csv(self, results, csv_path):
        """保存结果到CSV文件 - 保存所有分数项"""
        try:
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            
            # 检查文件是否存在，如果不存在则写入表头
            file_exists = os.path.exists(csv_path)
            
            with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # 如果文件不存在，写入表头
                if not file_exists:
                    header = [
                        "filename".ljust(30),
                        "correlation".ljust(12),
                        "balance".ljust(12),
                        "richness".ljust(12),
                        "clarity".ljust(12),
                        "coherence".ljust(12),
                        "final_score".ljust(12)
                    ]
                    writer.writerow(header)
                
                # 写入数据
                for result in results:
                    if 'error' not in result:
                        filename = os.path.basename(result.get('image_path', ''))
                        correlation = result.get('correlation', '')
                        balance = result.get('balance', '')
                        richness = result.get('richness', '')
                        clarity = result.get('clarity', '')
                        coherence = result.get('coherence', '')
                        final_score = result.get('final_score', '')
                        
                        row = [
                            filename.ljust(30),
                            str(correlation).ljust(12),
                            str(balance).ljust(12),
                            str(richness).ljust(12),
                            str(clarity).ljust(12),
                            str(coherence).ljust(12),
                            str(final_score).ljust(12)
                        ]
                        writer.writerow(row)
                    else:
                        # 如果有错误，记录错误信息
                        filename = os.path.basename(result.get('image_path', 'unknown'))
                        row = [filename.ljust(30)] + ["error".ljust(12)] * 6
                        writer.writerow(row)
            
            logger.info(f"质量评估结果已保存: {csv_path}")
            
        except Exception as e:
            logger.error(f"保存CSV失败: {e}")
    
    def generate_image_description(self, image_path: str) -> Dict:
        """为单张图像生成文本描述"""
        try:
            if not os.path.exists(image_path):
                raise ValueError(f"图像不存在: {image_path}")
            
            logger.info(f"生成图像描述: {image_path}")
            
            # 准备描述生成的提示词
            description_prompt = """Please provide a detailed description of this image. Include:
1. Main subjects and objects in the scene
2. Environment and setting (indoor/outdoor, location type)
3. Visual characteristics (colors, lighting, weather if applicable)
4. Spatial layout and composition
5. Any notable features or points of interest

Provide a concise but comprehensive description in 2-3 sentences."""
            
            # 构建消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": description_prompt}
                    ]
                }
            ]
            
            # 准备输入
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            
            # 生成描述
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=256)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                description = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
            
            result = {
                "image_path": image_path,
                "description": description.strip(),
                "timestamp": time.time()
            }
            
            logger.info(f"描述生成完成: {image_path}")
            return result
            
        except Exception as e:
            logger.error(f"描述生成失败: {image_path} - {e}")
            return {
                "image_path": image_path,
                "error": str(e),
                "timestamp": time.time()
            }
    
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