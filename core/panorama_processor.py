"""
全景图像处理器核心类
协调所有API服务完成完整的处理流程
"""

import os
import sys
import logging
import time
import json
from typing import List, Dict, Any
from PIL import Image

# 添加配置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, '..', 'config')
sys.path.append(config_path)

from utils.api_utils import APIClient

logger = logging.getLogger(__name__)

class PanoramaProcessor:
    """全景图像处理器 - 协调所有API服务"""
    
    def __init__(self, output_root_dir: str = "output", api_version: str = "original", output_file: str = "results.json"):
        """
        初始化处理器
        
        Args:
            output_root_dir: 输出根目录
            api_version: API版本选择，'original' 或 'shared_left'
            output_file: 输出文件名（默认 results.json）
        """
        self.output_root_dir = output_root_dir
        self.api_version = api_version
        self.output_file = output_file
        
        # 使用统一配置
        from config.api_config import get_api_ports, get_gpu_id, API_BASE_URL
        
        self.API_PORTS = get_api_ports(api_version)
        self.API_BASE_URL = API_BASE_URL
        gpu_id = get_gpu_id(api_version)
        
        logger.info(f"🔄 使用 {api_version} 版本API (端口 {self.API_PORTS['preprocess']}-{self.API_PORTS['quality']}, GPU {gpu_id})")
        
        self.api_client = APIClient(API_BASE_URL)
        self.setup_output_dirs()
    
    def setup_output_dirs(self):
        """设置输出目录 - 不再需要全局目录，每个组有自己的目录结构"""
        # 只需要确保输出根目录存在
        os.makedirs(self.output_root_dir, exist_ok=True)
        logger.info(f"创建输出根目录: {self.output_root_dir}")
    
    def super_resolve_pairs(self, pairs_data: List[Dict]) -> List[Dict]:
        """超分辨率处理图像对 - 优化版：先切分再批量超分"""
        logger.info(f"开始超分辨率处理 {len(pairs_data)} 个图像对")
        
        # preprocess API返回的是一个包含多个图像对的列表，需要展开
        osediff_pairs_data = []
        for panorama_result in pairs_data:
            # 每个panorama_result包含多个图像对
            if isinstance(panorama_result, list):
                # 如果返回的是列表，直接使用
                for pair_data in panorama_result:
                    if 'pair_image' in pair_data:
                        osediff_pairs_data.append({
                            'pair_image': pair_data['pair_image'],
                            # 保留原始参数，以便后续使用
                            'main_params': pair_data.get('main_params', {}),
                            'rand_params': pair_data.get('rand_params', {}),
                            'main_params_file': pair_data.get('main_params_file', ''),
                            'rand_params_file': pair_data.get('rand_params_file', ''),
                            'interval': pair_data.get('interval', 0),
                            'yaw_interval': pair_data.get('yaw_interval', [])
                        })
            elif isinstance(panorama_result, dict) and 'pair_image' in panorama_result:
                # 如果返回的是单个字典
                osediff_pairs_data.append({
                    'pair_image': panorama_result['pair_image'],
                    # 保留原始参数，以便后续使用
                    'main_params': panorama_result.get('main_params', {}),
                    'rand_params': panorama_result.get('rand_params', {}),
                    'main_params_file': panorama_result.get('main_params_file', ''),
                    'rand_params_file': panorama_result.get('rand_params_file', ''),
                    'interval': panorama_result.get('interval', 0),
                    'yaw_interval': panorama_result.get('yaw_interval', [])
                })
        
        logger.info(f"转换为OSEDiff格式: {len(osediff_pairs_data)} 个图像对")
        
        # 优化：先切分所有图像对成单图，然后批量超分，最后再拼接
        try:
            from PIL import Image
            import re
            
            # 第一步：切分所有图像对并收集单图
            all_single_images = []  # 存储所有单图的信息
            for pair_data in osediff_pairs_data:
                pair_image_path = pair_data['pair_image']
                
                if not os.path.exists(pair_image_path):
                    logger.warning(f"图像对不存在: {pair_image_path}")
                    continue
                
                # 读取并切分图像对
                pair_img = Image.open(pair_image_path)
                width, height = pair_img.size
                half_width = width // 2
                
                # 切分左右图
                left_img = pair_img.crop((0, 0, half_width, height))
                right_img = pair_img.crop((half_width, 0, width, height))
                
                # 保存临时切分图像
                base_name = os.path.splitext(os.path.basename(pair_image_path))[0]
                temp_dir = os.path.join(os.path.dirname(pair_image_path), 'temp_split')
                os.makedirs(temp_dir, exist_ok=True)
                
                left_temp_path = os.path.join(temp_dir, f"{base_name}_left_temp.jpg")
                right_temp_path = os.path.join(temp_dir, f"{base_name}_right_temp.jpg")
                
                left_img.save(left_temp_path, quality=95)
                right_img.save(right_temp_path, quality=95)
                
                # 记录单图信息
                all_single_images.append({
                    'input_path': left_temp_path,
                    'type': 'left',
                    'pair_data': pair_data,
                    'original_pair_path': pair_image_path
                })
                all_single_images.append({
                    'input_path': right_temp_path,
                    'type': 'right',
                    'pair_data': pair_data,
                    'original_pair_path': pair_image_path
                })
            
            logger.info(f"切分完成，共 {len(all_single_images)} 张单图待超分")
            
            # 第二步：批量超分所有单图
            # 提取group目录用于保存超分结果
            if all_single_images:
                first_pair_path = all_single_images[0]['original_pair_path']
                match = re.search(r'/group_\d+/', first_pair_path)
                if match:
                    group_dir_match = match.group(0)
                    group_root_dir = first_pair_path[:first_pair_path.find(group_dir_match) + len(group_dir_match) - 1]
                    group_root_dir = os.path.abspath(group_root_dir)
                    
                    # 创建临时超分输出目录
                    sr_temp_dir = os.path.join(group_root_dir, 'temp_sr')
                    os.makedirs(sr_temp_dir, exist_ok=True)
                    
                    # 批量调用超分API
                    for img_info in all_single_images:
                        response = self.api_client.call_api(
                            port=self.API_PORTS['osediff'],
                            endpoint='super_resolution',
                            data={
                                'input_path': img_info['input_path'],
                                'output_dir': group_root_dir,
                                'align_method': 'adain'
                            }
                        )
                        
                        if response.get('success'):
                            output_path = response.get('output_path')
                            # 移动到临时目录
                            import shutil
                            filename = os.path.basename(output_path)
                            target_path = os.path.join(sr_temp_dir, filename)
                            shutil.move(output_path, target_path)
                            img_info['sr_path'] = target_path
                        else:
                            logger.error(f"单图超分失败: {img_info['input_path']}")
                            img_info['sr_path'] = None
                    
                    logger.info(f"批量超分完成")
                    
                    # 第三步：重新拼接成图像对
                    results = []
                    pair_dict = {}  # 用于按原始图像对分组
                    
                    for img_info in all_single_images:
                        pair_path = img_info['original_pair_path']
                        if pair_path not in pair_dict:
                            pair_dict[pair_path] = {'left': None, 'right': None, 'pair_data': img_info['pair_data']}
                        
                        pair_dict[pair_path][img_info['type']] = img_info.get('sr_path')
                    
                    # 拼接左右图
                    for pair_path, pair_info in pair_dict.items():
                        left_sr = pair_info['left']
                        right_sr = pair_info['right']
                        
                        if left_sr and right_sr and os.path.exists(left_sr) and os.path.exists(right_sr):
                            # 读取超分后的左右图
                            left_img = Image.open(left_sr)
                            right_img = Image.open(right_sr)
                            
                            # 拼接
                            total_width = left_img.width + right_img.width
                            total_height = max(left_img.height, right_img.height)
                            merged_img = Image.new('RGB', (total_width, total_height))
                            merged_img.paste(left_img, (0, 0))
                            merged_img.paste(right_img, (left_img.width, 0))
                            
                            # 保存拼接后的图像对到osediff目录
                            base_name = os.path.splitext(os.path.basename(pair_path))[0]
                            osediff_dir = os.path.join(group_root_dir, 'osediff')
                            os.makedirs(osediff_dir, exist_ok=True)
                            output_pair_path = os.path.join(osediff_dir, f"{base_name}.jpg")
                            merged_img.save(output_pair_path, quality=95)
                            
                            # 构建结果
                            result_entry = pair_info['pair_data'].copy()
                            result_entry['super_resolved'] = output_pair_path
                            result_entry['align_method'] = 'adain'
                            results.append(result_entry)
                            
                            logger.info(f"图像对超分完成: {output_pair_path}")
                        else:
                            logger.warning(f"跳过不完整的图像对: {pair_path}")
                    
                    # 清理临时文件
                    import shutil
                    temp_dir = os.path.join(os.path.dirname(first_pair_path), 'temp_split')
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                    if os.path.exists(sr_temp_dir):
                        shutil.rmtree(sr_temp_dir)
                    
                    logger.info(f"超分辨率处理完成，生成了 {len(results)} 个结果")
                    return results
                else:
                    logger.error("无法从路径提取组目录")
                    return []
            else:
                logger.warning("没有图像对需要超分")
                return []
                
        except Exception as e:
            logger.error(f"超分辨率处理失败: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def evaluate_quality(self, pairs_data: List[Dict]) -> List[Dict]:
        """评估图像对质量"""
        logger.info(f"开始质量评估 {len(pairs_data)} 个图像对")
        
        # 转换OSEDiff返回的数据格式为Quality API期望的格式
        quality_pairs_data = []
        for pair_data in pairs_data:
            if 'super_resolved' in pair_data:
                # OSEDiff返回的格式，转换为Quality API期望的格式
                quality_pairs_data.append({
                    'pair_image': pair_data['super_resolved'],
                    'original_pair': pair_data.get('original_pair', ''),
                    'align_method': pair_data.get('align_method', 'adain'),
                    # 保留原始参数
                    'main_params': pair_data.get('main_params', {}),
                    'rand_params': pair_data.get('rand_params', {}),
                    'main_params_file': pair_data.get('main_params_file', ''),
                    'rand_params_file': pair_data.get('rand_params_file', ''),
                    'interval': pair_data.get('interval', 0),
                    'yaw_interval': pair_data.get('yaw_interval', [])
                })
            elif 'pair_image' in pair_data:
                # 已经是Quality API期望的格式，保留所有原始字段
                quality_pairs_data.append(pair_data)
        
        logger.info(f"转换为Quality API格式: {len(quality_pairs_data)} 个图像对")
        
        try:
            response = self.api_client.call_api(
                port=self.API_PORTS['quality'],
                endpoint='evaluate_pairs',
                data={'pairs_data': quality_pairs_data}
            )
            
            if response.get('success'):
                results = response.get('results', [])
                logger.info(f"质量评估完成，评估了 {len(results)} 个图像对")
                return results
            else:
                logger.error(f"质量评估失败: {response.get('error')}")
                return []
                
        except Exception as e:
            logger.error(f"质量评估API调用失败: {e}")
            return []
    
    def generate_panorama2_descriptions(self, filtered_results: List[Dict], panorama2_path: str, group_id: int) -> Dict:
        """为 panorama2 的右图生成文本描述"""
        logger.info(f"开始为 panorama2 的右图生成描述")
        
        from PIL import Image
        import re
        
        # 提取 panorama2 的文件名（不含扩展名）
        panorama2_basename = os.path.splitext(os.path.basename(panorama2_path))[0]
        
        # 收集 panorama2 的图像对并提取右图
        panorama2_right_images = []
        group_root_dir = None
        
        for result in filtered_results:
            pair_image_path = result.get('image_path') or result.get('pair_image')
            if not pair_image_path:
                continue
            
            # 判断是否属于 panorama2
            if panorama2_basename in os.path.basename(pair_image_path):
                try:
                    # 读取图像对
                    pair_img = Image.open(pair_image_path)
                    width, height = pair_img.size
                    half_width = width // 2
                    
                    # 提取右图
                    right_img = pair_img.crop((half_width, 0, width, height))
                    
                    # 保存临时右图
                    base_name = os.path.splitext(os.path.basename(pair_image_path))[0]
                    
                    # 提取组目录
                    match = re.search(r'/group_\d+/', pair_image_path)
                    if match:
                        group_dir_match = match.group(0)
                        group_root_dir = pair_image_path[:pair_image_path.find(group_dir_match) + len(group_dir_match) - 1]
                        group_root_dir = os.path.abspath(group_root_dir)
                        
                        # 创建描述目录
                        desc_dir = os.path.join(group_root_dir, 'descriptions')
                        os.makedirs(desc_dir, exist_ok=True)
                        
                        # 保存右图
                        right_img_path = os.path.join(desc_dir, f"{base_name}_right.jpg")
                        right_img.save(right_img_path, quality=95)
                        
                        panorama2_right_images.append({
                            'image_path': right_img_path,
                            'original_pair': pair_image_path,
                            'interval': result.get('interval'),
                            'yaw_interval': result.get('yaw_interval')
                        })
                        
                        logger.info(f"提取 panorama2 右图: {right_img_path}")
                    
                except Exception as e:
                    logger.error(f"提取右图失败: {pair_image_path} - {e}")
                    continue
        
        logger.info(f"共提取了 {len(panorama2_right_images)} 张 panorama2 的右图")
        
        # 批量生成描述
        if panorama2_right_images:
            try:
                # 提取所有图像路径
                image_paths = [img['image_path'] for img in panorama2_right_images]
                
                # 调用 Quality API 生成描述
                response = self.api_client.call_api(
                    port=self.API_PORTS['quality'],
                    endpoint='generate_descriptions',
                    data={'image_paths': image_paths}
                )
                
                if response.get('success'):
                    descriptions = response.get('results', [])
                    
                    # 将描述与图像信息关联
                    for img_info, desc_result in zip(panorama2_right_images, descriptions):
                        img_info['description'] = desc_result.get('description', '')
                        img_info['description_error'] = desc_result.get('error', '')
                    
                    # 保存描述到 JSON 文件
                    if group_root_dir:
                        desc_json_path = os.path.join(group_root_dir, 'descriptions', 'panorama2_right_descriptions.json')
                        with open(desc_json_path, 'w', encoding='utf-8') as f:
                            json.dump(panorama2_right_images, f, indent=2, ensure_ascii=False)
                        logger.info(f"描述已保存: {desc_json_path}")
                    
                    logger.info(f"描述生成完成，共 {len(descriptions)} 条")
                    return {
                        'success': True,
                        'descriptions': panorama2_right_images,
                        'count': len(panorama2_right_images)
                    }
                else:
                    logger.error(f"描述生成失败: {response.get('error')}")
                    return {'success': False, 'error': response.get('error')}
                    
            except Exception as e:
                logger.error(f"描述生成API调用失败: {e}")
                return {'success': False, 'error': str(e)}
        else:
            logger.warning("没有 panorama2 的右图需要生成描述")
            return {'success': True, 'descriptions': [], 'count': 0}
    
    def process_image_group(self, panorama_path1: str, panorama_path2: str, group_id: int) -> Dict:
        """处理图片组（两张全景图）"""
        logger.info(f"开始处理图片组 #{group_id}: {os.path.basename(panorama_path1)} + {os.path.basename(panorama_path2)}")
        start_time = time.time()
        
        try:
            # 1. 预处理两张全景图，并确保右图参数一致
            preprocess_results1 = self.preprocess_panoramas_for_group(panorama_path1, group_id, is_first=True)
            preprocess_results2 = self.preprocess_panoramas_for_group(panorama_path2, group_id, is_first=False)
            
            if not preprocess_results1 or not preprocess_results2:
                raise Exception("预处理失败")
            
            # 合并两组预处理结果
            all_preprocess_results = preprocess_results1 + preprocess_results2
            
            # 2. 超分辨率处理
            sr_results = self.super_resolve_pairs(all_preprocess_results)
            if not sr_results:
                raise Exception("超分辨率处理失败")
            
            # 3. 质量评估
            quality_results = self.evaluate_quality(sr_results)
            if not quality_results:
                raise Exception("质量评估失败")
            
            # 4. 质量过滤（按组取最低分）
            filtered_results = self.filter_high_quality_results_by_group(quality_results, threshold=0.7)
            if not filtered_results:
                logger.warning(f"图片组 #{group_id} 没有高质量图像")
                return {
                    "group_id": group_id,
                    "panorama1": panorama_path1,
                    "panorama2": panorama_path2,
                    "error": "没有高质量图像",
                    "processing_time": time.time() - start_time,
                    "success": False
                }
            
            # 4.5. 为 panorama2 的右图生成描述
            desc_result = self.generate_panorama2_descriptions(filtered_results, panorama_path2, group_id)
            if desc_result.get('success'):
                logger.info(f"成功生成 {desc_result.get('count', 0)} 条图像描述")
            else:
                logger.warning(f"描述生成失败: {desc_result.get('error', 'unknown')}")
            
            # 5. 切分图像对并执行参数插值
            logger.info(f"开始切分图像对并执行参数插值，传入 {len(filtered_results)} 个图像对")
            if filtered_results:
                logger.info(f"第一个图像对的数据示例: {filtered_results[0]}")
            
            # 切分图像对为左右图
            split_results = self.split_pairs_for_interpolation(filtered_results, group_id)
            if not split_results:
                raise Exception("切分图像对失败")
            
            # 执行参数插值生成新图像
            interpolated_results = self.generate_interpolated_images(
                split_results, panorama_path1, panorama_path2, group_id
            )
            if not interpolated_results:
                raise Exception("参数插值生成图像失败")
            
            # 对插值后的图像进行超分处理
            interpolated_sr_results = self.super_resolve_interpolated_pairs(interpolated_results)
            if not interpolated_sr_results:
                raise Exception("插值图像超分处理失败")
            
            # 6. 创建最终数据（按yaw区间组织）
            final_data = self.create_final_data_with_interpolation(
                split_results,
                interpolated_sr_results,
                sr_results,
                group_id,
                panorama_path1,
                panorama_path2,
                desc_result  # 传入描述信息
            )
            
            processing_time = time.time() - start_time
            
            result = {
                "group_id": group_id,
                "panorama1": panorama_path1,
                "panorama2": panorama_path2,
                "preprocess_results": all_preprocess_results,
                "super_resolution_results": sr_results,
                "quality_results": quality_results,
                "filtered_results": filtered_results,
                "descriptions": desc_result,  # 添加描述信息
                "split_results": split_results,
                "interpolated_results": interpolated_results,
                "interpolated_sr_results": interpolated_sr_results,
                "final_data": final_data,
                "processing_time": processing_time,
                "success": True
            }
            
            logger.info(f"图片组 #{group_id} 处理完成 (耗时: {processing_time:.2f}秒)")
            return result
            
        except Exception as e:
            logger.error(f"处理图片组 #{group_id} 失败: {e}")
            return {
                "group_id": group_id,
                "panorama1": panorama_path1,
                "panorama2": panorama_path2,
                "error": str(e),
                "processing_time": time.time() - start_time,
                "success": False
            }
    
    def preprocess_panoramas_for_group(self, panorama_path: str, group_id: int, is_first: bool) -> List[Dict]:
        """为图片组预处理单张全景图"""
        logger.info(f"预处理 {'第一张' if is_first else '第二张'}全景图: {os.path.basename(panorama_path)}")
        
        # 转换为绝对路径
        if not os.path.isabs(panorama_path):
            panorama_path = os.path.abspath(panorama_path)
        
        try:
            # 调用预处理API，传入组ID信息
            response = self.api_client.call_api(
                port=self.API_PORTS['preprocess'],
                endpoint='preprocess_for_group',
                data={
                    'image_path': panorama_path,
                    'group_id': group_id,
                    'is_first': is_first
                }
            )
            
            if response.get('success'):
                results = response.get('results', [])
                logger.info(f"预处理完成，生成了 {len(results)} 个结果")
                return results
            else:
                logger.error(f"预处理失败: {response.get('error')}")
                return []
                
        except Exception as e:
            logger.error(f"预处理API调用失败: {e}")
            return []
    
    def filter_high_quality_results_by_group(self, quality_results: List[Dict], threshold: float = 0.7) -> List[Dict]:
        """按组过滤高质量结果（在同一yaw区间内取最低分）"""
        # 先按interval分组
        interval_dict = {}
        for result in quality_results:
            interval = result.get('interval', 0)
            if interval not in interval_dict:
                interval_dict[interval] = []
            interval_dict[interval].append(result)
        
        # 对每个interval内的pair取最低分
        filtered_results = []
        for interval, results in interval_dict.items():
            # 提取所有分数并转换为float
            scores = []
            for r in results:
                score = r.get('final_score')
                if score is not None:
                    try:
                        scores.append(float(score))
                    except (ValueError, TypeError):
                        logger.warning(f"无法转换分数为float: {score}")
            
            if scores:
                min_score = min(scores)
                
                if min_score >= threshold:
                    logger.info(f"yaw区间 {interval} 通过质量筛选 (最低分: {min_score:.3f})")
                    filtered_results.extend(results)
                else:
                    logger.info(f"yaw区间 {interval} 未通过质量筛选 (最低分: {min_score:.3f})")
            else:
                logger.warning(f"yaw区间 {interval} 没有有效的分数，跳过")
        
        logger.info(f"质量过滤完成，从 {len(quality_results)} 个结果中保留了 {len(filtered_results)} 个")
        return filtered_results
    
    def save_single_group_result(self, result: Dict):
        """保存单个图片组的处理结果（实时保存）"""
        try:
            # 构建完整的输出文件路径
            output_file_path = os.path.join(self.output_root_dir, self.output_file)
            
            # 读取现有的results
            existing_results = []
            if os.path.exists(output_file_path):
                with open(output_file_path, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
            
            # 添加本次的结果（如果是成功的话）
            if result.get('success', False) and 'final_data' in result:
                new_entries = result.get('final_data', [])
                existing_results.extend(new_entries)
                
                # 保存更新的results.json
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    json.dump(existing_results, f, ensure_ascii=False, indent=2)
                
                logger.info(f"实时更新 {self.output_file}，当前共有 {len(existing_results)} 个数据条目")
            
            # 生成或更新 group_info.json（无论成功失败都生成）
            group_id = result.get('group_id')
            panorama1 = result.get('panorama1', '')
            panorama2 = result.get('panorama2', '')
            processing_time = result.get('processing_time', 0)
            success = result.get('success', False)
            
            # 确定group目录位置
            group_dir_name = f"group_{group_id:04d}"
            if 'preprocess' in self.output_root_dir:
                parent_dir = os.path.dirname(self.output_root_dir)
                group_dir = os.path.join(parent_dir, group_dir_name)
            else:
                group_dir = os.path.join(self.output_root_dir, group_dir_name)
            
            # 构建group_info
            if success and 'final_data' in result:
                final_data = result.get('final_data', [])
                yaw_intervals = []
                for entry in final_data:
                    interval_info = entry.get('yaw_interval', {})
                    yaw_intervals.append({
                        'interval_id': interval_info.get('interval_id'),
                        'yaw_min': interval_info.get('yaw_min'),
                        'yaw_max': interval_info.get('yaw_max')
                    })
                
                group_info = {
                    'group_id': group_id,
                    'panorama1': os.path.basename(panorama1),
                    'panorama2': os.path.basename(panorama2),
                    'panorama1_path': panorama1,
                    'panorama2_path': panorama2,
                    'num_quadruples': len(final_data),
                    'yaw_intervals': yaw_intervals,
                    'processing_time': processing_time,
                    'success': True
                }
            else:
                # 失败或没有高质量数据的情况
                error_msg = result.get('error', '无数据或处理失败')
                group_info = {
                    'group_id': group_id,
                    'panorama1': os.path.basename(panorama1) if panorama1 else '',
                    'panorama2': os.path.basename(panorama2) if panorama2 else '',
                    'panorama1_path': panorama1 if panorama1 else '',
                    'panorama2_path': panorama2 if panorama2 else '',
                    'num_quadruples': 0,
                    'yaw_intervals': [],
                    'processing_time': processing_time,
                    'success': False,
                    'error': error_msg
                }
            
            # 保存 group_info.json
            group_info_file = os.path.join(group_dir, 'group_info.json')
            with open(group_info_file, 'w', encoding='utf-8') as f:
                json.dump(group_info, f, ensure_ascii=False, indent=2)
            logger.info(f"组信息已实时保存: {group_info_file}")
            
        except Exception as e:
            logger.error(f"保存单个组结果失败: {e}")
    
    def count_current_intervals(self) -> int:
        """统计当前 results.json 中的 yaw_interval 数量"""
        output_file_path = os.path.join(self.output_root_dir, self.output_file)
        
        if not os.path.exists(output_file_path):
            return 0
        
        try:
            with open(output_file_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            # results.json 是扁平化的 yaw_interval 列表
            # 每个条目就是一个 yaw_interval，直接返回列表长度
            if isinstance(results, list):
                return len(results)
            else:
                return 0
        except Exception as e:
            logger.warning(f"统计 yaw_interval 数量时出错: {e}")
            return 0
    
    def split_pairs_for_interpolation(self, filtered_results: List[Dict], group_id: int) -> List[Dict]:
        """切分图像对为左右两部分（用于插值）"""
        logger.info(f"开始切分 {len(filtered_results)} 个图像对")
        
        # 提取组目录
        split_results = []
        for result in filtered_results:
            try:
                pair_image_path = result.get('pair_image', '')
                if not pair_image_path or not os.path.exists(pair_image_path):
                    logger.warning(f"图像对不存在: {pair_image_path}")
                    continue
                
                # 从路径中提取组目录
                import re
                match = re.search(r'/group_\d+/', pair_image_path)
                if not match:
                    logger.error(f"无法从路径中提取组目录: {pair_image_path}")
                    continue
                
                # 提取匹配到的目录路径 (如 /group_0001/)
                group_dir_match = match.group(0)
                # 找到这个目录之前的部分
                group_root_dir = pair_image_path[:pair_image_path.find(group_dir_match) + len(group_dir_match) - 1]
                group_root_dir = os.path.abspath(group_root_dir)
                
                # 创建interpolated目录用于存放插值结果
                interpolated_dir = os.path.join(group_root_dir, "interpolated")
                os.makedirs(interpolated_dir, exist_ok=True)
                
                # 切分图像对
                img = Image.open(pair_image_path)
                width = img.width
                mid = width // 2
                left_img = img.crop((0, 0, mid, img.height))
                right_img = img.crop((mid, 0, width, img.height))
                
                # 保存切分后的图像
                base_name = os.path.splitext(os.path.basename(pair_image_path))[0]
                left_path = os.path.join(interpolated_dir, f"{base_name}_left.jpg")
                right_path = os.path.join(interpolated_dir, f"{base_name}_right.jpg")
                left_img.save(left_path)
                right_img.save(right_path)
                
                split_results.append({
                    'pair_image': pair_image_path,
                    'left_image': left_path,
                    'right_image': right_path,
                    'main_params': result.get('main_params'),
                    'rand_params': result.get('rand_params'),
                    'interval': result.get('interval'),
                    'yaw_interval': result.get('yaw_interval'),
                    'group_id': group_id
                })
                
            except Exception as e:
                logger.error(f"切分图像对失败: {e}")
                continue
        
        logger.info(f"切分完成，生成了 {len(split_results)} 个结果")
        return split_results
    
    def generate_interpolated_images(self, split_results: List[Dict], panorama1_path: str, panorama2_path: str, group_id: int) -> List[Dict]:
        """使用插值参数从全景图生成新图像 - 调用Preprocess API"""
        logger.info(f"开始生成插值图像，共有 {len(split_results)} 组数据")
        
        try:
            response = self.api_client.call_api(
                port=self.API_PORTS['preprocess'],
                endpoint='generate_interpolated_images',
                data={
                    'split_results': split_results,
                    'panorama1_path': panorama1_path,
                    'panorama2_path': panorama2_path,
                    'group_id': group_id
                }
            )
            
            if response.get('success'):
                results = response.get('results', [])
                logger.info(f"插值图像生成完成，生成了 {len(results)} 个结果")
                return results
            else:
                logger.error(f"插值图像生成失败: {response.get('error')}")
                return []
                
        except Exception as e:
            logger.error(f"插值图像生成API调用失败: {e}")
            return []
    
    def super_resolve_interpolated_pairs(self, interpolated_results: List[Dict]) -> List[Dict]:
        """对插值后的图像进行超分处理"""
        logger.info(f"开始对插值图像进行超分处理，共 {len(interpolated_results)} 组")
        
        # 先获取第一个插值图像的group目录
        if not interpolated_results:
            logger.warning("没有插值图像需要超分")
            return interpolated_results
        
        # 从第一个结果提取group信息
        first_interp = interpolated_results[0].get('interpolated_images', [])
        if not first_interp:
            logger.warning("插值图像列表为空")
            return interpolated_results
        
        first_path = first_interp[0].get('path', '')
        import re
        match = re.search(r'/group_\d+/', first_path)
        if not match:
            logger.error("无法从路径提取组目录")
            return interpolated_results
        
        # 提取匹配到的目录路径 (如 /group_0001/)
        group_dir_match = match.group(0)
        # 找到这个目录之前的部分
        group_root_dir = first_path[:first_path.find(group_dir_match) + len(group_dir_match) - 1]
        group_root_dir = os.path.abspath(group_root_dir)
        
        # 创建超分输出目录
        sr_output_dir = os.path.join(group_root_dir, "interpolated_sr")
        os.makedirs(sr_output_dir, exist_ok=True)
        
        # 收集所有插值图像路径
        all_interp_images = []
        
        for interp_group in interpolated_results:
            interp_images = interp_group.get('interpolated_images', [])
            for interp_img in interp_images:
                try:
                    input_path = interp_img.get('path')
                    if not input_path or not os.path.exists(input_path):
                        logger.warning(f"插值图像不存在: {input_path}")
                        continue
                    
                    all_interp_images.append((interp_img, input_path))
                    
                except Exception as e:
                    logger.error(f"收集插值图像失败: {interp_img.get('path', 'unknown')} - {e}")
                    continue
        
        logger.info(f"共收集了 {len(all_interp_images)} 张插值图像，开始批量超分处理")
        
        # 批量调用单张图像超分API
        if all_interp_images:
            try:
                # 为每个图像调用单张超分接口
                for interp_img, input_path in all_interp_images:
                    response = self.api_client.call_api(
                        port=self.API_PORTS['osediff'],
                        endpoint='super_resolution',
                        data={
                            'input_path': input_path,
                            'output_dir': group_root_dir,
                            'align_method': 'adain'
                        }
                    )
                    
                    if response.get('success'):
                        output_path = response.get('output_path')
                        # 将输出文件移动到interpolated_sr目录
                        import shutil
                        filename = os.path.basename(output_path)
                        target_path = os.path.join(sr_output_dir, filename)
                        shutil.move(output_path, target_path)
                        interp_img['super_resolved'] = target_path
                        logger.info(f"插值图像超分成功: {target_path}")
                    else:
                        logger.error(f"插值图像超分失败: {response.get('error')}")
                        
            except Exception as e:
                logger.error(f"批量超分处理失败: {e}")
        
        logger.info(f"插值图像超分处理完成")
        return interpolated_results
    
    def create_final_data_with_interpolation(self, split_results: List[Dict], interpolated_sr_results: List[Dict], 
                                            sr_results: List[Dict], group_id: int, panorama1_path: str, panorama2_path: str, 
                                            desc_result: Dict = None) -> List[Dict]:
        """创建包含插值图像的最终数据"""
        logger.info(f"创建包含插值的最终数据")
        
        # 构建描述字典，方便按 interval 查找
        descriptions_by_interval = {}
        if desc_result and desc_result.get('success') and desc_result.get('descriptions'):
            for desc in desc_result.get('descriptions', []):
                interval = desc.get('interval')
                if interval:
                    descriptions_by_interval[interval] = desc.get('description', '')
        
        final_data = []
        
        # 按interval组织数据
        interval_to_panoramas = {}  # {interval: {'panorama1': {...}, 'panorama2': {...}}}
        
        for split_result in split_results:
            interval = split_result.get('interval')
            if interval not in interval_to_panoramas:
                interval_to_panoramas[interval] = {}
            
            # 判断是panorama1还是panorama2
            pair_image = split_result.get('pair_image', '')
            pair_basename = os.path.splitext(os.path.basename(pair_image))[0]
            panorama1_basename = os.path.splitext(os.path.basename(panorama1_path))[0]
            panorama2_basename = os.path.splitext(os.path.basename(panorama2_path))[0]
            
            if pair_basename.startswith(panorama1_basename):
                interval_to_panoramas[interval]['panorama1'] = split_result
            elif pair_basename.startswith(panorama2_basename):
                interval_to_panoramas[interval]['panorama2'] = split_result
            else:
                # 如果无法判断，尝试从pair_image路径判断
                if panorama1_basename in pair_image:
                    interval_to_panoramas[interval]['panorama1'] = split_result
                else:
                    interval_to_panoramas[interval]['panorama2'] = split_result
        
        # 为每个interval创建最终数据条目
        for interval, panoramas in sorted(interval_to_panoramas.items()):
            if 'panorama1' not in panoramas or 'panorama2' not in panoramas:
                logger.warning(f"Interval {interval} 缺少完整的panorama数据，跳过")
                continue
            
            p1_split = panoramas['panorama1']
            p2_split = panoramas['panorama2']
            
            # 组织插值图像数据和参数
            p1_interp_data = []
            p2_interp_data = []
            p1_params_sequence = []
            p2_params_sequence = []
            
            for interp_group in interpolated_sr_results:
                panorama = interp_group.get('panorama')
                interp_images = interp_group.get('interpolated_images', [])
                
                if panorama == 'panorama1' and interp_group.get('interval') == interval:
                    # 按照从左到右的顺序组织：A1 (left) -> interp_01 -> ... -> interp_09 -> A2 (right)
                    images = [p1_split.get('left_image')]  # 起始左图
                    params = [p1_split.get('main_params')]  # 起始左图参数
                    
                    for interp_img in sorted(interp_images, key=lambda x: x.get('weight_idx', 0)):
                        images.append(interp_img.get('super_resolved', interp_img.get('path')))
                        params.append(interp_img.get('params'))  # 添加插值参数
                    
                    images.append(p1_split.get('right_image'))  # 结束右图
                    params.append(p1_split.get('rand_params'))  # 结束右图参数
                    
                    p1_interp_data = images
                    p1_params_sequence = params
                    
                elif panorama == 'panorama2' and interp_group.get('interval') == interval:
                    images = [p2_split.get('left_image')]
                    params = [p2_split.get('main_params')]
                    
                    for interp_img in sorted(interp_images, key=lambda x: x.get('weight_idx', 0)):
                        images.append(interp_img.get('super_resolved', interp_img.get('path')))
                        params.append(interp_img.get('params'))
                    
                    images.append(p2_split.get('right_image'))
                    params.append(p2_split.get('rand_params'))
                    
                    p2_interp_data = images
                    p2_params_sequence = params
            
            # 创建最终数据条目
            final_entry = {
                'group_id': group_id,
                'yaw_interval': {
                    'interval_id': interval,
                    'yaw_min': p1_split.get('yaw_interval', (0, 0))[0],
                    'yaw_max': p1_split.get('yaw_interval', (0, 0))[1]
                },
                'panorama1': {
                    'original_path': panorama1_path,
                    'interpolated_sequence': p1_interp_data,  # 11张图片序列
                    'params_sequence': p1_params_sequence     # 11组参数序列
                },
                'panorama2': {
                    'original_path': panorama2_path,
                    'interpolated_sequence': p2_interp_data,  # 11张图片序列
                    'params_sequence': p2_params_sequence,    # 11组参数序列
                    'right_image_description': descriptions_by_interval.get(interval, '')  # 添加右图描述
                }
            }
            
            final_data.append(final_entry)
            logger.info(f"创建interval {interval} 的最终数据条目，包含 {len(p1_interp_data)} + {len(p2_interp_data)} 张图像")
        
        logger.info(f"最终数据创建完成，共 {len(final_data)} 个条目")
        return final_data 