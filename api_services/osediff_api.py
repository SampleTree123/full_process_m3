import os
import sys
import logging
import argparse
from flask import Flask, request, jsonify
from PIL import Image
import torch
from torchvision import transforms

# 配置路径（必须在 from config 之前）
current_dir = os.path.dirname(os.path.abspath(__file__))
full_process_dir = os.path.dirname(current_dir)
sys.path.insert(0, full_process_dir)

from config.api_config import get_path

osediff_path = get_path('osediff_repo')
ram_path = os.path.join(osediff_path, 'ram')

# 配置日志：仅输出到 stdout，由 start_services 重定向到 logs/start/osediff_log.txt，避免在项目根目录重复写 osediff_log.txt
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger(__name__)

# 添加模块路径并切换工作目录
sys.path.append(osediff_path)
sys.path.append(ram_path)
original_cwd = os.getcwd()
os.chdir(osediff_path)

# 导入OSEDiff模块
try:
    from osediff import OSEDiff_test
    from my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
    from ram.models.ram_lora import ram
    from ram import inference_ram as inference
    logger.info("✅ OSEDiff模块导入成功")
except ImportError as e:
    logging.error(f"无法导入OSEDiff模块: {e}")
    sys.exit(1)

app = Flask(__name__)

# 全局变量
model = None
ram_model = None
weight_dtype = torch.float16

# 转换器
tensor_transforms = transforms.Compose([
    transforms.ToTensor(),
])

ram_transforms = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class OSEDiffAPI:
    """OSEDiff超分辨率API服务 - 基于test_osediff.py"""
    
    def __init__(self, model_path, pretrained_model_name_or_path='stabilityai/stable-diffusion-2-1-base', ram_path=None, ram_ft_path=None, device='cuda', upscale=2, process_size=512, mixed_precision='fp16', output_dir=None):
        self.model_path = model_path
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.ram_path = ram_path
        self.ram_ft_path = ram_ft_path
        self.device = device
        self.upscale = upscale
        self.process_size = process_size
        self.mixed_precision = mixed_precision
        self.weight_dtype = torch.float16 if mixed_precision == 'fp16' else torch.float32
        # 统一规范输出目录为绝对路径（避免因 chdir 导致相对路径锚定到 OSEDiff 仓库）
        def _to_abs_output_dir(path: str) -> str:
            if not path:
                return None
            if os.path.isabs(path):
                return path
            return os.path.abspath(os.path.join(original_cwd, path))

        self.output_dir = _to_abs_output_dir(output_dir)
        logger.info(f"[OSEDiffAPI] 输出根目录设置为: {self.output_dir}")
        
        # 初始化模型
        self._init_models()
    
    def _init_models(self):
        """初始化OSEDiff和RAM模型 - 基于test_osediff.py"""
        try:
            # 创建参数对象 - 直接按照test_osediff.py的方式
            class Args:
                def __init__(self, model_path, pretrained_model_name_or_path, ram_path, ram_ft_path, process_size, upscale, mixed_precision):
                    self.pretrained_model_name_or_path = pretrained_model_name_or_path
                    self.seed = 42
                    self.process_size = process_size
                    self.upscale = upscale
                    self.align_method = 'adain'
                    self.osediff_path = model_path
                    self.prompt = ''
                    self.ram_path = ram_path
                    self.ram_ft_path = ram_ft_path
                    self.save_prompts = True
                    self.mixed_precision = mixed_precision
                    self.merge_and_unload_lora = False
                    self.vae_decoder_tiled_size = 224
                    self.vae_encoder_tiled_size = 1024
                    self.latent_tiled_size = 96
                    self.latent_tiled_overlap = 32
                    # 新增：支持本地模型路径
                    self.local_files_only = True
                    self.trust_remote_code = True
            
            args = Args(self.model_path, self.pretrained_model_name_or_path, self.ram_path, self.ram_ft_path, self.process_size, self.upscale, self.mixed_precision)
            
            # 初始化OSEDiff模型
            global model
            model = OSEDiff_test(args)
            
            # 初始化RAM模型
            global ram_model
            ram_model = ram(
                pretrained=self.ram_path,
                pretrained_condition=self.ram_ft_path,
                image_size=384,
                vit='swin_l'
            )
            ram_model.eval()
            ram_model.to(self.device)
            
            # 设置权重类型
            global weight_dtype
            weight_dtype = torch.float32
            if self.mixed_precision == "fp16":
                weight_dtype = torch.float16
            
            # 设置权重类型
            ram_model = ram_model.to(dtype=weight_dtype)
            
            logger.info("✅ OSEDiff和RAM模型初始化成功")
            
        except Exception as e:
            logger.error(f"❌ 模型初始化失败: {str(e)}")
            raise
    
    def process_single_image(self, input_image_path, output_dir, align_method='adain'):
        """处理单张图像 - 直接按照test_osediff.py的方式"""
        logger.info(f"开始处理图像: {input_image_path}")
        
        input_image = Image.open(input_image_path).convert('RGB')
        
        # 确保输入图像是8的倍数
        ori_width, ori_height = input_image.size
        rscale = self.upscale
        resize_flag = False
        if ori_width < self.process_size//rscale or ori_height < self.process_size//rscale:
            scale = (self.process_size//rscale)/min(ori_width, ori_height)
            input_image = input_image.resize((int(scale*ori_width), int(scale*ori_height)))
            resize_flag = True
        input_image = input_image.resize((input_image.size[0]*rscale, input_image.size[1]*rscale))

        new_width = input_image.width - input_image.width % 8
        new_height = input_image.height - input_image.height % 8
        input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
        
        # 获取提示词并处理图像
        validation_prompt, lq = self.get_validation_prompt(input_image)
        logger.info(f"处理 {input_image_path}, 标签: {validation_prompt}")
        
        with torch.no_grad():
            lq = lq*2-1
            output_image = model(lq, prompt=validation_prompt)
            output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
            if align_method == 'adain':
                output_pil = adain_color_fix(target=output_pil, source=input_image)
            elif align_method == 'wavelet':
                output_pil = wavelet_color_fix(target=output_pil, source=input_image)
            if resize_flag:
                output_pil = output_pil.resize((int(self.upscale*ori_width), int(self.upscale*ori_height)))
        
        # 保存结果
        if not os.path.isabs(output_dir):
            output_dir = os.path.abspath(os.path.join(original_cwd, output_dir))
        osediff_output_dir = os.path.join(output_dir, "osediff")
        os.makedirs(osediff_output_dir, exist_ok=True)
        
        output_path = os.path.join(osediff_output_dir, os.path.basename(input_image_path))
        output_pil.save(output_path)
        logger.info(f"超分辨率处理完成: {output_path}")
        return output_path
    
    def get_validation_prompt(self, image, prompt=''):
        """获取验证提示词"""
        lq = tensor_transforms(image).unsqueeze(0).to(self.device)
        lq_ram = ram_transforms(lq).to(dtype=weight_dtype)
        captions = inference(lq_ram, ram_model)
        return f"{captions[0]}, {prompt},", lq

# 全局API实例
osediff_api = None

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({"status": "healthy", "service": "osediff"})

@app.route('/super_resolution_batch', methods=['POST'])
def super_resolution_batch_endpoint():
    """批量图像超分辨率 - 一次请求处理多张图像
    
    返回结果只包含必要字段：success, output_path (error 仅在失败时)
    """
    try:
        data = request.get_json()
        input_paths = data.get('input_paths', [])
        output_dir = data.get('output_dir', 'output/osediff')
        # 规范输出目录为绝对路径
        if output_dir and not os.path.isabs(output_dir):
            output_dir = os.path.abspath(os.path.join(original_cwd, output_dir))
        align_method = data.get('align_method', 'adain')
        
        if not input_paths:
            return jsonify({"error": "缺少input_paths参数"}), 400
        
        logger.info(f"批量超分处理: {len(input_paths)} 张图像")
        results = []
        for input_path in input_paths:
            try:
                output_path = osediff_api.process_single_image(input_path, output_dir, align_method)
                # 只返回必要字段，避免冗余传输
                results.append({"success": True, "output_path": output_path})
            except Exception as e:
                logger.error(f"批量超分中单张处理失败: {input_path} - {e}")
                results.append({"success": False, "output_path": None, "error": str(e)})
        
        success_count = sum(1 for r in results if r['success'])
        logger.info(f"批量超分完成: {success_count}/{len(input_paths)} 成功")
        return jsonify({"success": True, "results": results})
        
    except Exception as e:
        logger.error(f"批量超分辨率处理失败: {e}")
        return jsonify({"error": str(e)}), 500

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='OSEDiff超分辨率API服务')
    parser.add_argument('--port', type=int, default=5001, help='服务端口')
    parser.add_argument('--output_dir', type=str, default='output/osediff', help='输出目录')
    parser.add_argument('--osediff_path', type=str, required=True, help='OSEDiff模型路径')
    parser.add_argument('--pretrained_model_name_or_path', type=str, default='stabilityai/stable-diffusion-2-1-base', help='预训练模型路径')
    parser.add_argument('--ram_path', type=str, required=True, help='RAM模型路径')
    parser.add_argument('--ram_ft_path', type=str, default=None, help='RAM微调模型路径')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--upscale', type=int, default=2, help='放大倍数')
    parser.add_argument('--process_size', type=int, default=512, help='处理尺寸')
    parser.add_argument('--mixed_precision', type=str, default='fp16', help='混合精度')
    args = parser.parse_args()
    
    # 初始化API
    global osediff_api
    osediff_api = OSEDiffAPI(
        model_path=args.osediff_path,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        ram_path=args.ram_path,
        ram_ft_path=args.ram_ft_path,
        device=args.device,
        upscale=args.upscale,
        process_size=args.process_size,
        mixed_precision=args.mixed_precision,
        output_dir=os.path.abspath(os.path.join(original_cwd, args.output_dir)) if args.output_dir and not os.path.isabs(args.output_dir) else args.output_dir
    )
    
    logger.info(f"OSEDiff超分辨率API服务启动 - 端口: {args.port}")
    app.run(host='0.0.0.0', port=args.port, debug=False)

if __name__ == '__main__':
    main() 