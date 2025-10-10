#!/usr/bin/env python3
"""
图像理解与处理 Demo
实现图像特征提取、物体识别、场景理解、OCR 文字识别等功能
"""

import os
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import base64
from PIL import Image
import io


@dataclass
class ImageAnalysisResult:
    """图像分析结果数据类"""
    objects: List[Dict[str, Any]]  # 检测到的物体
    scenes: List[str]  # 场景标签
    text_content: Optional[str]  # OCR 识别的文字
    features: Dict[str, Any]  # 图像特征
    metadata: Dict[str, Any]  # 元数据


@dataclass
class ObjectDetection:
    """物体检测结果"""
    label: str
    confidence: float
    bbox: List[float]  # [x, y, width, height]


class ImageUnderstandingProcessor:
    """图像理解处理器"""
    
    def __init__(self):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.webp']
        
    def load_image(self, image_path: str) -> Optional[Image.Image]:
        """加载图像文件"""
        try:
            if not os.path.exists(image_path):
                print(f"❌ 图像文件不存在: {image_path}")
                return None
                
            file_ext = Path(image_path).suffix.lower()
            if file_ext not in self.supported_formats:
                print(f"❌ 不支持的图像格式: {file_ext}")
                return None
                
            image = Image.open(image_path)
            print(f"✅ 成功加载图像: {image_path} ({image.size})")
            return image
            
        except Exception as e:
            print(f"❌ 加载图像失败: {e}")
            return None
    
    def extract_features(self, image: Image.Image) -> Dict[str, Any]:
        """提取图像基础特征"""
        features = {
            "dimensions": image.size,
            "mode": image.mode,
            "format": image.format,
            "aspect_ratio": image.width / image.height if image.height > 0 else 0
        }
        return features
    
    def detect_objects(self, image: Image.Image) -> List[ObjectDetection]:
        """物体检测（模拟实现）"""
        # 在实际应用中，这里会集成 YOLO、Detectron2 等目标检测模型
        objects = [
            ObjectDetection(label="person", confidence=0.95, bbox=[100, 150, 200, 300]),
            ObjectDetection(label="car", confidence=0.87, bbox=[300, 200, 150, 100]),
            ObjectDetection(label="tree", confidence=0.78, bbox=[50, 50, 100, 150])
        ]
        return objects
    
    def recognize_scenes(self, image: Image.Image) -> List[str]:
        """场景识别（模拟实现）"""
        # 在实际应用中，这里会集成场景分类模型
        scenes = ["outdoor", "urban", "daytime", "street"]
        return scenes
    
    def extract_text(self, image: Image.Image) -> Optional[str]:
        """OCR 文字识别（模拟实现）"""
        # 在实际应用中，这里会集成 Tesseract、EasyOCR 等 OCR 引擎
        text_content = "示例文字：欢迎使用多模态图像处理系统"
        return text_content
    
    def analyze_image(self, image_path: str) -> Optional[ImageAnalysisResult]:
        """综合分析图像"""
        print(f"🔍 开始分析图像: {image_path}")
        
        image = self.load_image(image_path)
        if not image:
            return None
        
        # 提取特征
        features = self.extract_features(image)
        print(f"📊 图像特征: {features}")
        
        # 物体检测
        objects = self.detect_objects(image)
        print(f"🎯 检测到 {len(objects)} 个物体")
        
        # 场景识别
        scenes = self.recognize_scenes(image)
        print(f"🏞️ 场景标签: {scenes}")
        
        # 文字识别
        text_content = self.extract_text(image)
        if text_content:
            print(f"📝 识别文字: {text_content}")
        
        # 构建结果
        result = ImageAnalysisResult(
            objects=[{"label": obj.label, "confidence": obj.confidence, "bbox": obj.bbox} 
                    for obj in objects],
            scenes=scenes,
            text_content=text_content,
            features=features,
            metadata={
                "processing_time": "0.5s",
                "model_version": "v1.0",
                "image_size": f"{image.width}x{image.height}"
            }
        )
        
        print("✅ 图像分析完成")
        return result
    
    def save_analysis_result(self, result: ImageAnalysisResult, output_path: str):
        """保存分析结果到 JSON 文件"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "objects": result.objects,
                    "scenes": result.scenes,
                    "text_content": result.text_content,
                    "features": result.features,
                    "metadata": result.metadata
                }, f, ensure_ascii=False, indent=2)
            print(f"💾 分析结果已保存到: {output_path}")
        except Exception as e:
            print(f"❌ 保存结果失败: {e}")


def demo_image_understanding():
    """演示图像理解功能"""
    print("🚀 开始图像理解演示")
    print("=" * 50)
    
    processor = ImageUnderstandingProcessor()
    
    # 创建测试图像（在实际应用中这里会使用真实图像文件）
    test_image_path = "test_image.jpg"
    
    # 创建一个简单的测试图像
    try:
        # 创建一个红色矩形图像
        test_image = Image.new('RGB', (640, 480), color='red')
        test_image.save(test_image_path)
        print(f"📸 创建测试图像: {test_image_path}")
    except Exception as e:
        print(f"❌ 创建测试图像失败: {e}")
        return
    
    # 分析图像
    result = processor.analyze_image(test_image_path)
    
    if result:
        print("\n📋 分析结果摘要:")
        print(f"   物体数量: {len(result.objects)}")
        print(f"   场景标签: {', '.join(result.scenes)}")
        print(f"   图像尺寸: {result.features['dimensions']}")
        print(f"   文字内容: {result.text_content}")
        
        # 保存结果
        output_path = "image_analysis_result.json"
        processor.save_analysis_result(result, output_path)
        
        # 清理测试文件
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
            print(f"🗑️ 清理测试文件: {test_image_path}")
    
    print("\n✅ 图像理解演示完成")


if __name__ == "__main__":
    demo_image_understanding()