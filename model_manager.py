"""
Shared Model Manager
Centralized model loading, caching, and management to eliminate code duplication
Supports lazy loading, caching, and efficient resource management
"""
import os
import sys
import torch
from typing import Optional, Dict, Any
from functools import lru_cache

# Add recognize-anything to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'recognize-anything'))

import config

class ModelManager:
    """Singleton model manager for shared model instances"""
    _instance = None
    _models = {}
    _device = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._device = config.DEVICE
        print(f"🔧 ModelManager initialized on {self._device.upper()}")
    
    @property
    def device(self):
        """Get current device"""
        return self._device
    
    def get_yolo(self, force_reload: bool = False):
        """Get or load YOLO model (cached)"""
        if 'yolo' not in self._models or force_reload:
            print("👁️ Loading YOLO model...")
            from ultralytics import YOLO
            self._models['yolo'] = YOLO(config.YOLO_MODEL_PATH)
            print("✅ YOLO model loaded and cached")
        return self._models['yolo']
    
    def get_ram_model(self, force_reload: bool = False):
        """Get or load RAM model (cached)"""
        if 'ram_model' not in self._models or force_reload:
            print("👁️ Loading RAM model...")
            from ram.models import ram
            from ram import get_transform
            
            model = ram(
                pretrained=config.RAM_CHECKPOINT_PATH,
                vit=config.RAM_VIT_TYPE,
                image_size=config.RAM_IMAGE_SIZE
            )
            model = model.to(self._device)
            model.eval()
            self._models['ram_model'] = model
            self._models['ram_transform'] = get_transform(image_size=config.RAM_IMAGE_SIZE)
            print("✅ RAM model loaded and cached")
        return self._models['ram_model'], self._models['ram_transform']
    
    def get_blip(self, force_reload: bool = False):
        """Get or load BLIP model (cached)"""
        if 'blip_processor' not in self._models or force_reload:
            print("🧠 Loading BLIP model...")
            from transformers import BlipProcessor, BlipForConditionalGeneration
            
            processor = BlipProcessor.from_pretrained(config.BLIP_MODEL_ID)
            model = BlipForConditionalGeneration.from_pretrained(
                config.BLIP_MODEL_ID
            ).to(self._device)
            model.eval()
            
            self._models['blip_processor'] = processor
            self._models['blip_model'] = model
            print("✅ BLIP model loaded and cached")
        return self._models['blip_processor'], self._models['blip_model']
    
    def get_clip(self, force_reload: bool = False):
        """Get or load CLIP model (cached)"""
        if 'clip_processor' not in self._models or force_reload:
            print("🔍 Loading CLIP model...")
            from transformers import CLIPProcessor, CLIPModel
            
            processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL_ID)
            model = CLIPModel.from_pretrained(config.CLIP_MODEL_ID).to(self._device)
            model.eval()
            
            self._models['clip_processor'] = processor
            self._models['clip_model'] = model
            print("✅ CLIP model loaded and cached")
        return self._models['clip_processor'], self._models['clip_model']
    
    def preload_all(self):
        """Preload all models (useful for benchmarking and faster first run)"""
        print("🔄 Preloading all models...")
        self.get_yolo()
        self.get_ram_model()
        self.get_blip()
        self.get_clip()
        print("✅ All models preloaded and ready (cached for future use)")
    
    def get_all_models(self):
        """Get all models at once (returns tuple)"""
        return {
            "yolo": self.get_yolo(),
            "ram": self.get_ram_model(),
            "blip": self.get_blip(),
            "clip": self.get_clip()
        }
    
    def clear_cache(self):
        """Clear all cached models (free memory)"""
        print("🧹 Clearing model cache...")
        for model_name in list(self._models.keys()):
            if hasattr(self._models[model_name], 'to'):
                del self._models[model_name]
            del self._models[model_name]
        self._models.clear()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print("✅ Model cache cleared")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        info = {
            "device": self._device,
            "loaded_models": list(self._models.keys()),
            "cuda_available": torch.cuda.is_available(),
            "memory_allocated": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
            "memory_reserved": torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        }
        return info


# Global singleton instance
_model_manager = None

def get_model_manager() -> ModelManager:
    """Get global ModelManager instance"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
