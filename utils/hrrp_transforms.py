import numpy as np
import random
import torch
from torchvision import transforms

class HRRPTransforms:
    """HRRP数据增强策略，兼容torchvision transforms接口"""
    def __init__(self, args):
        self.ssl_training = args.get('ssl_training', False)  # 从args中获取ssl_training
        aug_config = args.get("augmentation", {})
        self.weak_shift = aug_config.get("weak_shift", 5)
        self.strong_shift = aug_config.get("strong_shift", 10)
        self.noise_level = aug_config.get("noise_level", 0.05)
        self.scale_range = aug_config.get("scale_range", [0.8, 1.2])
        self.enable_weak = aug_config.get("enable_weak", True)
        self.enable_strong = aug_config.get("enable_strong", True)
        
        # 修改为只包含一个自定义转换
        self.train_trsf = [self]  # 直接使用实例本身作为转换
        self.test_trsf = [
            transforms.Lambda(self.to_tensor_transform)
        ]
        self.common_trsf = []

    def to_tensor_transform(self, x):
        """Convert numpy array to tensor"""
        if isinstance(x, torch.Tensor):
            return x
        return torch.FloatTensor(x)

    def weak_augment(self, x):
        """弱增强: 小范围平移"""
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        if not self.enable_weak:
            return x
        shift = random.randint(-self.weak_shift, self.weak_shift)
        return np.roll(x, shift)

    def strong_augment(self, x):
        """强增强: 大范围平移 + 高斯噪声 + 随机振幅调制"""
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        if not self.enable_strong:
            return x
            
        # 1. 大范围平移
        shift = random.randint(-self.strong_shift, self.strong_shift)
        x_aug = np.roll(x, shift)
        
        # 2. 添加高斯噪声
        noise = np.random.normal(0, self.noise_level, x.shape)
        x_aug = x_aug + noise
        
        # 3. 随机振幅调制
        scale = random.uniform(*self.scale_range)
        x_aug = x_aug * scale
        
        return x_aug

    def __call__(self, x):
        """实现单一入口的转换逻辑"""
        if isinstance(x, torch.Tensor):
            x = x.numpy()
            
        if self.ssl_training:
            # SSL模式：返回弱增强和强增强
            weak_aug = self.weak_augment(x.copy())
            strong_aug = self.strong_augment(x.copy())
            return (self.to_tensor_transform(weak_aug), 
                   self.to_tensor_transform(strong_aug))
        else:
            # 普通模式：只返回弱增强
            return self.to_tensor_transform(self.weak_augment(x))