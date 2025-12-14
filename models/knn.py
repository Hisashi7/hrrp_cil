import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
from sklearn.neighbors import KNeighborsClassifier
import time
import copy
import os
import pickle


class KNNParameter(nn.Parameter):
    """Custom Parameter class for KNN parameters"""
    def __new__(cls, data, requires_grad=True):
        return super(KNNParameter, cls).__new__(cls, data, requires_grad)


class KNNWrapper(nn.Module):
    """Wrapper class to make KNN compatible with PyTorch parameter counting"""
    def __init__(self):
        super().__init__()
        self.knn = None
        self._parameters_tensor = None
        
    def update_parameters(self, knn_model):
        """Convert KNN parameters to PyTorch parameters"""
        self.knn = knn_model
        if knn_model is not None:
            # Get training samples and flatten
            samples = knn_model._fit_X.flatten()
            # Get training labels
            labels = knn_model._y.flatten()
            # Concatenate all parameters
            all_params = np.concatenate([samples, labels])
            # Convert to tensor
            self._parameters_tensor = KNNParameter(torch.from_numpy(all_params).float())
            # Register parameter
            self.register_parameter('knn_params', self._parameters_tensor)


class KNN(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.knn = None
        self.knn_models = []
        self._network = KNNWrapper()
        self.topk = args.get('topk', 5)
        
    def process_data(self, loader):
        """直接处理原始数据，不进行特征提取"""
        data, labels = [], []
        for _, inputs, targets in loader:
            # 将图像数据展平为一维向量
            inputs = inputs.view(inputs.size(0), -1)
            data.append(inputs.numpy())
            labels.append(targets.numpy())
        return np.concatenate(data), np.concatenate(labels)

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        # 准备训练数据
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.args["batch_size"],
            shuffle=True,
            num_workers=self.args["num_workers"]
        )
        
        # 准备测试数据
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes),
            source="test",
            mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.args["batch_size"],
            shuffle=False,
            num_workers=self.args["num_workers"]
        )

        self._train(self.train_loader, self.test_loader)
        save_path = f"saved_pth/knn/model_{self._cur_task}.pkl"
        self.save_model(save_path)
        self._network.update_parameters(self.knn)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)

    def _train(self, train_loader, test_loader):
        # 直接处理训练数据
        logging.info("Processing training data...")
        train_data, train_labels = self.process_data(train_loader)
        
        # 训练KNN
        logging.info("Training KNN...")
        start_time = time.time()
        self.knn = KNeighborsClassifier(n_neighbors=self._total_classes)
        self.knn.fit(train_data, train_labels)
        self.knn_models.append(self.knn)
        
        training_time = time.time() - start_time
        logging.info(f"KNN training completed in {training_time:.2f} seconds")

        # 评估性能
        if test_loader is not None:
            test_acc = self._compute_accuracy(None, test_loader)
            logging.info(f"Test accuracy: {test_acc:.2f}%")

    def _compute_accuracy(self, model, loader):
        data, labels = self.process_data(loader)
        if self.knn is not None:
            predictions = self.knn.predict(data)
            correct = np.sum(predictions == labels)
            total = len(labels)
            return np.around(correct * 100 / total, decimals=2)
        return 0.0

    def _eval_cnn(self, loader):
        data, targets = self.process_data(loader)
        if self.knn is not None:
            # 获取概率预测
            proba = self.knn.predict_proba(data)
            # 转换为torch tensor以使用topk
            outputs = torch.from_numpy(proba).float()
            # 确保topk不超过类别数量
            k = min(self.topk, outputs.size(1))
            predicts = torch.topk(outputs, k=k, dim=1, largest=True, sorted=True)[1]
            
            if k < self.topk:
                padding = torch.zeros((predicts.size(0), self.topk - k), 
                                   dtype=predicts.dtype)
                predicts = torch.cat([predicts, padding], dim=1)
            
            return predicts.numpy(), targets

        return np.zeros((len(targets), self.topk)), targets

    def after_task(self):
        self._known_classes = self._total_classes

    def build_rehearsal_memory(self, data_manager, per_class):
        """Build rehearsal memory by selecting samples based on neighbor distances"""
        logging.info(f"Building rehearsal memory for KNN... ({per_class} samples per class)")
        
        if self._fixed_memory:
            self._construct_exemplar_unified(data_manager, per_class)
        else:
            self._reduce_exemplar(data_manager, per_class)
            self._construct_exemplar(data_manager, per_class)

    def _reduce_exemplar(self, data_manager, m):
        """Reduce exemplar set for each class"""
        logging.info(f"Reducing exemplars... ({m} per class)")
        dummy_data, dummy_targets = copy.deepcopy(self._data_memory), copy.deepcopy(self._targets_memory)
        self._data_memory, self._targets_memory = np.array([]), np.array([])

        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            samples = dummy_data[mask]
            if len(samples) > 0:
                # 计算样本到其k个最近邻的平均距离
                distances, _ = self.knn.kneighbors(samples)
                mean_distances = distances.mean(axis=1)
                # 选择距离适中的样本（既不太近也不太远）
                indices = np.argsort(np.abs(mean_distances - np.median(mean_distances)))[:m]
                selected_data = samples[indices]
                selected_targets = np.full(min(m, len(indices)), class_idx)
                
                self._data_memory = np.concatenate((self._data_memory, selected_data)) if len(self._data_memory) != 0 else selected_data
                self._targets_memory = np.concatenate((self._targets_memory, selected_targets)) if len(self._targets_memory) != 0 else selected_targets

    def _construct_exemplar(self, data_manager, m):
        """Construct exemplar set for new classes"""
        logging.info(f"Constructing exemplars... ({m} per class)")
        
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, _ = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True
            )
            
            if len(data) > 0:
                # 计算样本到其k个最近邻的平均距离
                distances, _ = self.knn.kneighbors(data)
                mean_distances = distances.mean(axis=1)
                # 选择具有代表性的样本
                indices = np.argsort(np.abs(mean_distances - np.median(mean_distances)))[:m]
                selected_data = data[indices]
                selected_targets = np.full(min(m, len(indices)), class_idx)
                
                self._data_memory = np.concatenate((self._data_memory, selected_data)) if len(self._data_memory) != 0 else selected_data
                self._targets_memory = np.concatenate((self._targets_memory, selected_targets)) if len(self._targets_memory) != 0 else selected_targets

    def _construct_exemplar_unified(self, data_manager, m):
        """Unified version of exemplar construction"""
        self._reduce_exemplar(data_manager, m)
        self._construct_exemplar(data_manager, m)
    
    def save_model(self, save_path):
        """Save the trained KNN model
        Args:
            save_path: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save model using pickle
        with open(save_path, 'wb') as f:
            pickle.dump(self.knn, f)
        
        logging.info(f"Model saved to {save_path}")