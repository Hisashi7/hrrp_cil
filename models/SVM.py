import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
from sklearn.svm import SVC
import time
import copy
import os
import pickle


class SVMParameter(nn.Parameter):
    """Custom Parameter class for SVM parameters"""
    def __new__(cls, data, requires_grad=True):
        return super(SVMParameter, cls).__new__(cls, data, requires_grad)

class SVMWrapper(nn.Module):
    """Wrapper class to make SVM compatible with PyTorch parameter counting"""
    def __init__(self):
        super().__init__()
        self.svm = None
        self._parameters_tensor = None
        
    def update_parameters(self, svm_model):
        """Convert SVM parameters to PyTorch parameters"""
        self.svm = svm_model
        if svm_model is not None:
            # Get support vectors and flatten
            sv = svm_model.support_vectors_.flatten()
            # Get dual coefficients
            dual_coef = svm_model.dual_coef_.flatten()
            # Get intercept
            intercept = svm_model.intercept_.flatten()
            # Concatenate all parameters
            all_params = np.concatenate([sv, dual_coef, intercept])
            # Convert to tensor
            self._parameters_tensor = SVMParameter(torch.from_numpy(all_params).float())
            # Register parameter
            self.register_parameter('svm_params', self._parameters_tensor)

class SVM(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.svm = None
        self.svm_models = []
        self._network = SVMWrapper()
        
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
        save_path = f"saved_pth/svm/model_{self._cur_task}.pkl"
        self.save_model(save_path)
        self._network.update_parameters(self.svm)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)

    def _train(self, train_loader, test_loader):
        # 直接处理训练数据
        logging.info("Processing training data...")
        train_data, train_labels = self.process_data(train_loader)
        
        # 训练SVM
        logging.info("Training SVM...")
        start_time = time.time()
        self.svm = SVC(kernel='rbf', C=1.0, probability=True)
        self.svm.fit(train_data, train_labels)
        self.svm_models.append(self.svm)
        
        training_time = time.time() - start_time
        logging.info(f"SVM training completed in {training_time:.2f} seconds")

        # 评估性能
        if test_loader is not None:
            test_acc = self._compute_accuracy(None, test_loader)
            logging.info(f"Test accuracy: {test_acc:.2f}%")

    def _compute_accuracy(self, model, loader):
        data, labels = self.process_data(loader)
        if self.svm is not None:
            predictions = self.svm.predict(data)
            correct = np.sum(predictions == labels)
            total = len(labels)
            return np.around(correct * 100 / total, decimals=2)
        return 0.0

    def _eval_cnn(self, loader):
        data, targets = self.process_data(loader)
        if self.svm is not None:
            # 获取概率预测
            proba = self.svm.predict_proba(data)
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
        """Build rehearsal memory by selecting samples near the decision boundary"""
        logging.info(f"Building rehearsal memory for SVM... ({per_class} samples per class)")
        
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
            # Get decision values for samples of this class
            samples = dummy_data[mask]
            if len(samples) > 0:
                decision_values = np.abs(self.svm.decision_function(samples))
                # Sort by proximity to decision boundary (smaller absolute decision value)
                indices = np.argsort(decision_values[:, class_idx])[:m]
                selected_data = samples[indices]
                selected_targets = np.full(min(m, len(indices)), class_idx)
                
                # Store selected samples
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
                # Get decision values for samples of this class
                decision_values = np.abs(self.svm.decision_function(data))
                # Select samples closest to decision boundary
                indices = np.argsort(decision_values[:, class_idx])[:m]
                selected_data = data[indices]
                selected_targets = np.full(min(m, len(indices)), class_idx)
                
                # Store selected samples
                self._data_memory = np.concatenate((self._data_memory, selected_data)) if len(self._data_memory) != 0 else selected_data
                self._targets_memory = np.concatenate((self._targets_memory, selected_targets)) if len(self._targets_memory) != 0 else selected_targets

    def _construct_exemplar_unified(self, data_manager, m):
        """Unified version of exemplar construction"""
        logging.info(f"Constructing unified exemplars... ({m} per class)")
        
        # Handle old classes
        for class_idx in range(self._known_classes):
            self._reduce_exemplar(data_manager, m)
        
        # Handle new classes
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, _ = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True
            )
            
            if len(data) > 0:
                decision_values = np.abs(self.svm.decision_function(data))
                indices = np.argsort(decision_values)[:m]
                selected_data = data[indices]
                selected_targets = np.full(min(m, len(indices)), class_idx)
                
                self._data_memory = np.concatenate((self._data_memory, selected_data)) if len(self._data_memory) != 0 else selected_data
                self._targets_memory = np.concatenate((self._targets_memory, selected_targets)) if len(self._targets_memory) != 0 else selected_targets

    def save_model(self, save_path):
        """Save the trained SVM model
        Args:
            save_path: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save model using pickle
        with open(save_path, 'wb') as f:
            pickle.dump(self.svm, f)
        
        logging.info(f"Model saved to {save_path}")