import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
from utils.inc_net import IncrementalNet
from sklearn.svm import SVC
import time

class SVMNet(nn.Module):
    """包装特征提取器和SVM的网络类"""
    def __init__(self, args):
        super().__init__()
        self.convnet = IncrementalNet(args, False).convnet
        self.svm = None
        self.features = None
    
    def forward(self, x):
        features = self.convnet(x)
        # 确保返回的是张量而不是字典
        if isinstance(features, dict):
            features = features['features']
        features = features.view(features.size(0), -1)
        return {
            "features": features,
            "logits": features  # 为了保持与原框架一致，返回相同的值作为logits
        }

class SVM(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = SVMNet(args)
        self.svm_models = []
        
    def extract_features(self, loader):
        features, labels = [], []
        self._network.eval()
        with torch.no_grad():
            for _, inputs, targets in loader:
                inputs = inputs.to(self._device)
                outputs = self._network(inputs)
                features.append(outputs["features"].cpu().numpy())
                labels.append(targets.numpy())
        return np.concatenate(features), np.concatenate(labels)

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

    def _compute_accuracy(self, loader):
        """Compute accuracy with the current SVM model"""
        self._network.eval()
        features, labels = self.extract_features(loader)
        
        if self._network.svm is not None:
            predictions = self._network.svm.predict(features)
            correct = (predictions == labels).sum()
            total = len(labels)
            return np.around(correct * 100 / total, decimals=2)
        return 0.0
    
    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        
        # 提取特征
        logging.info("Extracting features for training...")
        train_features, train_labels = self.extract_features(train_loader)
        
        # 训练SVM
        logging.info("Training SVM...")
        start_time = time.time()
        svm = SVC(kernel='rbf', C=1.0, probability=True)
        svm.fit(train_features, train_labels)
        self.svm_models.append(svm)
        self._network.svm = svm  # 保存当前SVM模型到网络中
        
        training_time = time.time() - start_time
        logging.info(f"SVM training completed in {training_time:.2f} seconds")

        # 评估性能
        if test_loader is not None:
            test_acc = self._compute_accuracy(test_loader)
            logging.info(f"Test accuracy: {test_acc:.2f}%")

    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)

        return cnn_accy, None

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, inputs, targets in loader:
            inputs = inputs.to(self._device)
            with torch.no_grad():
                features = self._network(inputs)["features"]
                # 获取SVM的概率预测
                proba = self._network.svm.predict_proba(features.cpu().numpy())
                # 转换为torch tensor以使用topk
                outputs = torch.from_numpy(proba).float().to(self._device)
                predicts = torch.topk(
                    outputs, k=self.topk, dim=1, largest=True, sorted=True
                )[1]  # [bs, topk]
                y_pred.append(predicts.cpu().numpy())
                y_true.append(targets.numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]


    def after_task(self):
        self._known_classes = self._total_classes