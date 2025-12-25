import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from convs.resnet import resnet18
import seaborn as sns
from tqdm import tqdm

class HRRPDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        
    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
    def __len__(self):
        return len(self.data)

def mapping(labels):
    labels_set = np.unique(labels)
    map = np.full((1, max(labels_set)+1), -1)
    lab = 0
    for i in range(map.shape[1]):
        if i in labels_set:
            map[0][i] = lab
            lab += 1

    labels = np.array([map[0][labels[i]].item() for i in range(labels.shape[0])])
    return labels

def load_data():
    # 加载训练集和测试集
    train_data = sio.loadmat('data/hrrp/原理样机_2024.3.28/TRAIN_RESAMPLE.mat')
    test_data = sio.loadmat('data/hrrp/原理样机_2024.3.28/TEST.mat')
    
    X_train = train_data['data_resample'][:, 1:]  # 假设数据字段名为'data'
    y_train = mapping(np.array(train_data['data_resample'][:, 0], dtype=np.int64))  # 假设标签字段名为'label'
    X_test = test_data['TEST'][:, 1:]
    y_test = mapping(np.array(test_data['TEST'][:, 0], dtype=np.int64))

    return X_train, y_train, X_test, y_test

def augment_signal(signal):
    # 实现两种数据增强方式
    aug1 = signal + torch.randn_like(signal) * 0.1  # 加性高斯噪声
    aug2 = signal * (1 + torch.randn_like(signal) * 0.1)  # 乘性噪声
    return aug1, aug2

class ContrastiveLearningModel:
    def __init__(self, args, num_classes=3):  # 每组3个类别
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = resnet18(pretrained=False, args=args).to(self.device)
        self.classifier = nn.Linear(self.model.out_dim, num_classes).to(self.device)
        self.contrastive_criterion = nn.CosineSimilarity(dim=1)
        self.classification_criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(list(self.model.parameters()) + 
                                  list(self.classifier.parameters()), lr=0.001)
        self.softmax = nn.Softmax(dim=1)
        
    def train_step(self, batch):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        
        # 对每个样本进行两种增强
        aug1, aug2 = augment_signal(x)
        
        # 获取特征和预测
        features1 = self.model(aug1)['features']
        features2 = self.model(aug2)['features']
        
        # 分类预测
        logits1 = self.classifier(features1)
        logits2 = self.classifier(features2)
        
        # 计算对比损失
        contrastive_loss = -self.contrastive_criterion(features1, features2).mean()
        
        # 计算分类损失
        classification_loss = (self.classification_criterion(logits1, y) + 
                             self.classification_criterion(logits2, y)) / 2
        
        # 总损失 = 对比损失 + 分类损失
        loss = contrastive_loss + classification_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), contrastive_loss.item(), classification_loss.item()
    
    def get_features(self, x):
        self.model.eval()
        with torch.no_grad():
            aug1, aug2 = augment_signal(x.to(self.device))
            feat1 = self.model(aug1)['features']
            feat2 = self.model(aug2)['features']
        return feat1, feat2
    
    def predict(self, x, threshold=0.8):
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            features = self.model(x)['features']
            logits = self.classifier(features)
            probs = self.softmax(logits)
            max_probs, predictions = torch.max(probs, dim=1)
            # 将概率值低于阈值的样本标记为异常(-1)
            predictions[max_probs < threshold] = -1
            return predictions, max_probs

def evaluate_openset(model, loader, threshold=0.8, known_classes=None):
    if known_classes is None:
        known_classes = set([0, 1, 2])  # 第一组类别
    
    # 创建预测值到原始类别的映射
    pred_to_orig = {i: cls for i, cls in enumerate(sorted(known_classes))}
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    test_pbar = tqdm(loader, desc='Evaluating open-set')
    for x, y in test_pbar:
        preds, probs = model.predict(x, threshold)
        # 将非-1的预测值映射回原始类别
        preds_mapped = torch.tensor([pred_to_orig[p.item()] if p.item() != -1 else -1 for p in preds])
        all_preds.extend(preds_mapped.cpu().numpy())
        all_labels.extend(y.numpy())
        all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # 计算性能指标
    correct_known = 0
    total_known = 0
    correct_unknown = 0
    total_unknown = 0
    
    for pred, label in zip(all_preds, all_labels):
        if label in known_classes:
            total_known += 1
            if pred == label:
                correct_known += 1
        else:
            total_unknown += 1
            if pred == -1:  # 正确识别为异常样本
                correct_unknown += 1
    
    known_acc = correct_known / total_known if total_known > 0 else 0
    unknown_acc = correct_unknown / total_unknown if total_unknown > 0 else 0
    
    return {
        'known_acc': known_acc,
        'unknown_acc': unknown_acc,
        'avg_acc': (known_acc + unknown_acc) / 2,
        'predictions': all_preds,
        'probabilities': all_probs
    }

def train_and_evaluate_group(X_train, y_train, X_test, y_test, group_idx):
    # 准备当前组的数据
    mask_train = np.isin(y_train, group_idx)
    mask_test = np.isin(y_test, group_idx)
    
    # 将标签映射到0,1,2
    label_map = {old: new for new, old in enumerate(group_idx)}
    mapped_y_train = np.array([label_map[y] for y in y_train[mask_train]])
    
    train_dataset = HRRPDataset(X_train[mask_train], mapped_y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 初始化模型
    args = {"dataset": "cifar", "model_name": "resnet18"}
    model = ContrastiveLearningModel(args, num_classes=3)
    
    # 训练模型
    n_epochs = 200
    pbar = tqdm(range(n_epochs), desc='Training')
    for epoch in pbar:
        total_loss = total_contrastive = total_classification = 0
        batch_pbar = tqdm(train_loader, leave=False, desc=f'Epoch {epoch}')
        for batch in batch_pbar:
            loss, con_loss, cls_loss = model.train_step(batch)
            total_loss += loss
            total_contrastive += con_loss
            total_classification += cls_loss
            
            # 更新进度条描述
            batch_pbar.set_postfix({
                'loss': f'{loss:.4f}',
                'con_loss': f'{con_loss:.4f}',
                'cls_loss': f'{cls_loss:.4f}'
            })
            
        avg_loss = total_loss / len(train_loader)
        avg_con = total_contrastive / len(train_loader)
        avg_cls = total_classification / len(train_loader)
        
        # 更新主进度条描述
        pbar.set_postfix({
            'avg_loss': f'{avg_loss:.4f}',
            'avg_con': f'{avg_con:.4f}',
            'avg_cls': f'{avg_cls:.4f}'
        })
    
    # 在测试集上提取特征并计算二范数
    test_dataset = HRRPDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    all_norms = []
    all_labels = []
    
    test_pbar = tqdm(test_loader, desc='Extracting features')
    for x, y in test_pbar:
        feat1, feat2 = model.get_features(x)
        norms = torch.norm(feat1 - feat2, dim=1).cpu().numpy()
        all_norms.extend(norms)
        all_labels.extend(y.numpy())
    
    # 在测试集上评估开集识别性能
    print("\nEvaluating open-set recognition...")
    results = evaluate_openset(model, test_loader, threshold=0.8, known_classes=set(group_idx))
    print(f"Known Classes Accuracy: {results['known_acc']:.4f}")
    print(f"Unknown Classes Accuracy: {results['unknown_acc']:.4f}")
    print(f"Average Accuracy: {results['avg_acc']:.4f}")
    
    return np.array(all_norms), np.array(all_labels), results

def main():
    # 加载数据
    X_train, y_train, X_test, y_test = load_data()
    
    # 定义三组类别
    groups = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    
    # 为每组训练模型并收集结果
    all_results = []
    openset_results = []
    all_labels = y_test
    
    for group_idx in groups:
        print(f"\nTraining group {group_idx}")
        norms, labels, openset_result = train_and_evaluate_group(X_train, y_train, X_test, y_test, group_idx)
        all_results.append((norms, labels))
        openset_results.append(openset_result)
    
    # 可视化结果
    plt.figure(figsize=(15, 10))
    
    # 绘制特征范数分布
    for i, (norms, labels) in enumerate(all_results):
        plt.subplot(2, 3, i+1)
        for class_idx in range(9):
            class_norms = norms[labels == class_idx]
            if len(class_norms) > 0:
                sns.kdeplot(data=class_norms, label=f'Class {class_idx}')
        plt.title(f'Group {groups[i]}')
        plt.xlabel('L2 Norm')
        plt.ylabel('Density')
        plt.legend()
    
    # 绘制概率分布
    for i, result in enumerate(openset_results):
        plt.subplot(2, 3, i+4)
        known_mask = np.isin(all_labels, groups[i])
        known_probs = result['probabilities'][known_mask]
        unknown_probs = result['probabilities'][~known_mask]
        
        if len(known_probs) > 0:
            sns.kdeplot(data=known_probs, label='Known', color='blue')
        if len(unknown_probs) > 0:
            sns.kdeplot(data=unknown_probs, label='Unknown', color='red')
            
        plt.axvline(x=0.8, color='green', linestyle='--', label='Threshold')
        plt.title(f'Group {groups[i]} Probability Distribution')
        plt.xlabel('Maximum Softmax Probability')
        plt.ylabel('Density')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('openset_recognition_results.png')
    plt.close()

if __name__ == "__main__":
    main()
