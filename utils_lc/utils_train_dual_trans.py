
"""
训练 dual_trans 模型的代码
 (Dual Transformer Based Prediction for Lane Change Intentions and Trajectories in Mixed Traffic Environment)


"""

import os
import time
import numpy as np
import pickle
import random
import math
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset


# 创建一个数据集类来处理批处理
class LaneChangeDataset(Dataset):
    def __init__(self, data_list, data_prepare_func):
        self.data_list = data_list
        self.data_prepare_func = data_prepare_func
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        state_sequence, tag, other_info = self.data_list[idx]
        return state_sequence, tag, other_info


# 批处理数据的整理函数
def collate_fn(batch, data_prepare_func, device):
    state_sequences = []
    tags = []
    other_infos = []
    
    # 收集所有样本
    for state_sequence, tag, other_info in batch:
        state_sequences.append(state_sequence)
        tags.append(tag)
        other_infos.append(other_info)
    
    # 找到最长序列的长度（用于填充）
    max_len = max(len(seq) for seq in state_sequences)
    
    # 准备特征张量和标签张量
    batch_size = len(batch)
    feature_dim = len(state_sequences[0][0]) if state_sequences else 0
    
    # 初始化填充后的张量
    padded_features = torch.zeros(max_len, batch_size, feature_dim).to(device)
    padded_labels = torch.zeros(max_len, batch_size, dtype=torch.long).to(device)
    
    # 填充特征和标签
    for i, (seq, tag) in enumerate(zip(state_sequences, tags)):
        seq_tensor = torch.tensor(seq, dtype=torch.float).to(device)
        padded_features[:len(seq), i, :] = seq_tensor
        
        # 标签是每个时间步的标签，都是相同的
        labels = torch.tensor([tag] * len(seq), dtype=torch.long).to(device)
        padded_labels[:len(seq), i] = labels
    
    return padded_features, padded_labels, other_infos


def transformer_train(opt, net_path, model, input_data, data_prepare_func):
    # 设置超参数
    num_epoch = opt.num_epoch
    batch_size = opt.batch_size  # 可以根据需要调整批大小
    
    print(f"数据集大小: {len(input_data)}")
    size = len(input_data)
    training_data = input_data[:int(size * 0.98)]
    testing_data = input_data[int(size * 0.98):]
    
    # 创建数据集和数据加载器
    train_dataset = LaneChangeDataset(training_data, data_prepare_func)
    test_dataset = LaneChangeDataset(testing_data, data_prepare_func)
    
    # 创建数据加载器，使用自定义的collate_fn
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, data_prepare_func, opt.device)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, data_prepare_func, opt.device)
    )
    
    print(f"训练集: {len(train_dataset)}个样本, {len(train_loader)}批")
    print(f"测试集: {len(test_dataset)}个样本, {len(test_loader)}批")
    
    # 分析训练集类别分布
    train_labels = [tag for _, tag, _ in training_data]
    class_counts = {}
    for label in train_labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    print("训练集类别分布:")
    for label, count in class_counts.items():
        print(f"类别 {label}: {count} 样本 ({count / len(train_labels):.2%})")
    
    # 计算类别权重，处理不平衡数据
    class_weights = None
    if len(class_counts) > 1:
        # 计算逆频率作为权重
        weights = [1.0 / class_counts.get(i, 1) for i in range(3)]  # 3个类别
        class_weights = torch.tensor(weights, device=opt.device)
        print(f"类别权重: {weights}")
    
    model = model.to(opt.device)
    
    # 使用NLLLoss，适用于log_softmax的输出
    loss_function = nn.NLLLoss(weight=class_weights)
    
    # 设置优化器 - 使用AdamW而不是Adam
    initial_lr = 0.001
    weight_decay = 1e-4  # L2正则化
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=initial_lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 使用OneCycleLR学习率调度器，适合Transformer模型
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * num_epoch
    scheduler = OneCycleLR(
        optimizer,
        max_lr=initial_lr,
        total_steps=total_steps,
        pct_start=0.1,  # 10%的时间用于预热
        div_factor=10.0,  # 初始学习率 = max_lr/div_factor
        final_div_factor=100.0,  # 最终学习率 = max_lr/(div_factor*final_div_factor)
        anneal_strategy='cos'  # 余弦退火
    )
    
    # 初始化最佳指标
    best_metrics = {
        'accuracy': 0.0,
        'loss': float('inf')
    }
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    for epoch in range(1, num_epoch + 1):
        # 打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch}/{num_epoch}, Learning Rate: {current_lr:.6f}")
        
        start = time.time()
        epoch_loss = 0.0
        correct, total = 0, 0
        
        # 训练阶段
        model.train()
        for batch_idx, (features, labels, _) in enumerate(train_loader):
            # 清除梯度
            optimizer.zero_grad()
            
            # 前向传播
            # features: [seq_len, batch_size, input_dim]
            # 模型预期输入: [seq_len, batch_size, input_dim]
            tag_scores = model(features)
            
            # 只使用最后一个有效时间步的输出
            # 创建一个布尔掩码，指示哪些位置有实际数据（非零）
            # 假设特征全为0表示填充数据
            non_zero_mask = (features.sum(dim=2) != 0)  # [seq_len, batch_size]
            
            batch_loss = 0
            batch_size = features.size(1)
            
            for b in range(batch_size):
                # 找到该样本的最后一个非零位置
                valid_indices = torch.nonzero(non_zero_mask[:, b])
                if len(valid_indices) == 0:
                    continue  # 跳过完全填充的样本
                
                # 获取该样本最后一个有效时间步的预测和标签
                final_score = tag_scores[0, b:b + 1]
                final_label = labels[-1, b:b + 1]
                
                # 计算损失
                loss = loss_function(final_score, final_label)
                batch_loss += loss
                
                # 计算准确率
                _, predicted = final_score.max(1)
                total += 1
                correct += (predicted == final_label).sum().item()
            
            # 平均损失
            batch_loss = batch_loss / batch_size
            epoch_loss += batch_loss.item()
            
            # 反向传播和优化
            batch_loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # # 打印批次进度
            # if (batch_idx + 1) % 10 == 0 or batch_idx == len(train_loader) - 1:
            #     print(f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {batch_loss.item():.4f}, "
            #           f"Acc: {correct / total:.4f}")
        
        # 计算训练集平均损失和准确率
        train_loss = epoch_loss / len(train_loader)
        train_acc = correct / total if total > 0 else 0
        
        end = time.time()
        print(f"训练时间: {end - start:.2f}秒")
        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
        
        # 测试阶段
        model.eval()
        test_loss = 0.0
        test_predictions = []
        test_true_labels = []
        miss, false_alarm = 0, 0
        correct, total = 0, 0
        
        with torch.no_grad():
            for features, labels, _ in test_loader:
                # 前向传播
                tag_scores = model(features)
                
                # 处理每个样本
                batch_size = features.size(1)
                
                # 找到每个样本的最后一个有效时间步
                non_zero_mask = (features.sum(dim=2) != 0)  # [seq_len, batch_size]
                
                batch_loss = 0
                
                for b in range(batch_size):
                    # 找到该样本的最后一个非零位置
                    valid_indices = torch.nonzero(non_zero_mask[:, b])
                    if len(valid_indices) == 0:
                        continue  # 跳过完全填充的样本
                    
                    # 获取该样本最后一个有效时间步的预测和标签
                    final_score = tag_scores[0, b:b + 1]
                    final_label = labels[-1, b:b + 1]
                    
                    # 计算损失
                    loss = loss_function(final_score, final_label)
                    batch_loss += loss
                    
                    # 记录预测和真实标签
                    _, predicted = final_score.max(1)
                    pred_idx = predicted.item()
                    true_idx = final_label.item()
                    
                    test_predictions.append(pred_idx)
                    test_true_labels.append(true_idx)
                    
                    # 统计miss和false alarm
                    if true_idx != 0 and pred_idx == 0:
                        miss += 1
                    if true_idx == 0 and pred_idx != 0:
                        false_alarm += 1
                    
                    if pred_idx == true_idx:
                        correct += 1
                    total += 1
                
                # 平均损失
                batch_loss = batch_loss / batch_size
                test_loss += batch_loss.item()
        
        # 计算测试集指标
        test_loss = test_loss / len(test_loader) if len(test_loader) > 0 else 0
        test_acc = correct / total if total > 0 else 0
        miss_rate = miss / total if total > 0 else 0
        false_alarm_rate = false_alarm / total if total > 0 else 0
        
        print(f"测试损失: {test_loss:.4f}")
        print(f"测试准确率: {test_acc:.4f}")
        print(f"Miss率: {miss_rate:.4f}")
        print(f"误报率: {false_alarm_rate:.4f}")
        
        # 每5个epoch打印混淆矩阵
        if epoch % 5 == 0 or epoch == num_epoch:
            cm = confusion_matrix(test_true_labels, test_predictions)
            print("混淆矩阵:")
            print(cm)
        
        # 判断是否需要保存模型
        improved = False
        
        # # 基于测试准确率保存模型
        # if test_acc > best_metrics['accuracy']:
        #     best_metrics['accuracy'] = test_acc
        #     improved = True
        #     print(f"新的最佳准确率: {test_acc:.4f}")
        #
        # # 基于测试损失保存模型
        # if test_loss < best_metrics['loss']:
        #     best_metrics['loss'] = test_loss
        #     improved = True
        #     print(f"新的最低测试损失: {test_loss:.4f}")
        
        if epoch == num_epoch:
            improved = True
            
        if improved:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                'best_metrics': best_metrics,
            }, net_path)
            print(f"模型已保存到 {net_path}")
    
    print("训练完成!")
    
    
def transformer_train_no_batch(opt, net_path, model, input_data, data_prepare_func):
    # 设置超参数
    num_epoch = 100
    
    print(f"数据集大小: {len(input_data)}")
    size = len(input_data)
    training_set = input_data[:int(size * 0.85)]
    testing_set = input_data[int(size * 0.85):]
    
    print(f"训练集: {len(training_set)}个样本")
    print(f"测试集: {len(testing_set)}个样本")
    
    # 分析训练集类别分布
    train_labels = [tag for _, tag, _ in training_set]
    class_counts = {}
    for label in train_labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    print("训练集类别分布:")
    for label, count in class_counts.items():
        print(f"类别 {label}: {count} 样本 ({count / len(train_labels):.2%})")
    
    # 计算类别权重，处理不平衡数据
    class_weights = None
    if len(class_counts) > 1:
        # 计算逆频率作为权重
        weights = [1.0 / class_counts.get(i, 1) for i in range(3)]  # 3个类别
        class_weights = torch.tensor(weights, device=opt.device)
        print(f"类别权重: {weights}")
    
    model = model.to(opt.device)
    
    # 使用NLLLoss，适用于已经softmax的输出
    loss_function = nn.NLLLoss(weight=class_weights)
    
    # 设置优化器 - 使用AdamW而不是Adam
    initial_lr = 0.001
    weight_decay = 1e-4  # L2正则化
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=initial_lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 使用OneCycleLR学习率调度器，适合Transformer模型
    steps_per_epoch = len(training_set)
    total_steps = steps_per_epoch * num_epoch
    scheduler = OneCycleLR(
        optimizer,
        max_lr=initial_lr,
        total_steps=total_steps,
        pct_start=0.1,  # 10%的时间用于预热
        div_factor=10.0,  # 初始学习率 = max_lr/div_factor
        final_div_factor=100.0,  # 最终学习率 = max_lr/(div_factor*final_div_factor)
        anneal_strategy='cos'  # 余弦退火
    )
    
    # 初始化最佳指标
    best_metrics = {
        'accuracy': 0.0,
        'loss': float('inf')
    }
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    for epoch in range(1, num_epoch + 1):
        # 打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch}/{num_epoch}, Learning Rate: {current_lr:.6f}")
        
        random.shuffle(training_set)
        start = time.time()
        epoch_loss = 0.0
        correct, total = 0, 0
        
        # 训练阶段
        model.train()
        for batch_idx, (state_sequence, tag, other_info) in enumerate(training_set):
            # 清除梯度
            optimizer.zero_grad()
            
            # 准备输入数据
            sentence_in, label = data_prepare_func(state_sequence, tag)
            
            # 检查输入形状
            if sentence_in.dim() == 2:  # [seq_len, features]
                sentence_in = sentence_in.unsqueeze(1)  # [seq_len, 1, features]
            
            # 前向传播
            tag_scores = model(sentence_in)
            
            # 只使用最后一个时间步的输出
            final_scores = tag_scores[-1]  # [1, 3]
            final_label = label[-1]  # 最后一个时间步的标签
            final_label = final_label.unsqueeze(0)  # 确保只有一个标签
            
            # 计算损失
            loss = loss_function(final_scores, final_label)
            epoch_loss += loss.item()
            
            # 反向传播和优化
            loss.backward()
            
            optimizer.step()
            scheduler.step()
            
            # 计算准确率
            _, predicted = final_scores.max(1)
            total += final_label.size(0)
            correct += (predicted == final_label).sum().item()
        
        # 计算训练集平均损失和准确率
        train_loss = epoch_loss / len(training_set)
        train_acc = correct / total
        
        end = time.time()
        print(f"训练时间: {end - start:.2f}秒")
        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
        
        # 测试阶段
        model.eval()
        test_loss = 0.0
        test_predictions = []
        test_true_labels = []
        miss, false_alarm = 0, 0
        correct, total = 0, 0
        
        with torch.no_grad():
            for state_sequence, tag, other_info in testing_set:
                sentence_in, label = data_prepare_func(state_sequence, tag)
                
                if sentence_in.dim() == 2:
                    sentence_in = sentence_in.unsqueeze(1)
                
                # 前向传播
                tag_scores = model(sentence_in)
                
                # 只使用最后一个时间步
                final_scores = tag_scores[-1]
                final_label = label[-1]
                final_label = final_label.unsqueeze(0)  # 确保只有一个标签
                
                # 计算损失
                loss = loss_function(final_scores, final_label)
                test_loss += loss.item()
                
                # 记录预测和真实标签
                _, predicted = final_scores.max(1)
                pred_idx = predicted.item()
                true_idx = final_label.item()
                
                test_predictions.append(pred_idx)
                test_true_labels.append(true_idx)
                
                # 统计miss和false alarm
                if true_idx != 0 and pred_idx == 0:
                    miss += 1
                if true_idx == 0 and pred_idx != 0:
                    false_alarm += 1
                
                if pred_idx == true_idx:
                    correct += 1
                total += 1
        
        # 计算测试集指标
        test_loss = test_loss / len(testing_set)
        test_acc = correct / total
        miss_rate = miss / total
        false_alarm_rate = false_alarm / total
        
        print(f"测试损失: {test_loss:.4f}")
        print(f"测试准确率: {test_acc:.4f}")
        print(f"Miss率: {miss_rate:.4f}")
        print(f"误报率: {false_alarm_rate:.4f}")
        
        # 判断是否需要保存模型
        improved = False
        
        # 基于测试准确率保存模型
        if test_acc > best_metrics['accuracy']:
            best_metrics['accuracy'] = test_acc
            improved = True
            print(f"新的最佳准确率: {test_acc:.4f}")
        
        # 基于测试损失保存模型
        if test_loss < best_metrics['loss']:
            best_metrics['loss'] = test_loss
            improved = True
            print(f"新的最低测试损失: {test_loss:.4f}")
        
        if improved:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                'best_metrics': best_metrics,
            }, net_path)
            print(f"模型已保存到 {net_path}")
    
    print("训练完成!")


def load_model(model, net_path, optimizer=None, scheduler=None):
    """
    加载保存的模型和训练状态

    Args:
        model: 模型实例
        net_path: 模型保存路径
        optimizer: 优化器实例 (可选)
        scheduler: 学习率调度器实例 (可选)

    Returns:
        model: 加载权重后的模型
        optimizer: 加载状态后的优化器 (如果提供)
        scheduler: 加载状态后的学习率调度器 (如果提供)
        epoch: 训练轮次
        best_metrics: 最佳性能指标
    """
    if not os.path.exists(net_path):
        print(f"没有找到保存的模型: {net_path}")
        return model, optimizer, scheduler, 0, {'accuracy': 0, 'loss': float('inf')}
    
    checkpoint = torch.load(net_path)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 如果提供了优化器，加载优化器状态
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 如果提供了学习率调度器，加载调度器状态
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        if checkpoint['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    best_metrics = checkpoint.get('best_metrics', {'accuracy': 0, 'loss': float('inf')})
    return model, optimizer, scheduler, epoch, best_metrics