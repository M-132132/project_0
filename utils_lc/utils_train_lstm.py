
import os
import time
import numpy
import pickle
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR


def lstm_train(opt, net_path, model, input_data, data_prepare_func):
    # set hyper-parameters
    num_epoch = 100
    
    print(len(input_data))
    size = len(input_data)
    training_set = input_data[:int(size*0.9)]
    testing_set = input_data[int(size*0.9):]
    
    model = model.to(opt.device)
    loss_function = nn.NLLLoss()
    initial_lr = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    
    # 方案2：根据验证集效果自适应调整
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5,
                                  verbose=True, min_lr=1e-6)
    
    accuracy_best = 0.7
    
    for epoch in range(1, num_epoch+1):
        # 打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch}, Learning Rate: {current_lr:.6f}")
    
        random.shuffle(training_set)
        start = time.time()
        total_num = 0
        total_loss = 0.0
        
        # 训练阶段
        model.train()
        for state_sequence, tag, other_info in training_set:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            optimizer.zero_grad()
    
            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of state sequences.
            sentence_in, label = data_prepare_func(state_sequence, tag)
    
            # Step 3. Run our forward pass.
            tag_scores = model(sentence_in)
    
            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(tag_scores, label)
            total_loss += loss.item()
            total_num += 1
    
            loss.backward()
            optimizer.step()
    
        end = time.time()
        print("Run time:", end-start)
        print("Training loss:", total_loss/total_num)

        # 评估阶段
        model.eval()
        # 训练集评估
        correct, total = 0, 0
        miss, false_alarm = 0, 0
        with torch.no_grad():
            for state_sequence, tag, other_info in training_set:
                sentence_in, label = data_prepare_func(state_sequence, tag)
                output_scores = model(sentence_in)
                _, idx = output_scores[-1].max(0)
                if tag != 0 and idx == 0:
                    miss += 1
                if tag == 0 and idx != 0:
                    false_alarm += 1
                if idx == tag:
                    correct += 1
                total += 1
            print("Training miss rate:", miss / total)
            print("Training false alarm rate:", false_alarm / total)
            print("Training accuracy:", correct/total)
    
        # See what the scores are after testing
        correct, total = 0, 0
        miss, false_alarm = 0, 0
        with torch.no_grad():
            for state_sequence, tag, other_info in testing_set:
                sentence_in, label = data_prepare_func(state_sequence, tag)
                output_scores = model(sentence_in)
                _, idx = output_scores[-1].max(0)
                if tag != 0 and idx == 0:
                    miss += 1
                if tag == 0 and idx != 0:
                    false_alarm += 1
                if idx == tag:
                    correct += 1
                total += 1
            print("Testing miss rate:", miss / total)
            print("Testing false alarm rate:", false_alarm / total)
            test_acc = correct/total
            print("Testing accuracy:", test_acc)

        # 更新学习率
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(test_acc)  # 使用测试集准确率作为指标
        else:
            scheduler.step()
            
        test_acc = correct/total
        # 保存最佳模型
        if test_acc > accuracy_best and epoch % 1 == 0:
            accuracy_best = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_accuracy': accuracy_best,
            }, net_path)
    print("Training Finished!")


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
        best_accuracy: 最佳准确率
    """
    if not os.path.exists(net_path):
        print(f"没有找到保存的模型: {net_path}")
        return model, optimizer, scheduler, 0, 0
    
    checkpoint = torch.load(net_path)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 如果提供了优化器，加载优化器状态
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 如果提供了学习率调度器，加载调度器状态
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    best_accuracy = checkpoint.get('best_accuracy', 0)
    
    print(f"已加载模型 (epoch {epoch}, 最佳准确率: {best_accuracy:.4f})")
    
    return model, optimizer, scheduler, epoch, best_accuracy
