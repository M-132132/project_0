
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import special
from torch import optim
from torch.distributions import MultivariateNormal, Laplace
from torch.optim.lr_scheduler import MultiStepLR

# from models.base_model.base_model_pl import BaseModel
from models.base_model.base_model import BaseModel


class MapEncoderCNN(nn.Module):
    """
    用于道路图像的常规 CNN 编码器类。
    该类使用卷积神经网络来编码道路图像，提取特征用于后续处理。
    """
    def __init__(self, d_k=64, dropout=0.1, c=10):
        # 初始化函数，设置模型参数
        # d_k: 特征维度
        # dropout: dropout比率
        # c: 模式数量
        super(MapEncoderCNN, self).__init__()
        self.dropout = dropout
        self.c = c
        # 定义初始化函数，使用xavier正态分布初始化权重
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        
        # MAP ENCODER - 地图编码器部分
        fm_size = 7  # 特征图尺寸
        # 构建卷积神经网络序列
        self.map_encoder = nn.Sequential(
            init_(nn.Conv2d(3, 32, kernel_size=4, stride=1)), nn.ReLU(),  # 第一层卷积，输入3通道，输出32通道
            init_(nn.Conv2d(32, 32, kernel_size=4, stride=2)), nn.ReLU(),  # 第二层卷积，步长为2
            init_(nn.Conv2d(32, 32, kernel_size=3, stride=2)), nn.ReLU(),  # 第三层卷积
            init_(nn.Conv2d(32, 32, kernel_size=3, stride=2)), nn.ReLU(),  # 第四层卷积
            init_(nn.Conv2d(32, fm_size * self.c, kernel_size=2, stride=2)), nn.ReLU(),  # 最后一层卷积
            nn.Dropout2d(p=self.dropout)  # 2D dropout层
        )
        # 特征处理层
        self.map_feats = nn.Sequential(
            init_(nn.Linear(7 * 7 * fm_size, d_k)), nn.ReLU(),  # 全连接层1
            init_(nn.Linear(d_k, d_k)), nn.ReLU(),  # 全连接层2
        )
        self.fisher_information = None  # Fisher信息矩阵
        self.optimal_params = None  # 最优参数

    def forward(self, roads):
        '''
        前向传播函数
        :param roads: 道路图像，尺寸为 (B, 128, 128, 3)，其中B为批次大小
        :return: 返回道路特征，每个模式对应一个特征，尺寸为 (B, c, d_k)
        '''
        B = roads.size(0)  # 获取批次大小
        # 通过编码器提取特征并重塑形状，然后通过特征处理层
        return self.map_feats(self.map_encoder(roads).view(B, self.c, -1))


class MapEncoderPts(nn.Module):
    """
    这是一个用于编码道路信息的神经网络模型类。
    它处理形状为(B, num_road_segs, num_pts_per_road_seg, k_attr+1)的道路车道张量，
    其中B是批次大小，num_road_segs是道路段数量，num_pts_per_road_seg是每段道路的点数，
    k_attr+1是每个点的属性数量（+1表示包含一个掩码位）。
    """
    def __init__(self, d_k, map_attr=3, dropout=0.1):
        # 初始化函数
        # d_k: 编码特征的维度
        # map_attr: 地图属性的数量，默认为3
        # dropout: dropout比率，默认为0.1
        super(MapEncoderPts, self).__init__()
        self.dropout = dropout
        self.d_k = d_k
        self.map_attr = map_attr
        # 定义初始化函数，使用Xavier正态分布初始化权重
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        # 定义道路点特征的线性变换层
        self.road_pts_lin = nn.Sequential(init_(nn.Linear(map_attr, self.d_k)))
        # 定义多头自注意力层
        self.road_pts_attn_layer = nn.MultiheadAttention(self.d_k, num_heads=8, dropout=self.dropout)
        # 定义两个层归一化层
        self.norm1 = nn.LayerNorm(self.d_k, eps=1e-5)
        self.norm2 = nn.LayerNorm(self.d_k, eps=1e-5)
        # 定义地图特征提取的序列层
        self.map_feats = nn.Sequential(
            init_(nn.Linear(self.d_k, self.d_k)), nn.ReLU(), nn.Dropout(self.dropout),
            init_(nn.Linear(self.d_k, self.d_k)),
        )

    def get_road_pts_mask(self, roads):
        # 获取道路点的掩码
        # road_segment_mask: 道路段级别的掩码
        # road_pts_mask: 道路点级别的掩码
        road_segment_mask = torch.sum(roads[:, :, :, -1], dim=2) == 0
        road_pts_mask = (1.0 - roads[:, :, :, -1]).type(torch.BoolTensor).to(roads.device).view(-1, roads.shape[2])
        # 确保没有空行导致的NaN值
        road_pts_mask = road_pts_mask.masked_fill((road_pts_mask.sum(-1) == roads.shape[2]).unsqueeze(-1), False)
        return road_segment_mask, road_pts_mask

    def forward(self, roads, agents_emb):
        # 前向传播函数
        # roads: 输入的道路张量，形状为(B, S, P, k_attr+1)
        # agents_emb: 智能体的上下文嵌入，形状为(T_obs, B, d_k)
        # 返回: 编码后的道路段特征和道路段掩码
        B = roads.shape[0]  # 批次大小
        S = roads.shape[1]  # 道路段数量
        P = roads.shape[2]  # 每段道路的点数
        # 获取掩码
        road_segment_mask, road_pts_mask = self.get_road_pts_mask(roads)
        # 对道路点特征进行线性变换
        road_pts_feats = self.road_pts_lin(roads[:, :, :, :self.map_attr]).view(B * S, P, -1).permute(1, 0, 2)

        # 使用注意力机制结合智能体上下文嵌入和道路点特征
        agents_emb = agents_emb[-1].unsqueeze(2).repeat(1, 1, S, 1).view(-1, self.d_k).unsqueeze(0)
        road_seg_emb = self.road_pts_attn_layer(query=agents_emb, key=road_pts_feats, value=road_pts_feats,
                                                key_padding_mask=road_pts_mask)[0]
        # 应用层归一化
        road_seg_emb = self.norm1(road_seg_emb)
        # 应用地图特征提取层
        road_seg_emb2 = road_seg_emb + self.map_feats(road_seg_emb)
        # 再次应用层归一化
        road_seg_emb2 = self.norm2(road_seg_emb2)
        # 调整形状
        road_seg_emb = road_seg_emb2.view(B, S, -1)

        return road_seg_emb.permute(1, 0, 2), road_segment_mask


def init(module, weight_init, bias_init, gain=1):
    """
    This function provides weight and bias initializations for linear layers.
    """
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class OutputModel(nn.Module):
    """
    This class operates on the output of AutoBot-Ego's decoder representation. It produces the parameters of a
    bivariate Gaussian distribution.
    """

    def __init__(self, d_k=64):
        super(OutputModel, self).__init__()
        self.d_k = d_k
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        self.observation_model = nn.Sequential(
            init_(nn.Linear(d_k, d_k)), nn.ReLU(),
            init_(nn.Linear(d_k, d_k)), nn.ReLU(),
            init_(nn.Linear(d_k, 5))
        )
        self.min_stdev = 0.01

    def forward(self, agent_decoder_state):
        T = agent_decoder_state.shape[0]
        BK = agent_decoder_state.shape[1]
        pred_obs = self.observation_model(agent_decoder_state.reshape(-1, self.d_k)).reshape(T, BK, -1)

        x_mean = pred_obs[:, :, 0]
        y_mean = pred_obs[:, :, 1]
        x_sigma = F.softplus(pred_obs[:, :, 2]) + self.min_stdev
        y_sigma = F.softplus(pred_obs[:, :, 3]) + self.min_stdev
        rho = torch.tanh(pred_obs[:, :, 4]) * 0.9  # for stability
        return torch.stack([x_mean, y_mean, x_sigma, y_sigma, rho], dim=2)


# class AutoBotEgo(nn.Module):
class AutoBotEgo(BaseModel):
    """
    AutoBot-Ego 类：用于自动驾驶场景下的轨迹预测模型
    继承自 BaseModel 基类
    """
    def __init__(self, config, k_attr=2, map_attr=2):
        # 调用父类的初始化方法
        super(AutoBotEgo, self).__init__(config)
        self.config = config
        # 初始化函数：使用 Xavier 正态分布初始化权重，偏置初始化为 0
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        
        # 基本参数设置
        self.T = config['future_len']  # 预测未来时间步长
        self.past = config['past_len']  # 过去时间步长
        self.fisher_information = None  # Fisher 信息矩阵
        self.map_attr = map_attr  # 地图属性维度
        self.k_attr = k_attr  # 智能体动态属性维度
        self.d_k = config['hidden_size']  # 隐藏层大小
        self.c = config['num_modes']  # 预测模式数量
        
        # Transformer 相关参数
        self.L_enc = config['num_encoder_layers']  # 编码器层数
        self.dropout = config['dropout']  # dropout 概率
        self.num_heads = config['tx_num_heads']  # 注意力头数
        self.L_dec = config['num_decoder_layers']  # 解码器层数
        self.tx_hidden_size = config['tx_hidden_size']  # Transformer 隐藏层大小
        
        # ============================== 输入编码器 ==============================
        # 智能体动态特征编码器：将输入的动态特征映射到隐藏空间
        self.agents_dynamic_encoder = nn.Sequential(init_(nn.Linear(self.k_attr, self.d_k)))
        
        # ============================== AutoBot-Ego 编码器 ==============================
        # 社交注意力层：处理智能体之间的交互
        self.social_attn_layers = []
        # 时序注意力层：处理时间维度上的依赖
        self.temporal_attn_layers = []
        for _ in range(self.L_enc):
            # 社交注意力层的 Transformer 编码器
            tx_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_k, nhead=self.num_heads, dropout=self.dropout,
                                                          dim_feedforward=self.tx_hidden_size)
            self.social_attn_layers.append(nn.TransformerEncoder(tx_encoder_layer, num_layers=1))
            
            # 时序注意力层的 Transformer 编码器
            tx_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_k, nhead=self.num_heads, dropout=self.dropout,
                                                          dim_feedforward=self.tx_hidden_size)
            self.temporal_attn_layers.append(nn.TransformerEncoder(tx_encoder_layer, num_layers=1))
        
        # 将注意力层列表转换为 ModuleList
        self.temporal_attn_layers = nn.ModuleList(self.temporal_attn_layers)
        self.social_attn_layers = nn.ModuleList(self.social_attn_layers)
        
        # ============================== 地图编码器 ==========================
        # 地图编码器：处理地图信息
        self.map_encoder = MapEncoderPts(d_k=self.d_k, map_attr=self.map_attr, dropout=self.dropout)
        # 地图注意力层：处理地图与智能体之间的交互
        self.map_attn_layers = nn.MultiheadAttention(self.d_k, num_heads=self.num_heads, dropout=0.3)
        
        # ============================== AutoBot-Ego 解码器 ==============================
        # 可学习的查询向量：用于生成多模态预测
        self.Q = nn.Parameter(torch.Tensor(self.T, 1, self.c, self.d_k), requires_grad=True)
        nn.init.xavier_uniform_(self.Q)
        
        # Transformer 解码器层
        self.tx_decoder = []
        for _ in range(self.L_dec):
            self.tx_decoder.append(nn.TransformerDecoderLayer(d_model=self.d_k, nhead=self.num_heads,
                                                              dropout=self.dropout,
                                                              dim_feedforward=self.tx_hidden_size))
        self.tx_decoder = nn.ModuleList(self.tx_decoder)
        
        # ============================== 位置编码器 ==============================
        # 位置编码：为输入序列添加位置信息
        self.pos_encoder = PositionalEncoding(self.d_k, dropout=0.0, max_len=self.past)
        
        # ============================== 输出模型 ==============================
        # 输出模型：生成最终的轨迹预测
        self.output_model = OutputModel(d_k=self.d_k)
        
        # ============================== 模式概率预测 ==============================
        # 模式概率预测的可学习参数：用于预测每种模式的概率
        self.P = nn.Parameter(torch.Tensor(self.c, 1, self.d_k), requires_grad=True)
        nn.init.xavier_uniform_(self.P)
        
        # 模式地图注意力层
        self.mode_map_attn = nn.MultiheadAttention(self.d_k, num_heads=self.num_heads)
        
        # 概率解码器：预测模式概率
        self.prob_decoder = nn.MultiheadAttention(self.d_k, num_heads=self.num_heads, dropout=self.dropout)
        # 概率预测器：将特征映射到概率值
        self.prob_predictor = init_(nn.Linear(self.d_k, 1))
        
        # 损失函数
        self.criterion = Criterion(self.config)
        
        # Fisher 信息矩阵和最优参数（用于模型优化）
        self.fisher_information = None
        self.optimal_params = None
    
    def generate_decoder_mask(self, seq_len, device):
        """
        生成解码器掩码，用于遮挡后续信息
        该函数创建一个上三角矩阵，用于在自注意力机制中防止当前位置关注到未来位置
        Args:
            seq_len: 序列长度
            device: 计算设备（CPU/GPU）
        Returns:
            subsequent_mask: 布尔类型的上三角矩阵，True表示需要遮挡的位置
        """
        # 创建一个上三角矩阵，对角线以上为1
        subsequent_mask = (torch.triu(torch.ones((seq_len, seq_len), device=device), diagonal=1)).bool()
        return subsequent_mask
    
    def process_observations(self, ego, agents):
        """
        处理观测数据，提取智能体动态状态和掩码信息
        Args:
            ego: 自车观测数据，形状为(B, T, A+1)
            agents: 其他智能体观测数据，形状为(B, T, N, A+1)
        Returns:
            ego_tensor: 自车动态状态张量
            opps_tensor: 其他智能体动态状态张量
            opps_masks: 其他智能体的有效掩码
            env_masks: 环境掩码
        """
        # 处理自车相关信息
        ego_tensor = ego[:, :, :self.k_attr]  # 提取自车动态状态
        env_masks_orig = ego[:, :, -1]  # 获取原始环境掩码
        env_masks = (1.0 - env_masks_orig).to(torch.bool)  # 转换为布尔类型的环境掩码
        env_masks = env_masks.unsqueeze(1).repeat(1, self.c, 1).view(ego.shape[0] * self.c, -1)  # 调整掩码形状
        
        # 处理其他智能体相关信息
        temp_masks = torch.cat((torch.ones_like(env_masks_orig.unsqueeze(-1)), agents[:, :, :, -1]),
                               dim=-1)  # 合并自车和其他智能体的掩码
        opps_masks = (1.0 - temp_masks).to(torch.bool)  # 获取其他智能体的有效掩码
        opps_tensor = agents[:, :, :, :self.k_attr]  # 提取其他智能体动态状态
        return ego_tensor, opps_tensor, opps_masks, env_masks
    
    def temporal_attn_fn(self, agents_emb, agent_masks, layer):
        """
        执行时间维度上的注意力计算
        Args:
            agents_emb: 智能体嵌入向量，形状为(T, B, N, H)
            agent_masks: 智能体掩码，形状为(B, T, N)
            layer: 注意力层
        Returns:
            agents_temp_emb: 经过时间注意力处理后的嵌入向量
        """
        T_obs = agents_emb.size(0)  # 观测时间步数
        B = agent_masks.size(0)  # 批次大小
        num_agents = agent_masks.size(2)  # 智能体数量
        temp_masks = agent_masks.permute(0, 2, 1).reshape(-1, T_obs)  # 调整掩码形状以适应时间维度
        temp_masks = temp_masks.masked_fill((temp_masks.sum(-1) == T_obs).unsqueeze(-1), False)  # 处理全无效序列
        # 应用位置编码并通过注意力层
        agents_temp_emb = layer(self.pos_encoder(agents_emb.reshape(T_obs, B * (num_agents), -1)),
                                src_key_padding_mask=temp_masks)
        return agents_temp_emb.view(T_obs, B, num_agents, -1)  # 恢复原始形状
    
    def social_attn_fn(self, agents_emb, agent_masks, layer):
        """
        执行社交维度上的注意力计算
        Args:
            agents_emb: 智能体嵌入向量，形状为(T, B, N, H)
            agent_masks: 智能体掩码，形状为(B, T, N)
            layer: 注意力层
        Returns:
            agents_soc_emb: 经过社交注意力处理后的嵌入向量
        """
        T_obs, B, num_agents, dim = agents_emb.shape  # 获取各维度大小
        # 调整嵌入向量形状以适应社交维度
        agents_emb = agents_emb.permute(2, 1, 0, 3).reshape(num_agents, B * T_obs, -1)
        # 应用注意力层
        agents_soc_emb = layer(agents_emb, src_key_padding_mask=agent_masks.view(-1, num_agents))
        # 恢复原始形状
        agents_soc_emb = agents_soc_emb.view(num_agents, B, T_obs, -1).permute(2, 1, 0, 3)
        return agents_soc_emb
    
    def _forward(self, inputs):
        """
        前向传播函数，处理输入数据并生成预测结果
        :param ego_in: 自车输入数据，形状为[B, T_obs, k_attr+1]，最后一个维度是存在性掩码
        :param agents_in: 其他智能体输入数据，形状为[B, T_obs, M-1, k_attr+1]，最后一个维度是存在性掩码
        :param roads: 道路网络信息，可能是：
                     - 如果使用地图车道线：[B, S, P, map_attr+1]
                     - 如果使用地图图像：[B, 3, 128, 128]
                     - 如果都不使用：[B, 1, 1]
        :return:
            pred_obs: 预测轨迹，形状为[c, T, B, 5]，包含c条自车轨迹，每个点是二元高斯分布的参数
            mode_probs: 模式概率预测，形状为[B, c]，表示P(z|X_{1:T_obs})
        """
        # 解包输入数据
        ego_in, agents_in, roads = inputs['ego_in'], inputs['agents_in'], inputs['roads']
        
        B = ego_in.size(0)  # 获取批次大小
        
        # 处理观测数据，将输入特征编码到高维空间
        ego_tensor, _agents_tensor, opps_masks, env_masks = self.process_observations(ego_in, agents_in)
        agents_tensor = torch.cat((ego_tensor.unsqueeze(2), _agents_tensor), dim=2)
        
        # 使用动态编码器编码智能体数据, fea agents_tensor 2维 -> agents_emb 128维
        agents_emb = self.agents_dynamic_encoder(agents_tensor).permute(1, 0, 2, 3)
        
        # 通过AutoBot的编码器进行处理
        for i in range(self.L_enc):
            # 时间维度上的注意力处理
            agents_emb = self.temporal_attn_fn(agents_emb, opps_masks, layer=self.temporal_attn_layers[i])
            # 社交维度上的注意力处理
            agents_emb = self.social_attn_fn(agents_emb, opps_masks, layer=self.social_attn_layers[i])
        ego_soctemp_emb = agents_emb[:, :, 0]  # 只提取自车的编码
        
        # 处理地图特征
        orig_map_features, orig_road_segs_masks = self.map_encoder(roads, ego_soctemp_emb)
        # 为多模式预测扩展地图特征
        map_features = orig_map_features.unsqueeze(2).repeat(1, 1, self.c, 1).view(-1, B * self.c, self.d_k)
        road_segs_masks = orig_road_segs_masks.unsqueeze(1).repeat(1, self.c, 1).view(B * self.c, -1)
        
        # 为高效前向传播重复上下文张量
        context = ego_soctemp_emb.unsqueeze(2).repeat(1, 1, self.c, 1)
        context = context.view(-1, B * self.c, self.d_k)
        
        # AutoBot-Ego解码过程
        out_seq = self.Q.repeat(1, B, 1, 1).view(self.T, B * self.c, -1)  # 初始化输出序列
        time_masks = self.generate_decoder_mask(seq_len=self.T, device=ego_in.device)  # 生成时间掩码
        for d in range(self.L_dec):
            # 地图注意力处理
            ego_dec_emb_map = self.map_attn_layers(query=out_seq, key=map_features, value=map_features,
                                                   key_padding_mask=road_segs_masks)[0]
            out_seq = out_seq + ego_dec_emb_map  # 残差连接
            # 解码器处理
            out_seq = self.tx_decoder[d](out_seq, context, tgt_mask=time_masks, memory_key_padding_mask=env_masks)
        # 生成输出分布
        out_dists = self.output_model(out_seq).reshape(self.T, B, self.c, -1).permute(2, 0, 1, 3)

        # 模式概率预测
        mode_params_emb = self.P.repeat(1, B, 1)  # 重复模式参数
        mode_params_emb = self.prob_decoder(query=mode_params_emb, key=ego_soctemp_emb, value=ego_soctemp_emb)[0]

        # 地图注意力处理模式参数
        mode_params_emb = self.mode_map_attn(query=mode_params_emb, key=orig_map_features, value=orig_map_features,
                                             key_padding_mask=orig_road_segs_masks)[0] + mode_params_emb  # 残差连接
        # 计算模式概率
        mode_probs = F.softmax(self.prob_predictor(mode_params_emb).squeeze(-1), dim=0).transpose(0, 1)

        # 准备输出字典
        output = {}
        output['predicted_probability'] = mode_probs  # [B, c] 存储模式概率
        # 调整输出形状以便并行化处理
        output['predicted_trajectory'] = out_dists.permute(2, 0, 1, 3)  # [c, T, B, 5] -> [B, c, T, 5] 存储预测轨迹
        return output

    def forward(self, batch):
        """
        主前向传播函数，处理批量数据并计算损失

        :param batch: 包含输入数据的批量字典
        :return: 包含预测结果和损失的元组
        """
        # 准备模型输入
        model_input = {}
        inputs = batch['input_dict']
        
        # agents_in: [Batch Size 61, T_obs时间步长 15, M最大智能体数量num_agents 21, 特征数量 39]
        # 特征数量: 基础轨迹信息 (0-5): [x, y, z, length, width, height]; 类别one-hot编码 (6-10): 5维;
        # 时间嵌入 (11-31): 21维; 航向编码 (32-33): 2维; 速度信息 (34-35): 2维; 加速度信息 (36-37): 2维; 掩码信息 (38): 1维;
        
        # roads: [Batch Size 61, 场景中的道路段总数max_num_roads256,
        # 道路点数量max_points_per_lane20, 特征数量 29]
        # 特征: 位置信息 (0-2)[x, y, z]; 方向信息 (3-4): [dx, dy]; 前一位置 (5-7): [prev_x, prev_y, prev_z];
        # 地图类型编码 (8-27): 20维one-hot编码; 有效点掩码: 标记该点是否有效 (28)
        agents_in, agents_mask, roads = inputs['obj_trajs'], inputs['obj_trajs_mask'], inputs['map_polylines']
        # 提取自车数据
        ego_in = torch.gather(agents_in, 1, inputs['track_index_to_predict'].
                              view(-1, 1, 1, 1).repeat(1, 1, *agents_in.shape[-2:])).squeeze(1)
        ego_mask = torch.gather(agents_mask, 1, inputs['track_index_to_predict'].
                                view(-1, 1, 1).repeat(1, 1, agents_mask.shape[-1])).squeeze(1)
        
        # 处理智能体数据，添加掩码
        agents_in = torch.cat([agents_in[..., :2], agents_mask.unsqueeze(-1)], dim=-1)
        agents_in = agents_in.transpose(1, 2)
        ego_in = torch.cat([ego_in[..., :2], ego_mask.unsqueeze(-1)], dim=-1)
        
        # 处理道路数据，添加掩码
        roads = torch.cat([inputs['map_polylines'][..., :2],
                           inputs['map_polylines_mask'].unsqueeze(-1)], dim=-1)
        
        # 组装模型输入
        # 这里 ego_in, agents_in, roads 都是只有前三维的位置特征了
        model_input['ego_in'] = ego_in
        model_input['agents_in'] = agents_in
        model_input['roads'] = roads
        
        # 执行前向传播, model_input 是一个字典, 就要对它进行归因
        output = self._forward(model_input)
        # 计算损失
        loss = self.get_loss(batch, output)
        
        # 注释掉的代码：训练和测试时的不同处理方式
        # if self.training:
        #     loss = self.get_loss(batch, output)
        # else:
        #     loss = 0
        return output, loss

    def get_loss(self, batch, prediction):
        inputs = batch['input_dict']
        ground_truth = torch.cat([inputs['center_gt_trajs'][..., :2], inputs['center_gt_trajs_mask'].unsqueeze(-1)],
                                 dim=-1)
        loss = self.criterion(prediction, ground_truth, inputs['center_gt_final_valid_idx'])
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config['learning_rate'], eps=0.0001)
        scheduler = MultiStepLR(optimizer, milestones=self.config['learning_rate_sched'], gamma=0.5)
        return [optimizer], [scheduler]


class Criterion(nn.Module):
    """
    轨迹预测损失函数类
    实现了多模态预测的复合损失函数，包含：
    - 多模态负对数似然损失
    - 熵损失
    - KL散度损失
    - ADE/FDE辅助损失
    """
    def __init__(self, config):
        super(Criterion, self).__init__()
        self.config = config

    def forward(self, out, gt, center_gt_final_valid_idx):
        return self.nll_loss_multimodes(out, gt, center_gt_final_valid_idx)

    def get_BVG_distributions(self, pred):
        """
        获取二元高斯分布
        Args:
            pred: 预测参数 [mu_x, mu_y, sigma_x, sigma_y, rho]
        Returns:
            biv_gauss_dist: 二元高斯分布对象
        """
        B = pred.size(0)
        T = pred.size(1)
        mu_x = pred[:, :, 0].unsqueeze(2)
        mu_y = pred[:, :, 1].unsqueeze(2)
        sigma_x = pred[:, :, 2]
        sigma_y = pred[:, :, 3]
        rho = pred[:, :, 4]
        
        # 创建单个元素的协方差矩阵
        cov = torch.stack([
            torch.stack([sigma_x ** 2, rho * sigma_x * sigma_y], dim=-1),
            torch.stack([rho * sigma_x * sigma_y, sigma_y ** 2], dim=-1)
        ], dim=-2)

        # 扩展基础矩阵以匹配所需的形状
        """
        获取拉普拉斯分布
        Args:
            pred: 预测参数 [mu_x, mu_y, scale_x, scale_y]
        Returns:
            Laplace分布对象
        """
        biv_gauss_dist = MultivariateNormal(loc=torch.cat((mu_x, mu_y), dim=-1), covariance_matrix=cov,validate_args=False)
        return biv_gauss_dist

    def get_Laplace_dist(self, pred):
        return Laplace(pred[:, :, :2], pred[:, :, 2:4],validate_args=False)

    def nll_pytorch_dist(self, pred, data, mask, rtn_loss=True):
        """
        计算负对数似然损失
        Args:
            pred: 预测参数
            data: 真实数据
            mask: 掩码矩阵，指示有效位置
            rtn_loss: 是否返回损失值
        Returns:
            计算得到的损失值或概率值
        """
        # 使用拉普拉斯分布而非高斯分布
        # biv_gauss_dist = get_BVG_distributions(pred)
        biv_gauss_dist = self.get_Laplace_dist(pred)
        num_active_per_timestep = mask.sum()
        data_reshaped = data[:, :, :2]
        if rtn_loss:
            # 计算拉普拉斯分布的负对数似然损失
            # return (-biv_gauss_dist.log_prob(data)).sum(1)  # Gauss
            return ((-biv_gauss_dist.log_prob(data_reshaped)).sum(-1) * mask).sum(1)  # Laplace
        else:
            # 不返回损失值时，考虑掩码的影响
            # return (-biv_gauss_dist.log_prob(data)).sum(-1)  # Gauss
            # need to multiply by masks
            # return (-biv_gauss_dist.log_prob(data_reshaped)).sum(dim=(1, 2))  # Laplace
            return ((-biv_gauss_dist.log_prob(data_reshaped)).sum(dim=2) * mask).sum(1)  # Laplace

    def nll_loss_multimodes(self, output, data, center_gt_final_valid_idx):
        """
        多模态负对数似然损失函数
        Args:
            output: 模型输出，包含预测轨迹和概率
            data: 真实轨迹数据
            center_gt_final_valid_idx: 中心真实轨迹的有效索引
        Returns:
            final_loss: 总损失值
        """
        modes_pred = output['predicted_probability']
        pred = output['predicted_trajectory'].permute(1, 2, 0, 3)
        mask = data[..., -1]

        entropy_weight = self.config['entropy_weight']
        kl_weight = self.config['kl_weight']
        use_FDEADE_aux_loss = self.config['use_FDEADE_aux_loss']

        modes = len(pred)
        nSteps, batch_sz, dim = pred[0].shape

        log_lik_list = []
        with torch.no_grad():
            for kk in range(modes):
                # 计算每个模态的负对数似然
                nll = self.nll_pytorch_dist(pred[kk].transpose(0, 1), data, mask, rtn_loss=False)
                log_lik_list.append(-nll.unsqueeze(1))  # Add a new dimension to concatenate later
            
            # 计算后验概率
            log_lik = torch.cat(log_lik_list, dim=1)

            priors = modes_pred
            log_priors = torch.log(priors)
            log_posterior_unnorm = log_lik + log_priors

            # Compute logsumexp for normalization, ensuring no in-place operations
            logsumexp = torch.logsumexp(log_posterior_unnorm, dim=-1, keepdim=True)
            log_posterior = log_posterior_unnorm - logsumexp

            # Compute the posterior probabilities without in-place operations
            post_pr = torch.exp(log_posterior)
            # Ensure post_pr is a tensor on the correct device
            post_pr = post_pr.to(data.device)

        # Compute loss.
        loss = 0.0
        for kk in range(modes):
            nll_k = self.nll_pytorch_dist(pred[kk].transpose(0, 1), data, mask, rtn_loss=True) * post_pr[:, kk]
            loss += nll_k.mean()

        # 添加熵损失项 to ensure that individual predictions do not try to cover multiple modes.
        entropy_vals = []
        for kk in range(modes):
            entropy_vals.append(self.get_BVG_distributions(pred[kk]).entropy())
        entropy_vals = torch.stack(entropy_vals).permute(2, 0, 1)
        entropy_loss = torch.mean((entropy_vals).sum(2).max(1)[0])
        loss += entropy_weight * entropy_loss

        # 计算KL散度损失 between the prior and the posterior distributions.
        kl_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')  # type: ignore
        kl_loss = kl_weight * kl_loss_fn(torch.log(modes_pred), post_pr)

        # 计算ADE/FDE辅助损失 L2 norms between the best predictions and GT.
        if use_FDEADE_aux_loss:
            adefde_loss = self.l2_loss_fde(pred, data, mask)
        else:
            adefde_loss = torch.tensor(0.0).to(data.device)

        # post_entropy
        final_loss = loss + kl_loss + adefde_loss

        return final_loss

    def l2_loss_fde(self, pred, data, mask):
        """
        计算FDE（最终位移误差）和ADE（平均位移误差）损失
        Args:
            pred: 预测轨迹
            data: 真实轨迹
            mask: 掩码矩阵
        Returns:
            100倍的平均损失值
        """
        fde_loss = (torch.norm((pred[:, -1, :, :2].transpose(0, 1) - data[:, -1, :2].unsqueeze(1)),
                               2, dim=-1) * mask[:, -1:])
        ade_loss = (torch.norm((pred[:, :, :, :2].transpose(1, 2) - data[:, :, :2].unsqueeze(0)),
                               2, dim=-1) * mask.unsqueeze(0)).mean(dim=2).transpose(0, 1)
        loss, min_inds = (fde_loss + ade_loss).min(dim=1)
        return 100.0 * loss.mean()


