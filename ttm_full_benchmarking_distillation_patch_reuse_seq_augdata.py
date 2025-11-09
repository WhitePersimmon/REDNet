"""
# TTM zero-shot and few-shot benchmarking on multiple datasets
Pre-trained TTM models will be fetched from the HuggingFace TTM Model Repositories as described below.

1. TTM-Granite-R1 pre-trained models can be found here: [TTM-R1 Model Card](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r1)
2. TTM-Granite-R2 pre-trained models can be found here: [TTM-R2 Model Card](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2)
3. TTM-Research-Use pre-trained models can be found here: [TTM-Research-Use Model Card](https://huggingface.co/ibm-research/ttm-research-r2)

Every model card has a suite of TTM models. Please read the respective model cards for usage instructions.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import copy
import logging
import math
import os
import sys
import tempfile
import warnings
from typing import Optional, Tuple, Union
import gc
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from fvcore.nn import FlopCountAnalysis
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
)
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
from transformers.integrations import INTEGRATION_TO_CALLBACK
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging as transformers_logging,
    replace_return_docstrings,
)

# 设置日志
logger = logging.getLogger(__name__)

# 添加tsfm_public到Python路径
import sys
# sys.path.insert(0, '/home/fuda/edge/granite-tsfm/tsfm_public/toolkit')
sys.path.insert(0, '/home/fuda/edge/granite-tsfm/tsfm_public')
from toolkit.data_handling import load_dataset
from tsfm_public import TrackingCallback, count_parameters
from tsfm_public.models.tinytimemixer.utils import get_ttm_args
from tsfm_public.toolkit.get_model import get_model
from tsfm_public.toolkit.lr_finder import optimal_lr_finder
from tsfm_public.toolkit.visualization import plot_predictions

warnings.filterwarnings("ignore")

os.environ['HTTP_PROXY'] = 'http://10.134.16.27:1079'
os.environ['HTTPS_PROXY'] = 'http://10.134.16.27:1079'

# Arguments
args = get_ttm_args()

# Set seed
set_seed(args.random_seed)

## Important arguments
# Specify model parameters
CONTEXT_LENGTH = args.context_length
FORECAST_LENGTH = args.forecast_length
FREEZE_BACKBONE = True

# Other args
EPOCHS = args.num_epochs
NUM_WORKERS = args.num_workers
# Make sure all the datasets in the following `list_datasets` are
# saved in the `DATA_ROOT_PATH` folder. Or, change it accordingly.
# Refer to the load_datasets() function
# in notebooks/hfdemo/tinytimemixer/utils/ttm_utils.py
# to see how it is used.
DATA_ROOT_PATH = args.data_root_path

# This is where results will be saved
OUT_DIR = args.save_dir

MODEL_PATH = args.hf_model_path

print(f"{'*' * 20} Pre-training a TTM for context len = {CONTEXT_LENGTH}, forecast len = {FORECAST_LENGTH} {'*' * 20}")

## List of benchmark datasets (TTM was not pre-trained on any of these)

if args.datasets is None:
    list_datasets = [
        "etth1",
        "etth2",
        "ettm1",
        "ettm2",
        "weather",
        "electricity",
        "traffic",
        # "exchange",
        "aqshunyi",
        "aqwan",
        "zafnoo",
        "czelan",
        # "solar" # please note that, solar is part of TTM pre-training.
        #         # But, adding here to do in-distribution testing.
        #         # solar results should be ignored for TTM for zero-shot ranking.
    ]

else:
    list_datasets = [dataset.strip() for dataset in args.datasets.split(",")]


class ConceptDriftDataset(Dataset):
    def __init__(self, original_dataset, quadratic_strength=0.005, anomaly_start_pos=200, anomaly_length=10):
        self.original_dataset = original_dataset
        self.quadratic_strength = quadratic_strength
        self.anomaly_start_pos = anomaly_start_pos
        self.anomaly_length = anomaly_length
        
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, idx):
        # 获取原始样本
        sample = self.original_dataset[idx]
        
        # 拷贝样本，避免修改原始数据
        modified_sample = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
        
        # 获取past_values和future_values
        past_values = modified_sample['past_values']
        future_values = modified_sample['future_values']
        
        # 拼接past_values和future_values
        concat_values = torch.cat([past_values, future_values], dim=0)
        
        # 添加batch维度，以匹配apply_concept_drift函数的输入格式
        concat_values_batch = concat_values.unsqueeze(0)  # [1, length, dim]
        
        # 应用概念漂移
        modified_concat_values = apply_concept_drift(
            concat_values_batch, 
            self.quadratic_strength,
            self.anomaly_start_pos,
            self.anomaly_length
        ).squeeze(0)  # 移除batch维度，得到[length, dim]
        
        # 重新分割为past_values和future_values
        past_length = past_values.shape[0]
        modified_sample['past_values'] = modified_concat_values[:past_length]
        modified_sample['future_values'] = modified_concat_values[past_length:]
        
        return modified_sample


def apply_concept_drift(batch_xy, quadratic_strength=0.005, anomaly_start_pos=200, anomaly_length=10):
    """
    对PyTorch张量沿着length维度施加两种概念漂移：
    1. 二次趋势漂移
    2. 局部突变（连续10个点设为均值+3*标准差）
    
    参数:
        batch_xy: PyTorch张量，维度为[batch_size, length, dim]
        quadratic_strength: 二次趋势漂移的强度
        anomaly_start_pos: 局部突变的起始位置
        anomaly_length: 局部突变的长度
        
    返回:
        施加了概念漂移的新PyTorch张量
    """
    # 获取张量的形状信息
    batch_size, length, dim = batch_xy.shape
    device = batch_xy.device
    dtype = batch_xy.dtype
    
    # 创建结果张量的副本
    result = batch_xy.clone()
    
    # 1. 添加二次趋势漂移
    # 创建时间索引向量
    time = torch.arange(length, dtype=dtype, device=device)
    
    # 计算二次趋势
    trend = quadratic_strength * (time ** 2) / length
    
    # 扩展trend维度以便广播
    # [length] -> [1, length, 1]
    trend = trend.view(1, length, 1)
    
    # 应用二次趋势到所有batch和维度
    result = result + trend
    
    # 2. 添加局部突变
    # 计算原始张量的均值和标准差（沿着length维度）
    mean_values = torch.mean(batch_xy, dim=1, keepdim=True)  # [batch_size, 1, dim]
    std_values = torch.std(batch_xy, dim=1, keepdim=True)    # [batch_size, 1, dim]
    
    # 计算突变值：均值 + 3*标准差
    anomaly_values = mean_values + 3 * std_values  # [batch_size, 1, dim]
    
    # 在指定位置应用突变值
    # 创建一个mask用于选择突变位置
    mask = torch.zeros(length, dtype=torch.bool, device=device)
    mask[anomaly_start_pos:anomaly_start_pos+anomaly_length] = True
    
    # 扩展mask维度以便广播
    # [length] -> [1, length, 1]
    mask = mask.view(1, length, 1)
    
    # 将突变值应用到结果张量中
    # 扩展anomaly_values以匹配mask位置的形状
    anomaly_values_expanded = anomaly_values.expand(-1, length, -1)
    result = torch.where(mask, anomaly_values_expanded, result)
    
    return result


# 定义蒸馏的Student模型config
class StudentModelConfig():
    def __init__(self,teacher_config):
        # 数据部分
        self.context_length = teacher_config.context_length
        self.prediction_length = teacher_config.prediction_length

        # 记录一些teacher模型的参数
        self.teacher_d_model = teacher_config.d_model
        self.teacher_decoder_d_model = teacher_config.decoder_d_model
        self.teacher_num_patches = teacher_config.num_patches
        self.teacher_patch_length = teacher_config.patch_length
        self.teacher_patch_stride = teacher_config.patch_stride

        # student模型参数
        # self.d_model = self.teacher_d_model 
        self.d_model = self.teacher_decoder_d_model
        self.norm_mlp = teacher_config.norm_mlp
        self.layernorm_eps = teacher_config.norm_eps

        # student训练过程
        self.loss = teacher_config.loss


# 定义student模型- MLP
class TinyTimeMixerStdScaler(nn.Module):
    """
    Standardize features by calculating the mean and scaling along the first dimension, and then normalizes it by
    subtracting from the mean and dividing by the standard deviation.
    """

    def __init__(self, config):
        super().__init__()
        self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
        self.keepdim = config.keepdim if hasattr(config, "keepdim") else True
        self.minimum_scale = config.minimum_scale if hasattr(config, "minimum_scale") else 1e-5

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters:
            data (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                input for Batch norm calculation
            observed_indicator (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Calculating the scale on the observed indicator.
        Returns:
            tuple of `torch.Tensor` of shapes
                (`(batch_size, sequence_length, num_input_channels)`,`(batch_size, 1, num_input_channels)`,
                `(batch_size, 1, num_input_channels)`)
        """

        denominator = observed_indicator.sum(self.dim, keepdim=self.keepdim)
        denominator = denominator.clamp_min(torch.tensor(1, device=denominator.device))
        loc = (data * observed_indicator).sum(self.dim, keepdim=self.keepdim) / denominator

        variance = (((data - loc) * observed_indicator) ** 2).sum(self.dim, keepdim=self.keepdim) / denominator
        scale = torch.sqrt(variance + self.minimum_scale)
        return (data - loc) / scale, loc, scale


class ForecastingHead(nn.Module):
    def __init__(
        self, head_nf: int = 768 * 64, forecast_horizon: int = 96, head_dropout: int = 0
    ):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(head_nf, forecast_horizon)

    def forward(self, x, past_values=None, future_values=None):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x.transpose(-1, -2) # [batch_size x prediction_length x n_vars]


class StudentModel_PatchMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.scaler = TinyTimeMixerStdScaler(config)

        # TODO 尝试BatchNorm
        # self.norm = nn.LayerNorm(config.d_model, eps=config.layernorm_eps)
        
        # patch
        self.patch_length = config.teacher_patch_length
        self.patch_stride = config.teacher_patch_stride
        self.num_patches = config.teacher_num_patches

        # 定义MLP模型
        # self.backbone = nn.Sequential(
        #     nn.Linear(config.context_length, config.d_model),
        #     nn.LayerNorm(config.d_model, eps=config.layernorm_eps),
        #     nn.ReLU(),
        #     nn.Linear(config.d_model, config.d_model),
        # )
        self.student_model_version = config.student_model_version
        self.backbone = nn.Sequential(
            nn.Linear(self.patch_length, config.d_model),
            nn.LayerNorm(config.d_model, eps=config.layernorm_eps),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
        )
    
        
        self.patch_attention_weights = nn.Parameter(torch.randn(self.num_patches, self.num_patches))
        # self.attention_linear = nn.Linear(config.d_model, self.num_patches)
        self.ln = nn.LayerNorm(config.d_model, eps=config.layernorm_eps)
        self.ffn = nn.Linear(config.d_model, config.d_model)
        
        # 定义输出层
        self.head = ForecastingHead(
            head_nf=config.d_model * self.num_patches,
            forecast_horizon=config.prediction_length,
            head_dropout=0.0,
        )

        self.loss = config.loss

        # teacher batch_size, num_input_channels, num_patch, d_model
        # 完成到教师的映射过程
        self.teacher_num_patches = config.teacher_num_patches

        self.teacher_d_model = config.teacher_d_model
        # self.teacher_d_model = config.teacher_decoder_d_model


    def forward(
        self,
        past_values: torch.Tensor,
        future_values: Optional[torch.Tensor] = None,
        past_observed_mask: Optional[torch.Tensor] = None,
        future_observed_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_loss: bool = True,
        return_dict: Optional[bool] = None,
        freq_token: Optional[torch.Tensor] = None,
        static_categorical_values: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        '''
        输入:
        past_values (torch.Tensor) - 过去的时间序列值(batch_size, sequence_length, num_input_channels)
        future_values (torch.Tensor) - 未来的时间序列值(batch_size, sequence_length, num_input_channels)
        past_observed_mask (torch.Tensor) - 过去时间序列的观测掩码
        future_observed_mask (torch.Tensor) - 未来时间序列的观测掩码
        output_hidden_states (bool) - 是否返回隐藏状态
        return_loss (bool) - 是否返回损失
        return_dict (bool) - 是否返回字典
        freq_token (torch.Tensor) - 频率标记
        static_categorical_values (torch.Tensor) - 静态分类值

        输出:
        ModelOutput - 模型输出
        '''
        if self.loss == "mse":
            loss = nn.MSELoss(reduction="mean")
        elif self.loss == "mae":
            loss = nn.L1Loss(reduction="mean")
        
        if past_observed_mask is None:
            past_observed_mask = torch.ones_like(past_values)
        scaled_past_values, loc, scale = self.scaler(past_values, past_observed_mask) # batch_size, sequence_length, num_input_channels

        # output: [bs x num_patches x num_input_channels x patch_length]
        scaled_past_values = scaled_past_values.unfold(dimension=-2, size=self.patch_length, step=self.patch_stride)
        # output: [bs x num_input_channels x num_patches x patch_length]
        scaled_past_values = scaled_past_values.transpose(-2, -3).contiguous()

        # 通过过去时间序列值计算隐藏状态
        # hidden_states = self.backbone(scaled_past_values)
        patch_embeddings = self.backbone(scaled_past_values) # Shape: [B, C, N, D]

        # 3. Patch Attention Weighted Sum
        # attention_weights shape: [N, N]
        attention_scores = F.softmax(self.patch_attention_weights, dim=-1) # Shape: [N, N]
        # Reshape patch_embeddings for matmul: [B*C, N, D]
        B, C, N, D = patch_embeddings.shape
        patch_embeddings_reshaped = patch_embeddings.contiguous().view(B * C, N, D)
        # Apply attention: attention_scores[N, N] @ patch_embeddings_reshaped[B*C, N, D] -> [B*C, N, D]
        # Note: matmul([M, K], [B, K, P]) -> [B, M, P]. Here M=N, K=N, B=B*C, P=D
        # So we need matmul(attention_scores, patch_embeddings_reshaped)
        weighted_embeddings_reshaped = torch.matmul(attention_scores, patch_embeddings_reshaped) # Shape: [B*C, N, D]
        # Reshape back: [B, C, N, D]
        hidden_states = self.ln(patch_embeddings + weighted_embeddings_reshaped.view(B, C, N, D))
        hidden_states = self.ffn(hidden_states)
        
        # 完成对教师的映射
        hidden_states_to_teacher = hidden_states # [batch_size x num_input_channels x num_patches x d_model]

        y_hat = self.head(hidden_states, past_values=past_values, future_values=None)
        
        y_hat = y_hat * scale + loc # [batch_size x prediction_length x num_input_channels]

        if future_values is None:
            return y_hat
        
        # loss计算
        loss_val = loss(y_hat, future_values)

        # 返回模型输出
        return ModelOutput(
            loss=loss_val,
            prediction_outputs=y_hat,
            hidden_states_to_teacher=hidden_states_to_teacher,
        )    


# 定义 DistillationTrainer 类
class DistillationTrainerOnlyHidden(Trainer):
    def __init__(self, teacher_model, alpha_hidden_states=0.01, beta_teacher_prediction=0.5, *args, **kwargs):
        # 初始化父类，self.model 为学生模型
        super().__init__(*args, **kwargs)
        # 保存教师模型
        self.teacher_model = teacher_model
        
        # 设置教师模型为评估模式，确保不更新参数
        self.teacher_model.eval()
        # 设置损失权重
        self.alpha_hidden_states = alpha_hidden_states  # backbone_hidden_state 的损失权重
        self.beta_teacher_prediction = beta_teacher_prediction    # prediction_outputs 的损失权重

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        '''
        计算知识蒸馏的损失

        输入:
        model (PreTrainedModel) - 学生模型
        inputs (dict) - 模型输入字典，包含训练数据
        return_outputs (bool) - 是否返回模型输出

        输出:
        loss (torch.Tensor) - 总损失 (alpha_hidden_states * backbone_loss + beta_teacher_prediction * prediction_loss)
        '''
        
        # 数据增强: 频域信息交换 - 优化并行处理
        augmented_inputs = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # 获取原始的past_values和future_values
        past_values = augmented_inputs['past_values']  # [batch_size, sequence_length, num_input_channels]
        future_values = augmented_inputs['future_values']  # [batch_size, sequence_length, num_input_channels]
        batch_size, seq_len, channels = past_values.shape
        _, pred_len, _ = future_values.shape
        
        # 沿时间维度拼接
        concatenated = torch.cat([past_values, future_values], dim=1)  # [batch_size, seq_len+pred_len, channels]
        
        # 变形为[batch_size*channels, seq_len+pred_len]，每个通道作为独立序列
        total_seq_len = seq_len + pred_len
        reshaped = concatenated.permute(0, 2, 1).reshape(-1, total_seq_len)  # [batch_size*channels, seq_len+pred_len]
        num_sequences = reshaped.shape[0]  # batch_size*channels
        
        # 进行FFT变换，每个序列独立进行
        fft = torch.fft.fft(reshaped, dim=1)  # [batch_size*channels, seq_len+pred_len]
        
        # 为每个序列生成一个随机布尔掩码，决定哪些频域位置需要交换
        exchange_prob = 0.9
        bool_masks = torch.rand(num_sequences, total_seq_len, device=reshaped.device) < exchange_prob
        
        # 创建交换操作的源索引矩阵
        # 为每个序列的每个位置随机生成一个源序列索引(排除自身)
        source_indices = torch.randint(0, num_sequences-1, (num_sequences, total_seq_len), device=reshaped.device)
        
        # 调整源索引以排除自身序列
        for i in range(num_sequences):
            source_indices[i] += (source_indices[i] >= i).long()
        
        # 创建一个新的FFT张量用于存储替换后的结果
        fft_exchanged = fft.clone()
        
        # 创建所有序列和位置的索引网格
        seq_indices, pos_indices = torch.meshgrid(
            torch.arange(num_sequences, device=reshaped.device),
            torch.arange(total_seq_len, device=reshaped.device),
            indexing='ij'
        )
        
        # 只保留需要交换的索引
        exchange_seq_indices = seq_indices[bool_masks]
        exchange_pos_indices = pos_indices[bool_masks]
        exchange_source_indices = source_indices[bool_masks]
        
        # 执行批量交换操作
        fft_exchanged[exchange_seq_indices, exchange_pos_indices] = fft[exchange_source_indices, exchange_pos_indices]
        
        # 逆变换回时域
        ifft = torch.fft.ifft(fft_exchanged, dim=1)
        restored = ifft.real  # [batch_size*channels, seq_len+pred_len]
        
        # 重新变形回原始维度 [batch_size, seq_len+pred_len, channels]
        restored = restored.reshape(batch_size, channels, total_seq_len).permute(0, 2, 1)
        
        # 将拼接的张量分割回past_values和future_values
        x_aug, y_aug = torch.split(restored, [seq_len, pred_len], dim=1)
        
        # 更新augmented_inputs
        augmented_inputs['past_values'] = x_aug
        augmented_inputs['future_values'] = y_aug
        inputs = augmented_inputs

            
        # 如果teacher不在student的设备上，将teacher移动到student的设备上
        if next(self.teacher_model.parameters()).device != next(model.parameters()).device:
            self.teacher_model.to(next(model.parameters()).device)

        # 计算教师模型输出（不计算梯度）
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
        # 计算学生模型输出
        student_outputs = model(**inputs)
        # 定义均方误差损失函数
        mse_loss_fn = nn.MSELoss(reduction="mean")
        
        # 计算表征部分的损失
        # 使用backbone_hidden_state作为教师的隐藏状态
        # loss_backbone = mse_loss_fn(
        #     student_outputs.hidden_states_to_teacher, teacher_outputs.backbone_hidden_state
        # )
        # 使用decoder_hidden_state作为教师的隐藏状态
        loss_backbone = mse_loss_fn(
            student_outputs.hidden_states_to_teacher, teacher_outputs.decoder_hidden_state
        )
        
        # loss仅仅使用表征部分的差异
        # loss = loss_backbone

        # 计算 prediction_outputs 部分的损失
        loss_prediction = mse_loss_fn(
            student_outputs.prediction_outputs, teacher_outputs.prediction_outputs
        )

        loss = loss_backbone + loss_prediction
        # 合并损失
        # loss = student_outputs.loss \
        #         + self.alpha_hidden_states * loss_backbone \
        #         + self.beta_teacher_prediction * loss_prediction
        
        if return_outputs:
            return loss, student_outputs
        else:
            return loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        重写prediction_step确保正确提取prediction_outputs
        """
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss
            
            # 确保提取正确的预测值
            logits = outputs.prediction_outputs
        
        return (loss, logits, inputs["future_values"])
    

class DistillationTrainerByLabel(Trainer):
    def __init__(self, teacher_model, alpha_hidden_states=0.01, beta_teacher_prediction=0.0, *args, **kwargs):
        # 初始化父类，self.model 为学生模型
        super().__init__(*args, **kwargs)
        # 保存教师模型
        self.teacher_model = teacher_model
        
        # 设置教师模型为评估模式，确保不更新参数
        self.teacher_model.eval()
        # 设置损失权重
        self.alpha_hidden_states = alpha_hidden_states  # backbone_hidden_state 的损失权重
        self.beta_teacher_prediction = beta_teacher_prediction    # prediction_outputs 的损失权重

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        '''
        计算知识蒸馏的损失

        输入:
        model (PreTrainedModel) - 学生模型
        inputs (dict) - 模型输入字典，包含训练数据
        return_outputs (bool) - 是否返回模型输出

        输出:
        loss (torch.Tensor) - 总损失 (alpha_hidden_states * backbone_loss + beta_teacher_prediction * prediction_loss)
        '''
        # 如果teacher不在student的设备上，将teacher移动到student的设备上
        if next(self.teacher_model.parameters()).device != next(model.parameters()).device:
            self.teacher_model.to(next(model.parameters()).device)

        # 计算教师模型输出（不计算梯度）
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
        # 计算学生模型输出
        student_outputs = model(**inputs)
        # 定义均方误差损失函数
        mse_loss_fn = nn.MSELoss(reduction="mean")
        
        # 计算表征部分的损失
        # 使用backbone_hidden_state作为教师的隐藏状态
        # loss_backbone = mse_loss_fn(
        #     student_outputs.hidden_states_to_teacher, teacher_outputs.backbone_hidden_state
        # )
        # 使用decoder_hidden_state作为教师的隐藏状态
        loss_backbone = mse_loss_fn(
            student_outputs.hidden_states_to_teacher, teacher_outputs.decoder_hidden_state
        )


        # 计算 prediction_outputs 部分的损失
        loss_prediction = mse_loss_fn(
            student_outputs.prediction_outputs, teacher_outputs.prediction_outputs
        )

        # 合并损失
        loss = student_outputs.loss \
                + self.alpha_hidden_states * loss_backbone \
                + self.beta_teacher_prediction * loss_prediction
        
        if return_outputs:
            return loss, student_outputs
        else:
            return loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        重写prediction_step确保正确提取prediction_outputs
        """
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss
            
            # 确保提取正确的预测值
            logits = outputs.prediction_outputs
        
        return (loss, logits, inputs["future_values"])
    

# compute_metrics
def compute_metrics(eval_pred):
    # 解包eval_pred
    predictions, labels = eval_pred
    # 在DistillationTrainer中，predictions实际上是student_outputs对象
    # 需要从中提取prediction_outputs
    if isinstance(predictions, dict) or hasattr(predictions, "prediction_outputs"):
        predictions = predictions.prediction_outputs
    if isinstance(predictions, tuple):
        predictions = predictions[0]
        
    # 计算自定义指标
    mse = ((predictions - labels) ** 2).mean().item()
    mae = abs(predictions - labels).mean().item()
    
    return {
        "eval_loss": mse,
        "mse": mse,
        "mae": mae,
    }


all_results = {
    "dataset": [],
    "sub_dataset": [],
}

# Loop over data
for DATASET in list_datasets:
    # Set batch size
    if DATASET == "traffic":
        BATCH_SIZE = 8
    elif DATASET == "electricity":
        BATCH_SIZE = 32
    else:
        BATCH_SIZE = 64 
        # BATCH_SIZE = 256 # sota 用的是256
    
    print(f"Model will be loaded from {MODEL_PATH}")
    SUBDIR = f"{OUT_DIR}/{DATASET}"

    zeroshot_model = get_model(
        model_path=MODEL_PATH, context_length=CONTEXT_LENGTH, prediction_length=FORECAST_LENGTH
    )
    
    teacher_learning_rate = None  # `None` value indicates that the optimal_lr_finder() will be used

    for fewshot_percent in [100]:
        all_results["dataset"].append(DATASET)
        all_results["sub_dataset"].append(f"{DATASET}_{fewshot_percent}")

        print("-" * 20, f"Running few-shot {fewshot_percent}%", "-" * 20)
        # Data prep: Get dataset
        dset_train, dset_val, dset_test = load_dataset(
            DATASET,
            CONTEXT_LENGTH,
            FORECAST_LENGTH,
            fewshot_fraction=fewshot_percent / 100,
            dataset_root_path=DATA_ROOT_PATH,
            use_frequency_token=args.enable_prefix_tuning,
            enable_padding=False,
        )
        dset_shift_test = ConceptDriftDataset(
            dset_test,
            quadratic_strength=0.005,  # 二次趋势漂移的强度
            anomaly_start_pos=200,     # 设置在context序列后期位置
            anomaly_length=10          # 突变持续10个时间步
        )



        # change head dropout to 0.7 for ett datasets
        if "ett" in DATASET:
            finetune_forecast_model = get_model(
                model_path=MODEL_PATH,
                context_length=CONTEXT_LENGTH,
                prediction_length=FORECAST_LENGTH,
                head_dropout=0.7,
            )
        else:
            finetune_forecast_model = get_model(
                model_path=MODEL_PATH,
                context_length=CONTEXT_LENGTH,
                prediction_length=FORECAST_LENGTH,
            )

        if FREEZE_BACKBONE:
            print(
                "Number of params before freezing backbone",
                count_parameters(finetune_forecast_model),
            )

            # Freeze the backbone of the model
            for param in finetune_forecast_model.backbone.parameters():
                param.requires_grad = False

            # Count params
            print(
                "Number of params after freezing the backbone",
                count_parameters(finetune_forecast_model),
            )

        if teacher_learning_rate is None:
            teacher_learning_rate, finetune_forecast_model = optimal_lr_finder(
                finetune_forecast_model,
                dset_train,
                batch_size=BATCH_SIZE,
                enable_prefix_tuning=args.enable_prefix_tuning,
            )
        print(f"Using learning rate = {teacher_learning_rate}")

        # This is to save space during exhaustive benchmarking, use specific directory if the saved models are needed
        tmp_dir = f"{OUT_DIR}/{DATASET}_pct{fewshot_percent}_TTM_teacher_train"

        finetune_forecast_args = TrainingArguments(
            output_dir=tmp_dir,
            overwrite_output_dir=True,
            learning_rate=teacher_learning_rate,
            num_train_epochs=EPOCHS,
            do_eval=True,
            evaluation_strategy="epoch",
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            dataloader_num_workers=NUM_WORKERS,
            report_to=["tensorboard"],
            save_strategy="epoch",
            logging_strategy="epoch",
            save_total_limit=1,
            logging_dir=tmp_dir,  # Make sure to specify a logging directory
            load_best_model_at_end=True,  # Load the best model when training ends
            metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
            greater_is_better=False,  # For loss
            label_names=["future_values"],
            seed=args.random_seed,
        )

        # Create the early stopping callback
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=10,  # Number of epochs with no improvement after which to stop
            early_stopping_threshold=0.0,  # Minimum improvement required to consider as improvement
        )
        tracking_callback = TrackingCallback()

        # Optimizer and scheduler
        optimizer = AdamW(finetune_forecast_model.parameters(), lr=teacher_learning_rate)
        scheduler = OneCycleLR(
            optimizer,
            teacher_learning_rate,
            epochs=EPOCHS,
            steps_per_epoch=math.ceil(len(dset_train) / (BATCH_SIZE)),
        )

        finetune_forecast_trainer = Trainer(
            model=finetune_forecast_model,
            args=finetune_forecast_args,
            train_dataset=dset_train,
            eval_dataset=dset_val,
            callbacks=[early_stopping_callback, tracking_callback],
            optimizers=(optimizer, scheduler),
        )
        finetune_forecast_trainer.remove_callback(INTEGRATION_TO_CALLBACK["codecarbon"])

        # Fine tune
        finetune_forecast_trainer.train()

        teacher_fewshot_output = finetune_forecast_trainer.evaluate(dset_test)
        shift_output = finetune_forecast_trainer.evaluate(dset_shift_test)


        if f"teacher_mse" not in all_results:
            all_results[f"teacher_mse"] = []
        if f"teacher_shift_mse" not in all_results:
            all_results[f"teacher_shift_mse"] = []
        all_results[f"teacher_mse"].append(teacher_fewshot_output["eval_loss"])
        all_results[f"teacher_shift_mse"].append(shift_output["eval_loss"])

        finetune_forecast_trainer.model.loss = "mae"  # 直接使用MAE损失
        teacher_fewshot_output = finetune_forecast_trainer.evaluate(dset_test)
        shift_output = finetune_forecast_trainer.evaluate(dset_shift_test)
        finetune_forecast_trainer.model.loss = "mse"  # 恢复为MSE损失

        if f"teacher_mae" not in all_results:
            all_results[f"teacher_mae"] = []
        if f"teacher_shift_mae" not in all_results:
            all_results[f"teacher_shift_mae"] = []
        all_results[f"teacher_mae"].append(teacher_fewshot_output["eval_loss"])
        all_results[f"teacher_shift_mae"].append(shift_output["eval_loss"])


        # Set learning rate
        student_learning_rate = teacher_learning_rate 
        # student_learning_rate = 0.0001  # `None` value indicates that the optimal_lr_finder() will be used

        for student_model_version in ["sync_patch_mlp"]:#["patch&cross_mlp","patch_mlp"]:
                
            # ++++++++++++++++第一次训练+++++++++++++++++++++++
            # -------------直接训练Student模型---------------------
            student_model_config = StudentModelConfig(finetune_forecast_model.config)
            student_model_config.student_model_version = student_model_version

            # 创建student模型
            student_model = StudentModel_PatchMLP(student_model_config)

            if student_learning_rate is None:
                student_learning_rate, student_model = optimal_lr_finder(
                    student_model,
                    dset_train,
                    batch_size=BATCH_SIZE,
                    enable_prefix_tuning=args.enable_prefix_tuning,
                )
            print(f"Using learning rate = {student_learning_rate}")

            # 将学生模型移动到与教师模型相同的设备上
            device = next(finetune_forecast_trainer.model.parameters()).device
            student_model = student_model.to(device)

            # 模型参数规模与MAC
            if fewshot_percent == 5:
                print(f"模型参数量: {sum(p.numel() for p in student_model.parameters()):,}")
                # 从 dset_train 中获取输入张量（这里假设 'past_values' 是模型的输入之一）
                input_tensor = dset_train[0]['past_values'].unsqueeze(0).to(device)

                # 使用 fvcore 计算模型的参数数量
                macs = FlopCountAnalysis(student_model, input_tensor)

                # 直接计算全部参数量
                params = sum(p.numel() for p in student_model.parameters())
                
                # 创建一个包含参数量和 MACs 的字典
                params_and_mac = {
                    "dataset": DATASET,
                    "student_model_version": student_model_version,
                    "num_params": params,
                    "macs": macs.total(),
                }

                # 将计算结果转换为 DataFrame
                params_and_mac_df = pd.DataFrame([params_and_mac])

                # 定义结果文件的路径
                output_csv_path = os.path.join(OUT_DIR, "model_params_macs_results.csv")

                # 检查文件是否已存在，如果存在就追加，否则新建
                if os.path.exists(output_csv_path):
                    df_out = pd.read_csv(output_csv_path)
                    df_out = pd.concat([df_out, params_and_mac_df], axis=0)  # 追加新的行
                else:
                    df_out = params_and_mac_df  # 如果文件不存在，则创建新的 DataFrame

                # 将计算的参数量和 MACs 保存到同一个 CSV 文件中
                df_out.to_csv(output_csv_path, index=False)

            tmp_dir = f"{OUT_DIR}/{DATASET}_pct{fewshot_percent}_onlystudent_{student_model_version}_lr{student_learning_rate}"

            student_forecast_args = TrainingArguments(
                output_dir=tmp_dir,
                overwrite_output_dir=True,
                learning_rate=student_learning_rate,
                lr_scheduler_type="constant",  # 设置为constant使学习率保持不变
                num_train_epochs=EPOCHS*4,
                do_eval=True,
                evaluation_strategy="epoch",
                per_device_train_batch_size=BATCH_SIZE,
                per_device_eval_batch_size=BATCH_SIZE,
                dataloader_num_workers=NUM_WORKERS,
                report_to=["tensorboard"],
                save_strategy="epoch",
                logging_strategy="epoch",
                save_total_limit=1,
                logging_dir=tmp_dir,  # Make sure to specify a logging directory
                load_best_model_at_end=True,  # Load the best model when training ends
                metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
                greater_is_better=False,  # For loss
                label_names=["future_values"],
                seed=args.random_seed,
            )

            # Create the early stopping callback
            early_stopping_callback = EarlyStoppingCallback(
                early_stopping_patience=10,  # Number of epochs with no improvement after which to stop
                early_stopping_threshold=0.0,  # Minimum improvement required to consider as improvement
            )
            tracking_callback = TrackingCallback()

            # Optimizer and scheduler
            optimizer = AdamW(student_model.parameters(), lr=student_learning_rate)
            scheduler = None
            # scheduler = OneCycleLR(
            #     optimizer,
            #     student_learning_rate,
            #     epochs=EPOCHS,
            #     steps_per_epoch=math.ceil(len(dset_train) / (BATCH_SIZE)),
            # )
            # scheduler = ReduceLROnPlateau(
            #     optimizer,
            #     mode='min',      # 监控指标是越小越好 (通常是 eval_loss)
            #     factor=0.5,      # 学习率衰减因子
            #     patience=3,     # 容忍多少个评估周期没有改善
            #     threshold=0.0001,# 改善的最小阈值
            #     verbose=False,    # 打印学习率更新信息
            #     min_lr=1e-7      # 学习率下限 (可选)
            # )

            student_forecast_trainer = Trainer(
                model=student_model,
                args=student_forecast_args,
                train_dataset=dset_train,
                eval_dataset=dset_val,
                callbacks=[early_stopping_callback, tracking_callback],
                optimizers=(optimizer, scheduler),
            )

            student_forecast_trainer.train()

            student_fewshot_output = student_forecast_trainer.evaluate(dset_test)
            shift_output = student_forecast_trainer.evaluate(dset_shift_test)

            if f"only_{student_model_version}_mse" not in all_results:
                all_results[f"only_{student_model_version}_mse"] = [] 
            all_results[f"only_{student_model_version}_mse"].append(student_fewshot_output["eval_loss"])
            if f"only_{student_model_version}_shift_mse" not in all_results:
                all_results[f"only_{student_model_version}_shift_mse"] = []
            all_results[f"only_{student_model_version}_shift_mse"].append(shift_output["eval_loss"])
            
            student_forecast_trainer.model.loss = "mae"  # 直接使用MAE损失
            student_fewshot_output = student_forecast_trainer.evaluate(dset_test)
            shift_output = student_forecast_trainer.evaluate(dset_shift_test)
            student_forecast_trainer.model.loss = "mse"  # 恢复为MSE损失
            
            if f"only_{student_model_version}_mae" not in all_results:
                all_results[f"only_{student_model_version}_mae"] = [] 
            # write results
            all_results[f"only_{student_model_version}_mae"].append(student_fewshot_output["eval_loss"])

            if f"only_{student_model_version}_shift_mae" not in all_results:
                all_results[f"only_{student_model_version}_shift_mae"] = []
            all_results[f"only_{student_model_version}_shift_mae"].append(shift_output["eval_loss"])

            # -----------------基于特征蒸馏Student模型---------------------
            # 添加student_model_version到student_model_config
            student_model_config = StudentModelConfig(finetune_forecast_model.config)
            student_model_config.student_model_version = student_model_version

            # 创建student模型
            student_model = StudentModel_PatchMLP(student_model_config)
            # 复用教师模型的head到学生模型
            # student_model.head = copy.deepcopy(finetune_forecast_trainer.model.head)
            student_model.head = finetune_forecast_trainer.model.head
            
            # 确保head中的参数不会被训练
            for param in student_model.head.parameters():
                param.requires_grad = False
            # 将学生模型移动到与教师模型相同的设备上
            device = next(finetune_forecast_trainer.model.parameters()).device
            student_model = student_model.to(device)
            
            tmp_dir = f"{OUT_DIR}/{DATASET}_pct{fewshot_percent}_student_{student_model_version}_lr{student_learning_rate}"

            distillation_forecast_args = TrainingArguments(
                output_dir=tmp_dir,
                overwrite_output_dir=True,
                learning_rate=student_learning_rate*2,
                lr_scheduler_type="constant",  # 设置为constant使学习率保持不变
                num_train_epochs=EPOCHS*100,
                do_eval=True,
                evaluation_strategy="epoch",
                per_device_train_batch_size=BATCH_SIZE,
                per_device_eval_batch_size=BATCH_SIZE,
                dataloader_num_workers=NUM_WORKERS,
                report_to=["tensorboard"],
                save_strategy="epoch",
                logging_strategy="epoch",
                save_total_limit=1,
                logging_dir=tmp_dir,  # Make sure to specify a logging directory
                load_best_model_at_end=True,  # Load the best model when training ends
                metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
                greater_is_better=False,  # For loss
                seed=args.random_seed,
            )

            # Create the early stopping callback
            early_stopping_callback = EarlyStoppingCallback(
                early_stopping_patience=10,  # Number of epochs with no improvement after which to stop
                early_stopping_threshold=0.0,  # Minimum improvement required to consider as improvement
            )
            tracking_callback = TrackingCallback()

            # Optimizer and scheduler
            optimizer = AdamW(student_model.parameters(), lr=student_learning_rate*2)
            scheduler = None
            # scheduler = OneCycleLR(
            #     optimizer,
            #     student_learning_rate,
            #     epochs=EPOCHS*4,
            #     steps_per_epoch=math.ceil(len(dset_train) / (BATCH_SIZE)),
            # )

            distillation_forecast_trainer = DistillationTrainerOnlyHidden(
                teacher_model=finetune_forecast_trainer.model,
                alpha_hidden_states=0,
                beta_teacher_prediction=0,
                model=student_model,
                args=distillation_forecast_args,
                train_dataset=dset_train,
                eval_dataset=dset_val,
                callbacks=[early_stopping_callback, tracking_callback],
                optimizers=(optimizer, scheduler),
            )

            # Fine tune
            distillation_forecast_trainer.train()

            # Evaluation
            print(
                "+" * 20,
                f"Test MSE after few-shot {fewshot_percent}% fine-tuning",
                "+" * 20,
            )
            # student_fewshot_output = distillation_forecast_trainer.evaluate(dset_test)

            distillation_forecast_trainer.model.device = next(distillation_forecast_trainer.model.parameters()).device
            
            # ------------------第二次训练----------------
            student_model_after_hidden_distillation = distillation_forecast_trainer.model

            # 重新设置student_model的head参与训练
            for param in student_model_after_hidden_distillation.head.parameters():
                param.requires_grad = True

            student_learning_rate2 = student_learning_rate * 0.01

            # Optimizer and scheduler
            optimizer = AdamW(student_model_after_hidden_distillation.parameters(), lr=student_learning_rate2)
            scheduler = None
            # scheduler = OneCycleLR(
            #     optimizer,
            #     student_learning_rate,
            #     epochs=EPOCHS*4,
            #     steps_per_epoch=math.ceil(len(dset_train) / (BATCH_SIZE)),
            # )

            early_stopping_callback = EarlyStoppingCallback(
                early_stopping_patience=10,  # Number of epochs with no improvement after which to stop
                early_stopping_threshold=0.0,  # Minimum improvement required to consider as improvement
            )

            distillation_forecast_after_hidden_distillation_args = TrainingArguments(
                output_dir=f"{OUT_DIR}/{DATASET}_pct{fewshot_percent}_student_{student_model_version}_lr{student_learning_rate2}_after_hidden_distillation",
                overwrite_output_dir=True,
                learning_rate=student_learning_rate2,
                lr_scheduler_type="constant",  # 设置为constant使学习率保持不变
                num_train_epochs=EPOCHS*10,
                do_eval=True,
                evaluation_strategy="epoch",
                per_device_train_batch_size=BATCH_SIZE,
                per_device_eval_batch_size=BATCH_SIZE,
                dataloader_num_workers=NUM_WORKERS,
                report_to=["tensorboard"],
                save_strategy="epoch",
                logging_strategy="epoch",
                save_total_limit=1,
                logging_dir=f"{OUT_DIR}/{DATASET}_pct{fewshot_percent}_student_{student_model_version}_lr{student_learning_rate2}_after_hidden_distillation",  # Make sure to specify a logging directory
                load_best_model_at_end=True,  # Load the best model when training ends
                metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
                greater_is_better=False,  # For loss
                seed=args.random_seed,
            )

            distillation_forecast_after_hidden_distillation_trainer = DistillationTrainerByLabel(
                teacher_model=finetune_forecast_trainer.model,
                alpha_hidden_states=0.01,
                beta_teacher_prediction=0.00,
                model=student_model_after_hidden_distillation,
                args=distillation_forecast_after_hidden_distillation_args,
                train_dataset=dset_train,
                eval_dataset=dset_val,
                callbacks=[early_stopping_callback, tracking_callback],
                optimizers=(optimizer, scheduler),
            )

            # Fine tune
            distillation_forecast_after_hidden_distillation_trainer.train()
            student_fewshot_output_after_hidden_distillation = distillation_forecast_after_hidden_distillation_trainer.evaluate(dset_test)
            shift_output = distillation_forecast_after_hidden_distillation_trainer.evaluate(dset_shift_test)

            # 记录实验结果
            if f"{student_model_version}_ByLabel_mse" not in all_results:
                all_results[f"{student_model_version}_ByLabel_mse"] = [] 
            all_results[f"{student_model_version}_ByLabel_mse"].append(student_fewshot_output_after_hidden_distillation["eval_loss"])
            if f"{student_model_version}_ByLabel_shift_mse" not in all_results:
                all_results[f"{student_model_version}_ByLabel_shift_mse"] = []
            all_results[f"{student_model_version}_ByLabel_shift_mse"].append(shift_output["eval_loss"])

            distillation_forecast_after_hidden_distillation_trainer.model.loss = "mae"  # 直接使用MAE损失
            student_fewshot_output_after_hidden_distillation = distillation_forecast_after_hidden_distillation_trainer.evaluate(dset_test)
            shift_output = distillation_forecast_after_hidden_distillation_trainer.evaluate(dset_shift_test)
            distillation_forecast_after_hidden_distillation_trainer.model.loss = "mse"  # 恢复为MSE损失

            if f"{student_model_version}_ByLabel_mae" not in all_results:
                all_results[f"{student_model_version}_ByLabel_mae"] = [] 
            # write results
            all_results[f"{student_model_version}_ByLabel_mae"].append(student_fewshot_output_after_hidden_distillation["eval_loss"])

            if f"{student_model_version}_ByLabel_shift_mae" not in all_results:
                all_results[f"{student_model_version}_ByLabel_shift_mae"] = []
            all_results[f"{student_model_version}_ByLabel_shift_mae"].append(shift_output["eval_loss"])

            
        df_out = pd.DataFrame(all_results).round(3)
        print(df_out)
        df_out.to_csv(f"{OUT_DIR}/results_zero_few.csv")
