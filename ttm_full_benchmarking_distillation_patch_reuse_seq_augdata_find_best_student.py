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

proxy_url = 'http://10.134.16.27:1078'
os.environ['HTTP_PROXY'] = proxy_url
os.environ['HTTPS_PROXY'] = proxy_url

import requests

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
OUT_DIR = args.save_dir # 'results-ttm-mlp-distillation-patch-patch-reuse-seq-augdata-SensitivityTests/TTM_cl-512_fl-96_pl-64_apl-3_ne-25_es-True'

MODEL_PATH = args.hf_model_path

# 参数敏感性
intermediate_hidden_loss_ratio_first_stage = 0.5
teacher_output_loss_ratio_first_stage = 0.5
intermediate_hidden_loss_ratio_second_stage = 0.3
teacher_output_loss_ratio_second_stage = 0.3

ExpRepeatTimes = 1

# ExpVersionList = ['both_stages', 'without_attn', 'without_aug', 'only_first_stage', 'only_second_stage']

# ExpVersionList = ['both_stages', 'with_real_attn','with_gru', 'without_aug', 'only_first_stage', 'only_second_stage', 'only_second_stage_with_random']

student_model_version_list = ['student_64dmodel_gru']
# student_model_version_list = ['student_256dmodel_gru','student_128dmodel_gru', 'student_64dmodel_gru', 'student_32dmodel_gru', 'student_16dmodel_gru']

list_datasets = [args.dataset]
'''
if args.datasets is None:
    list_datasets = [
        "etth1",
        "etth2",
        "ettm1",
        "ettm2",
        # "weather",
        # "electricity",
        # "traffic",
        # "exchange",
        "aqshunyi",
        "aqwan",
        "zafnoo",
        "czelan",
    ]

else:
    list_datasets = [dataset.strip() for dataset in args.datasets.split(",")]
'''


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
    def __init__(self, config, use_attn=True):
        super().__init__()

        self.scaler = TinyTimeMixerStdScaler(config)
        
        # patch相关参数
        self.patch_length = config.teacher_patch_length
        self.patch_stride = config.teacher_patch_stride
        self.num_patches = config.teacher_num_patches
        self.d_model = config.d_model
        # 模型版本
        self.student_model_version = getattr(config, "student_model_version", None)
        
        if self.student_model_version == 'student_current' :
            self.mlp = nn.Sequential(
                nn.Linear(self.patch_length, self.d_model),
                nn.LayerNorm(self.d_model, eps=config.layernorm_eps),
                nn.ReLU(),
                nn.Linear(self.d_model, self.d_model),
            )
            self.patch_attention_weights = nn.Parameter(torch.full((self.num_patches, self.num_patches), 0.0))
            self.ln = nn.LayerNorm(self.d_model, eps=config.layernorm_eps)
            self.ffn = nn.Linear(self.d_model, self.d_model)

        elif self.student_model_version == 'student_onlymlp':
            # 仅使用MLP
            self.mlp = nn.Sequential(
                nn.Linear(self.patch_length, self.d_model),
                nn.LayerNorm(self.d_model, eps=config.layernorm_eps),
                nn.ReLU(),
                nn.Linear(self.d_model, self.d_model),
            )

        elif self.student_model_version == 'student_256dmodel_gru':
            self.mlp = nn.Sequential(
                nn.Linear(self.patch_length, self.d_model*2),
                nn.LayerNorm(self.d_model*2, eps=config.layernorm_eps),
                nn.ReLU(),
                nn.Linear(self.d_model*2, self.d_model*2),
            )
            self.gru = nn.GRU(
                input_size=self.d_model*2,
                hidden_size=self.d_model*2,
                batch_first=True,
                bidirectional=True
            )
            self.ln = nn.LayerNorm(self.d_model*4, eps=config.layernorm_eps)
            self.ffn = nn.Linear(self.d_model*4, self.d_model)

        elif self.student_model_version == 'student_128dmodel_gru':
            self.mlp = nn.Sequential(
                nn.Linear(self.patch_length, self.d_model),
                nn.LayerNorm(self.d_model, eps=config.layernorm_eps),
                nn.ReLU(),
                nn.Linear(self.d_model, self.d_model),
            )
            self.gru = nn.GRU(
                input_size=self.d_model,
                hidden_size=self.d_model,
                batch_first=True,
                bidirectional=True
            )
            self.ln = nn.LayerNorm(self.d_model*2, eps=config.layernorm_eps)
            self.ffn = nn.Linear(self.d_model*2, self.d_model)

        elif self.student_model_version == 'student_64dmodel_gru':
            self.mlp = nn.Sequential(
                nn.Linear(self.patch_length, self.d_model//2),
                nn.LayerNorm(self.d_model//2, eps=config.layernorm_eps),
                nn.ReLU(),
                nn.Linear(self.d_model//2, self.d_model//2),
            )
            self.gru = nn.GRU(
                input_size=self.d_model//2,
                hidden_size=self.d_model//2,
                batch_first=True,
                bidirectional=True
            )
            self.ln = nn.LayerNorm(self.d_model, eps=config.layernorm_eps)
            self.ffn = nn.Linear(self.d_model, self.d_model)

        elif self.student_model_version == 'student_32dmodel_gru':
            self.mlp = nn.Sequential(
                nn.Linear(self.patch_length, self.d_model//4),
                nn.LayerNorm(self.d_model//4, eps=config.layernorm_eps),
                nn.ReLU(),
                nn.Linear(self.d_model//4, self.d_model//4),
            )
            self.gru = nn.GRU(
                input_size=self.d_model//4,
                hidden_size=self.d_model//4,
                batch_first=True,
                bidirectional=True
            )
            self.ln = nn.LayerNorm(self.d_model//2, eps=config.layernorm_eps)
            self.ffn = nn.Linear(self.d_model//2, self.d_model)
        elif self.student_model_version == 'student_16dmodel_gru':
            self.mlp = nn.Sequential(
                nn.Linear(self.patch_length, self.d_model//8),
                nn.LayerNorm(self.d_model//8, eps=config.layernorm_eps),
                nn.ReLU(),
                nn.Linear(self.d_model//8, self.d_model//8),
            )
            self.gru = nn.GRU(
                input_size=self.d_model//8,
                hidden_size=self.d_model//8,
                batch_first=True,
                bidirectional=True
            )
            self.ln = nn.LayerNorm(self.d_model//4, eps=config.layernorm_eps)
            self.ffn = nn.Linear(self.d_model//4, self.d_model)
        elif self.student_model_version == 'student_selfattn':
            self.mlp = nn.Sequential(
                nn.Linear(self.patch_length, self.d_model),
                nn.LayerNorm(self.d_model, eps=config.layernorm_eps),
                nn.ReLU(),
                nn.Linear(self.d_model, self.d_model),
            )
            self.ln = nn.LayerNorm(self.d_model, eps=config.layernorm_eps)
            self.ffn = nn.Linear(self.d_model, self.d_model)
        # 定义输出层
        self.head = ForecastingHead(
            head_nf=self.d_model * self.num_patches,
            forecast_horizon=config.prediction_length,
            head_dropout=0.0,
        )

        self.loss = config.loss
        self.teacher_num_patches = config.teacher_num_patches
        self.teacher_d_model = config.teacher_d_model

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
        scaled_past_values, loc, scale = self.scaler(past_values, past_observed_mask) 

        # 将输入展开为patch: [bs x num_patches x num_input_channels x patch_length]
        scaled_past_values = scaled_past_values.unfold(dimension=-2, size=self.patch_length, step=self.patch_stride)
        # 调整维度顺序: [bs x num_input_channels x num_patches x patch_length]
        scaled_past_values = scaled_past_values.transpose(-2, -3).contiguous()

        # 对每个patch进行编码
        patch_embeddings = self.mlp(scaled_past_values)  # Shape: [B, C, N, D]
        
        # 根据不同模型版本处理patch之间的依赖关系
        B, C, N, D = patch_embeddings.shape
        
        if self.student_model_version == 'student_current':
            # 使用patch attention + LayerNorm + FFN
            patch_embeddings_reshaped = patch_embeddings.contiguous().view(B * C, N, D)
            
            # 应用patch attention weights
            attention_weights = F.softmax(self.patch_attention_weights, dim=-1)  # [N, N]
            weighted_embeddings_reshaped = torch.matmul(attention_weights, patch_embeddings_reshaped)  # [B*C, N, D]
            
            # LayerNorm和FFN
            hidden_states = self.ln(patch_embeddings_reshaped + weighted_embeddings_reshaped)
            hidden_states = self.ffn(hidden_states).view(B, C, N, D)
            
        elif self.student_model_version == 'student_onlymlp':
            # 仅使用MLP，不做额外处理
            hidden_states = patch_embeddings
            
        elif self.student_model_version in ['student_256dmodel_gru','student_128dmodel_gru', 'student_64dmodel_gru', 'student_32dmodel_gru', 'student_16dmodel_gru']:
            # 使用GRU + LayerNorm + FFN
            patch_embeddings_reshaped = patch_embeddings.contiguous().view(B * C, N, D)
            
            # 应用GRU沿着patch维度
            gru_output, _ = self.gru(patch_embeddings_reshaped)  # [B*C, N, D*2]
            
            # LayerNorm和FFN
            hidden_states = self.ln(gru_output)
            hidden_states = self.ffn(hidden_states).view(B, C, N, self.d_model)
        
        elif self.student_model_version == 'student_selfattn':

            patch_embeddings_reshaped = patch_embeddings.contiguous().view(B * C, N, D)
            
            # 直接使用patch embeddings作为query, key, value
            query = patch_embeddings_reshaped
            key = patch_embeddings_reshaped
            value = patch_embeddings_reshaped
            
            self.scale = self.d_model ** -0.5

            # 计算attention scores
            attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale  # [B*C, N, N]
            attention_probs = F.softmax(attention_scores, dim=-1)
            
            # 应用attention
            weighted_embeddings_reshaped = torch.matmul(attention_probs, value)  # [B*C, N, D]
            hidden_states = self.ln(patch_embeddings_reshaped + weighted_embeddings_reshaped)
            hidden_states = self.ffn(hidden_states).view(B, C, N, D)

        else:
            # 默认情况，仅使用patch embeddings
            hidden_states = patch_embeddings
            
        # 完成对教师的映射
        hidden_states_to_teacher = hidden_states  # [B, C, N, D]

        # 预测未来值
        y_hat = self.head(hidden_states, past_values=past_values, future_values=None)
        y_hat = y_hat * scale + loc  # [B, prediction_length, num_input_channels]

        if future_values is None:
            return y_hat
        
        # 计算损失
        loss_val = loss(y_hat, future_values)

        # 返回模型输出
        return ModelOutput(
            loss=loss_val,
            prediction_outputs=y_hat,
            hidden_states_to_teacher=hidden_states_to_teacher,
        )


# 定义 DistillationTrainer 类
class DistillationTrainerOnlyHidden(Trainer):
    def __init__(self, teacher_model, 
                 intermediate_hidden_loss_ratio=0.0, 
                 teacher_output_loss_ratio=0.0,
                 use_data_aug = True, 
                 *args, **kwargs):
        
        # 初始化父类，self.model 为学生模型
        super().__init__(*args, **kwargs)
        # 保存教师模型
        self.teacher_model = teacher_model
        
        # 设置教师模型为评估模式，确保不更新参数
        self.teacher_model.eval()
        # 设置损失权重
        self.intermediate_hidden_loss_ratio = intermediate_hidden_loss_ratio  # backbone_hidden_state 的损失权重
        self.teacher_output_loss_ratio = teacher_output_loss_ratio    # prediction_outputs 的损失权重

        # 消融实验-是否使用数据增强
        self.use_data_aug = use_data_aug 


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
        
        # loss仅仅使用表征部分的差异
        # loss = loss_backbone

        # 计算 prediction_outputs 部分的损失
        loss_prediction = mse_loss_fn(
            student_outputs.prediction_outputs, teacher_outputs.prediction_outputs
        )

        loss = self.intermediate_hidden_loss_ratio * loss_backbone + self.teacher_output_loss_ratio * loss_prediction
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
    def __init__(self, teacher_model, intermediate_hidden_loss_ratio=0.01, teacher_output_loss_ratio=0.0, *args, **kwargs):
        # 初始化父类，self.model 为学生模型
        super().__init__(*args, **kwargs)
        # 保存教师模型
        self.teacher_model = teacher_model
        
        # 设置教师模型为评估模式，确保不更新参数
        self.teacher_model.eval()
        # 设置损失权重
        self.intermediate_hidden_loss_ratio = intermediate_hidden_loss_ratio  # backbone_hidden_state 的损失权重
        self.teacher_output_loss_ratio = teacher_output_loss_ratio    # prediction_outputs 的损失权重

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
                + self.intermediate_hidden_loss_ratio * loss_backbone \
                + self.teacher_output_loss_ratio * loss_prediction
        
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
    

class DistillationTrainerByLabel_withRandom(Trainer):
    def __init__(self, teacher_model, intermediate_hidden_loss_ratio=0.01, teacher_output_loss_ratio=0.0, *args, **kwargs):
        # 初始化父类，self.model 为学生模型
        super().__init__(*args, **kwargs)
        # 保存教师模型
        self.teacher_model = teacher_model
        
        # 设置教师模型为评估模式，确保不更新参数
        self.teacher_model.eval()
        # 设置损失权重
        self.intermediate_hidden_loss_ratio = intermediate_hidden_loss_ratio  # backbone_hidden_state 的损失权重
        self.teacher_output_loss_ratio = teacher_output_loss_ratio    # prediction_outputs 的损失权重

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

        if self.state.epoch % 2 == 0:

            # 保存原始past_values的形状和设备信息
            original_shape = inputs['past_values'].shape
            device = inputs['past_values'].device
            
            uniform_noise = 2 * torch.rand(
                size=original_shape,
                device=device
            ) - 1
            # 暂存原始数据
            original_past_values = inputs['past_values']
            # 替换为高斯噪声
            inputs['past_values'] = uniform_noise
                


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

        if self.state.epoch % 2 == 0:
            # 合并损失
            loss = self.intermediate_hidden_loss_ratio * loss_backbone \
                    + self.teacher_output_loss_ratio * loss_prediction
        else:
            # 合并损失
            loss = student_outputs.loss \
                    + self.intermediate_hidden_loss_ratio * loss_backbone \
                    + self.teacher_output_loss_ratio * loss_prediction
        
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
    elif DATASET in ["aqshunyi","aqwan","zafnoo","czelan"]:
        BATCH_SIZE = 16 * 11
    else:
        BATCH_SIZE = 64

    print(f"Model will be loaded from {MODEL_PATH}")
    SUBDIR = f"{OUT_DIR}/{DATASET}"

    zeroshot_model = get_model(
        model_path=MODEL_PATH, context_length=CONTEXT_LENGTH, prediction_length=FORECAST_LENGTH
    )
    
    teacher_learning_rate = None  # `None` value indicates that the optimal_lr_finder() will be used

    fewshot_percent = 100
    # for fewshot_percent in [100]:
    all_results["dataset"].append(DATASET)
    all_results["sub_dataset"].append(f"{DATASET}_{fewshot_percent}")

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
        quadratic_strength=0.000,  # 二次趋势漂移的强度
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


    for student_model_version in student_model_version_list:
    
        # 用于存储重复实验结果的列表
        student_mse_results = []
        student_mae_results = []
        student_shift_mse_results = []
        student_shift_mae_results = []
        # ++++++++++++++++第一次训练+++++++++++++++++++++++
        # -------------直接训练Student模型---------------------
        for repeat_idx in range(ExpRepeatTimes):

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

            tmp_dir = f"{OUT_DIR}/{DATASET}_pct{fewshot_percent}_onlystudent_{student_model_version}"

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

            # 保存当前重复的结果
            student_mse_results.append(student_fewshot_output["eval_loss"])
            student_shift_mse_results.append(shift_output["eval_loss"])
            
            student_forecast_trainer.model.loss = "mae"  # 直接使用MAE损失
            student_fewshot_output = student_forecast_trainer.evaluate(dset_test)
            shift_output = student_forecast_trainer.evaluate(dset_shift_test)
            student_forecast_trainer.model.loss = "mse"  # 恢复为MSE损失
            
            student_mae_results.append(student_fewshot_output["eval_loss"])
            student_shift_mae_results.append(shift_output["eval_loss"])

            # 释放内存
            del student_model
            del student_forecast_trainer
            gc.collect()
            torch.cuda.empty_cache()

        # 计算均值和标准差
        mse_mean = np.mean(student_mse_results)
        mse_std = np.std(student_mse_results)
        mae_mean = np.mean(student_mae_results)
        mae_std = np.std(student_mae_results)
        shift_mse_mean = np.mean(student_shift_mse_results)
        shift_mse_std = np.std(student_shift_mse_results)
        shift_mae_mean = np.mean(student_shift_mae_results)
        shift_mae_std = np.std(student_shift_mae_results)
        # 记录均值和标准差
        if f"only_{student_model_version}_mse" not in all_results:
            all_results[f"only_{student_model_version}_mse"] = []
            all_results[f"only_{student_model_version}_mse_std"] = []
        all_results[f"only_{student_model_version}_mse"].append(mse_mean)
        all_results[f"only_{student_model_version}_mse_std"].append(mse_std)
        
        # if f"only_{student_model_version}_shift_mse" not in all_results:
        #     all_results[f"only_{student_model_version}_shift_mse"] = []
        #     all_results[f"only_{student_model_version}_shift_mse_std"] = []
        # all_results[f"only_{student_model_version}_shift_mse"].append(shift_mse_mean)
        # all_results[f"only_{student_model_version}_shift_mse_std"].append(shift_mse_std)
        
        if f"only_{student_model_version}_mae" not in all_results:
            all_results[f"only_{student_model_version}_mae"] = []
            all_results[f"only_{student_model_version}_mae_std"] = [] 
        all_results[f"only_{student_model_version}_mae"].append(mae_mean)
        all_results[f"only_{student_model_version}_mae_std"].append(mae_std)
        
        # if f"only_{student_model_version}_shift_mae" not in all_results:
        #     all_results[f"only_{student_model_version}_shift_mae"] = []
        #     all_results[f"only_{student_model_version}_shift_mae_std"] = []
        # all_results[f"only_{student_model_version}_shift_mae"].append(shift_mae_mean)
        # all_results[f"only_{student_model_version}_shift_mae_std"].append(shift_mae_std)

        # -----------------基于特征蒸馏Student模型---------------------
        # 添加student_model_version到student_model_config

        # 用于存储重复实验结果的列表
        distill_mse_results = []
        distill_mae_results = []
        distill_shift_mse_results = []
        distill_shift_mae_results = []

        result_key = (
            f"Distil-{student_model_version}"
        )

        for repeat_idx in range(ExpRepeatTimes):
            student_model_config = StudentModelConfig(finetune_forecast_model.config)
            student_model_config.student_model_version = student_model_version

            # 创建student模型
            student_model = StudentModel_PatchMLP(student_model_config)
            # 复用教师模型的head到学生模型
            student_model.head = copy.deepcopy(finetune_forecast_trainer.model.head)
            
            # 确保head中的参数不会被训练
            for param in student_model.head.parameters():
                param.requires_grad = False

            # 将学生模型移动到与教师模型相同的设备上
            device = next(finetune_forecast_trainer.model.parameters()).device
            student_model = student_model.to(device)
            
            tmp_dir = f"{OUT_DIR}/{DATASET}_pct{fewshot_percent}_student_{student_model_version}"

            # 第一阶段训练参数
            distillation_forecast_args = TrainingArguments(
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
                seed=args.random_seed+repeat_idx,
            )

            # 早停回调
            early_stopping_callback = EarlyStoppingCallback(
                early_stopping_patience=10,  # Number of epochs with no improvement after which to stop
                early_stopping_threshold=0.0,  # Minimum improvement required to consider as improvement
            )
            tracking_callback = TrackingCallback()

            # 优化器
            optimizer = AdamW(student_model.parameters(), lr=student_learning_rate)
            scheduler = None

            distillation_forecast_trainer = DistillationTrainerOnlyHidden(
                teacher_model=finetune_forecast_trainer.model,
                intermediate_hidden_loss_ratio=intermediate_hidden_loss_ratio_first_stage,
                teacher_output_loss_ratio=teacher_output_loss_ratio_first_stage,
                use_data_aug=True,
                model=student_model,
                args=distillation_forecast_args,
                train_dataset=dset_train,
                eval_dataset=dset_val,
                callbacks=[early_stopping_callback, tracking_callback],
                optimizers=(optimizer, scheduler),
            )
            
            # 训练
            distillation_forecast_trainer.train()

            distillation_forecast_trainer.model.device = next(distillation_forecast_trainer.model.parameters()).device
            
            # ------------------第二次训练----------------
            # 获取蒸馏后的学生模型
            student_model_after_hidden_distillation = distillation_forecast_trainer.model

            # 重新设置student_model的head参与训练
            for param in student_model_after_hidden_distillation.head.parameters():
                param.requires_grad = True

            student_learning_rate2 = student_learning_rate * 0.01

            # 优化器
            optimizer = AdamW(student_model_after_hidden_distillation.parameters(), lr=student_learning_rate2)
            scheduler = None

            early_stopping_callback = EarlyStoppingCallback(
                early_stopping_patience=10,  # Number of epochs with no improvement after which to stop
                early_stopping_threshold=0.0,  # Minimum improvement required to consider as improvement
            )

            tmp_dir = f"{OUT_DIR}/{DATASET}_pct{fewshot_percent}_student_{student_model_version}_after_hidden_distillation"
            
            # 第二阶段训练参数
            distillation_forecast_after_hidden_distillation_args = TrainingArguments(
                output_dir=tmp_dir,
                overwrite_output_dir=True,
                learning_rate=student_learning_rate2,
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
                seed=args.random_seed+repeat_idx,
            )

            distillation_forecast_after_hidden_distillation_trainer = DistillationTrainerByLabel(
                teacher_model=finetune_forecast_trainer.model,
                intermediate_hidden_loss_ratio=intermediate_hidden_loss_ratio_second_stage,
                teacher_output_loss_ratio=teacher_output_loss_ratio_second_stage,
                model=student_model_after_hidden_distillation,
                args=distillation_forecast_after_hidden_distillation_args,
                train_dataset=dset_train,
                eval_dataset=dset_val,
                callbacks=[early_stopping_callback, tracking_callback],
                optimizers=(optimizer, scheduler),
            )

            # 训练
            distillation_forecast_after_hidden_distillation_trainer.train()

            # 评估 MSE
            student_fewshot_output = distillation_forecast_after_hidden_distillation_trainer.evaluate(dset_test)
            shift_output = distillation_forecast_after_hidden_distillation_trainer.evaluate(dset_shift_test)
            
            # 收集当前实验指标
            distill_mse_results.append(student_fewshot_output["eval_loss"])
            distill_shift_mse_results.append(shift_output["eval_loss"])
            
            # 评估 MAE
            distillation_forecast_after_hidden_distillation_trainer.model.loss = "mae"
            student_fewshot_output = distillation_forecast_after_hidden_distillation_trainer.evaluate(dset_test)
            shift_output = distillation_forecast_after_hidden_distillation_trainer.evaluate(dset_shift_test)
            distillation_forecast_after_hidden_distillation_trainer.model.loss = "mse"
            
            distill_mae_results.append(student_fewshot_output["eval_loss"])
            distill_shift_mae_results.append(shift_output["eval_loss"])
            
            # 释放内存
            del student_model, student_model_after_hidden_distillation
            del distillation_forecast_trainer, distillation_forecast_after_hidden_distillation_trainer
            gc.collect()
            torch.cuda.empty_cache()

        # 计算均值和标准差
        mse_mean = np.mean(distill_mse_results)
        mse_std = np.std(distill_mse_results)
        mae_mean = np.mean(distill_mae_results)
        mae_std = np.std(distill_mae_results)
        shift_mse_mean = np.mean(distill_shift_mse_results)
        shift_mse_std = np.std(distill_shift_mse_results)
        shift_mae_mean = np.mean(distill_shift_mae_results)
        shift_mae_std = np.std(distill_shift_mae_results)


        # 保存均值和标准差
        # MSE
        if f"{result_key}_mse" not in all_results:
            all_results[f"{result_key}_mse"] = []
            all_results[f"{result_key}_mse_std"] = []
        all_results[f"{result_key}_mse"].append(mse_mean)
        all_results[f"{result_key}_mse_std"].append(mse_std)
        
        # Shift MSE
        # if f"{result_key}_shift_mse" not in all_results:
        #     all_results[f"{result_key}_shift_mse"] = []
        #     all_results[f"{result_key}_shift_mse_std"] = []
        # all_results[f"{result_key}_shift_mse"].append(shift_mse_mean)
        # all_results[f"{result_key}_shift_mse_std"].append(shift_mse_std)
        
        # MAE
        if f"{result_key}_mae" not in all_results:
            all_results[f"{result_key}_mae"] = []
            all_results[f"{result_key}_mae_std"] = [] 
        all_results[f"{result_key}_mae"].append(mae_mean)
        all_results[f"{result_key}_mae_std"].append(mae_std)
        
        # Shift MAE
        # if f"{result_key}_shift_mae" not in all_results:
        #     all_results[f"{result_key}_shift_mae"] = []
        #     all_results[f"{result_key}_shift_mae_std"] = []
        # all_results[f"{result_key}_shift_mae"].append(shift_mae_mean)
        # all_results[f"{result_key}_shift_mae_std"].append(shift_mae_std)


    # df_out = pd.DataFrame(all_results).round(3)
    # print(df_out)
    # df_out.to_csv(f"{OUT_DIR}/results_zero_few.csv")

    results_file_path = f"{OUT_DIR}/results_zero_few.csv"
    df_out = pd.DataFrame(all_results).round(3)

    if os.path.exists(results_file_path):
        existing_df = pd.read_csv(results_file_path)
        combined_df = pd.concat([existing_df, df_out], ignore_index=True)
        combined_df.to_csv(results_file_path, index=False)
        print(f"Appended {len(df_out)} new rows to existing results file: {results_file_path}")
    else:
        df_out.to_csv(results_file_path, index=False)
        print(f"Created new results file with {len(df_out)} entries: {results_file_path}")


response = requests.get(f'https://api.day.app/wwJ2955zQwsZuY9hcf6nEk/{DATASET}-findbeststudent-{mse_mean}')
