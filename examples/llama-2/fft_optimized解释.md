以下是对每行代码的注释：
```yaml
base_model: NousResearch/Llama - 2 - 7b - hf  # 基础模型的路径或名称
model_type: LlamaForCausalLM  # 模型类型为因果语言模型
tokenizer_type: LlamaTokenizer  # 分词器类型

load_in_8bit: false  # 是否以 8 位加载模型
load_in_4bit: false  # 是否以 4 位加载模型
strict: false  # 是否严格模式，加载模型时，检查权重与模型架构是否匹配、代码声明的模型结构与base_model 架构是否匹配等

datasets:  # 数据集配置开始
  - path: mhenrichsen/alpaca_2k_test  # 数据集路径
    type: alpaca  # 数据集类型为 alpaca
dataset_prepared_path: last_run_prepared  # 已处理数据集的保存路径
val_set_size: 0.05  # 验证集大小占比
output_dir: ./outputs/out  # 输出目录

sequence_len: 4096  # 序列长度
sample_packing: true  # 是否进行样本打包，数据预处理会进行样本打包，将多个样本合成一个批次，提高训练效率
pad_to_sequence_len: true  # 是否填充到序列长度，进行批处理，需要将样本长度统一，所以需要填充长短不一的样本

adapter:  # 适配器相关配置
lora_model_dir:  # LoRA 模型目录
lora_r:  # LoRA 模型的某个参数 r
lora_alpha:  # LoRA 模型的某个参数 alpha
lora_dropout:  # LoRA 模型的 dropout 值
lora_target_linear:  # LoRA 模型的目标线性层
lora_fan_in_fan_out:  # LoRA 模型的 fan_in_fan_out 配置

wandb_project:  # wandb 项目名称
wandb_entity:  # wandb 实体
wandb_watch:  # wandb 监控相关
wandb_name:  # wandb 运行名称
wandb_log_model:  # 是否在 wandb 中记录模型

gradient_accumulation_steps: 1  # 梯度累积步数
micro_batch_size: 1  # 小批量大小
num_epochs: 1  # 训练的轮数
optimizer: adamw_bnb_8bit  # 优化器类型
lr_scheduler: cosine  # 学习率调度器类型为余弦调度
learning_rate: 0.0002  # 学习率

train_on_inputs: false  # 是否在输入上进行训练
group_by_length: false  # 是否按长度分组
bf16: auto  # bfloat16 数据格式设置为自动
fp16:  # 半精度浮点数相关配置（这里为空）
tf32: false  # 是否使用 tf32

# 此配置项指明是否启用梯度检查点技术
# 梯度检查点技术可以通过在训练过程中仅保留部分中间结果来减少内存占用，从而在一定程度上提高训练大模型的可能性
gradient_checkpointing: true

# 此配置项用于设置早停策略的耐心值
# 早停策略是一种避免过拟合的方法，当模型在验证集上的表现连续若干次没有提升时，训练过程将提前终止
# 耐心值即为允许验证集性能不提升的最大检查次数
early_stopping_patience:
resume_from_checkpoint:  # 从检查点恢复训练的路径
local_rank:  # 本地进程的等级（多卡训练相关）
logging_steps: 1  # 记录日志的步数
xformers_attention:  # xformers 注意力相关配置
flash_attention: true  # 是否使用 flash 注意力机制
flash_attn_cross_entropy: false  # flash 注意力机制的交叉熵相关配置
flash_attn_rms_norm: true  # flash 注意力机制的均方根归一化相关配置
flash_attn_fuse_qkv: false  # flash 注意力机制融合 QKV 相关配置
flash_attn_fuse_mlp: true  # flash 注意力机制融合 MLP 相关配置

warmup_steps: 100  # 热身步数
evals_per_epoch: 4  # 每个训练轮次的评估次数
eval_table_size:  # 评估表的大小
saves_per_epoch: 1  # 每个训练轮次的保存次数
debug:  # 调试相关配置
deepspeed: #deepspeed_configs/zero2.json # 多 GPU 训练的 deepspeed 配置（这里是配置文件路径）
# weight_decay=0.1 表示我们将使用 0.1 作为 L2 正则化的系数。这意味着在每次梯度下降更新时，模型参数都会被缩小一定的比例，具体来说，每个权重 ( w ) 在更新之前
# 都会先减去 ( 0.1 \times w )，这相当于在损失函数上加了一个 ( 0.1 \times |w|^2 ) 的惩罚项。
weight_decay: 0.1  # 权重衰减
fsdp:  # 完全分片数据并行相关配置
fsdp_config:  # fsdp 的配置
special_tokens:  # 特殊标记相关配置
```



### wandb log替代 tensorboard，进行 精度 等的记录

以下是一个使用`wandb`（Weights & Biases）的配置示例：

```python
import wandb

# 初始化 wandb
wandb.init(project="my-project-name", entity="my-entity-name")

# 设置配置参数
config = wandb.config
config.learning_rate = 0.001
config.batch_size = 32
config.num_epochs = 10

# 在训练过程中可以随时记录其他指标
wandb.log({"training_loss": 0.5})

# 训练结束后可以记录最终的评估指标
wandb.log({"final_accuracy": 0.85})
```

解释：

1. **初始化 wandb**：
   - `wandb.init(project="my-project-name", entity="my-entity-name")`：这一步初始化`wandb`并将当前的运行与一个特定的项目（project）和实体（entity）关联起来。项目通常是为了组织相关的实验，实体可以是个人用户、团队或组织。这样可以方便地在`wandb`的界面上查看和比较不同的实验运行。

2. **设置配置参数**：
   - `config = wandb.config`：创建一个配置对象，用于存储实验的参数设置。
   - `config.learning_rate = 0.001`等：在这里可以设置各种实验参数，如学习率、批次大小、训练轮数等。这些参数会在`wandb`的界面上显示，方便后续分析不同参数设置对实验结果的影响。

3. **记录指标**：
   - `wandb.log({"training_loss": 0.5})`：在训练过程中，可以随时使用`wandb.log()`方法记录各种指标，比如训练损失、准确率等。这样可以实时跟踪实验的进展，并在`wandb`的界面上以图表的形式展示这些指标随时间的变化。
   - `wandb.log({"final_accuracy": 0.85})`：在训练结束后，可以记录最终的评估指标，以便对实验结果进行总结和比较。

使用`wandb`可以方便地跟踪和比较不同实验的参数设置和结果，同时它还提供了很多其他功能，如版本控制、协作、模型可视化等，对于进行机器学习和深度学习实验非常有帮助。


### flash attention配置项

上述配置项中的各个参数可以根据不同的应用场景和需求进行设置。以下是每个配置项的详细解释及何时设置为 true 或 false：
1. flash_attention: true
含义：是否启用 Flash Attention 机制。
设置建议：
true：当你的模型需要更快的注意力计算速度，并且内存使用效率较高时启用。
false：如果你的硬件不支持 Flash Attention 或者模型对精度有更高的要求时禁用。
2. flash_attn_cross_entropy: false
含义：Flash Attention 机制在计算交叉熵损失时的相关配置。
设置建议：
true：如果你的模型使用了交叉熵损失函数，并且希望在计算注意力时优化这部分操作时启用。
false：默认情况下禁用，除非明确需要优化交叉熵损失计算。
3. flash_attn_rms_norm: true
含义：Flash Attention 机制在使用 RMSNorm（均方根归一化）时的相关配置。
设置建议：
true：如果你的模型使用了 RMSNorm 并且希望在注意力机制中集成这一层时启用。
false：如果你的模型没有使用 RMSNorm 或者不需要在注意力机制中集成时禁用。
4. flash_attn_fuse_qkv: false
含义：Flash Attention 机制在融合 Query (Q)、Key (K) 和 Value (V) 时的相关配置。
设置建议：
true：如果你的模型需要在注意力计算过程中融合 QKV 以提高计算效率时启用。
false：如果你的模型不需要融合 QKV 或者已经有其他机制处理时禁用。
5. flash_attn_fuse_mlp: true
含义：Flash Attention 机制在融合 MLP（多层感知器）时的相关配置。
设置建议：
true：如果你的模型需要在注意力计算过程中融合 MLP 层以提高整体性能时启用。
false：如果你的模型不需要融合 MLP 或者已经有其他机制处理时禁用。
总结
启用 Flash Attention (flash_attention: true)：通常用于提高模型的计算速度和内存效率。
启用特定优化：根据模型的具体需求选择启用或禁用相关配置项。
这些配置项的选择应基于具体的模型架构和性能需求来进行调整。如果不确定如何设置，建议先尝试默认配置，然后根据实际效果进行调整。
