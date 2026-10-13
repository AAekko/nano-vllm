import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)                       # @dataclass 装饰器会自动为你生成 __init__ 方法，负责接收参数并将它们赋值给 self
class Config:
    model: str                               # 本地模型路径，里面包含模型配置config.json文件和权重model.safetensors文件等
    max_num_batched_tokens: int = 16384      # 一次批量处理的 token 预算
    max_num_seqs: int = 512                  # 一次批量处理的最大请求数量
    max_model_len: int = 4096                # 单条序列的最大长度
    gpu_memory_utilization: float = 0.9      # 显存预算比例
    tensor_parallel_size: int = 1            # 张量并行使用的 GPU 数量
    enforce_eager: bool = False              # 是否强制普通执行模式；为 False 时，模型执行器可使用 CUDA Graph 加速
    hf_config: AutoConfig | None = None      # hf 表示这里使用了 Hugging Face 的配置体系和读取工具，让 AutoConfig 读取模型路径文件夹下的config.json。注意这段代码读取的是本地模型目录，没有另外加载一个 Hugging Face 模型
    eos: int = -1                            # 表示生成结束的 token ID
    kvcache_block_size: int = 256            # 每个缓存块容纳多少个 token
    num_kvcache_blocks: int = -1             # 整个缓存池有多少个块，-1 表示尚未确定

    def __post_init__(self):                 # 在自动生成的 __init__ 方法执行完毕（所有字段都赋值完成）之后，Python 会自动调用 __post_init__ 方法
        assert os.path.isdir(self.model)     # 检查模型路径是不是一个存在的目录
        assert self.kvcache_block_size % 256 == 0  # 检查缓存块大小是否为 256 的倍数
        assert 1 <= self.tensor_parallel_size <= 8 # 检查张量并行数量是否在 1～8 之间
        self.hf_config = AutoConfig.from_pretrained(self.model)  # 读取模型自身的结构配置，例如层数、隐藏维度、注意力头数、最大位置数量等。这一步读取的是配置，模型权重由后面的模型执行器加载！！
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)  # 把长度上限设为“用户设置”和“模型声明的上限”中较小的那个。例如，你设置 8192，模型只声明支持 4096，最终配置就是 4096
