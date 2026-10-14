from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()   # 创建计数器，依次产生 0、1、2、3……

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)    # next() 是 Python 内置函数，用来从迭代器中取出下一个值，每调用一次就向前走一步
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)        # 完整 token 列表：输入 + 已生成的输出
        self.last_token = token_ids[-1]         # 当前最后一个 token，逐 token 生成时会用到
        self.num_tokens = len(self.token_ids)   # 当前 token 总数                     这里处理的输入是list，与下面第33行定义的那个函数无关，这里的len是python内置的函数
        self.num_prompt_tokens = len(token_ids) # 原始输入的 token 数                 
        self.num_cached_tokens = 0              # 前面多少个 token 已经有可用的 KV 缓存
        self.num_scheduled_tokens = 0           # 本轮准备交给模型处理的 token 数
        self.is_prefill = True                  # 是否处于处理输入、建立缓存的阶段
        self.block_table = []                   # 这条序列使用的物理 KV 缓存块编号
        self.temperature = sampling_params.temperature  #
        self.max_tokens = sampling_params.max_tokens    #
        self.ignore_eos = sampling_params.ignore_eos    # 这几个都是sampling_params里面定义的参数

    def __len__(self):                          # 这里处理的输入是 Sequence 对象。这里是在规定：当别人对 Sequence 对象调用 len() 时，应该返回什么
        return self.num_tokens

    def __getitem__(self, key):                 # key 就是写在 seq[...] 方括号里的内容，表示要取哪个位置，或者哪一段
        return self.token_ids[key]

    @property                                   # @property 是 Python 的一个装饰器，作用是：让方法可以像属性一样读取，不用加 ()
    def is_finished(self):                      # 例如，使用时写：seq.is_finished，Python 就会自动执行这个方法
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]  # 取值是左闭右开

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size  # 向上取整

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
