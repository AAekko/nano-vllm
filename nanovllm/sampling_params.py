from dataclasses import dataclass


@dataclass(slots=True)            # dataclass 自动生成常用方法
class SamplingParams:
    temperature: float = 1.0      # 控制选择下一个 token 时的随机程度
    max_tokens: int = 64          # 最大输出 token 数
    ignore_eos: bool = False      # 是否忽略 EOS

    def __post_init__(self):      # __post_init__ 会在自动生成的初始化方法完成后调用
        assert self.temperature > 1e-10, "greedy sampling is not permitted"     # 这个接口不支持用零温度来请求贪心解码，同时也排除了负温度和非常小的温
