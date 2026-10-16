import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:                        # LLMEngine 这个类负责组织整个推理流程

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)    # AutoTokenizer.from_pretrained() 是根据模型目录中的分词器配置、词表等文件，创建具体的 tokenizer 对象；use_fast=True 表示优先使用快速实现
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)              # tokenizer.encode() 来自 Hugging Face 的 transformers 库，作用是把文本转换成模型可以接收的 token ID 列表。本项目在 第32行 创建分词器，注意这是个具体的 tokenizer 对象
        seq = Sequence(prompt, sampling_params)                 # 创建 Sequence 类，实例化为seq，保存 token 和采样参数                              # 再次注意，python中将类实例化就是长这样，区别于C++
        self.scheduler.add(seq)                                 # 加入scheduler.waiting

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,                                                    # 在example.py中llm.generate()用到了generate函数，所以self表示当前的LLM实例，通过它访问 tokenizer、scheduler 等成员
        prompts: list[str] | list[list[int]],                    # | 在类型注解中表示“或者”，这里表示 prompts 可以是：字符串列表 或 token ID 列表的列表。比如，两种 prompts 输入形式分别是：prompts = ["你好", "介绍一下自己"]，prompts = [[101, 102], [201, 202, 203]]
        sampling_params: SamplingParams | list[SamplingParams],  # 这里表示可以是：一个采样参数对象：所有 prompt 共用  或  采样参数对象列表：每个 prompt 分别使用对应的参数
        use_tqdm: bool = True,                                   # use_tqdm 是作者起的参数名，意思是“是否使用 tqdm”。tqdm 是 Python 中用来显示进度条的库，这个项目用它显示已完成的请求数量和生成速度，具体实现在函数体里面
    ) -> list[str]:             # 这里表示函数返回一个列表，列表中的每个元素都是字符串，例如：["你好！", "100 以内的质数有 2、3、5、7……"]。  Tip：-> list[str] 只是类型标注，不会强制转换实际返回值。 注！！这里的返回值注释与实际实现不一致，最后返回的是字典列表，每个字典包含 text 和 token_ids
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)  # tqdm 是 Python 中用来显示进度条的库，pbar 保存这个进度条对象，后面会更新它。其中()中的四个依次表示：总任务数等于输入请求数，进度条前面显示 Generating，随终端宽度调整进度条宽度，use_tqdm=False 时禁用显示
        if not isinstance(sampling_params, list):                # isinstance() 是 Python 内置函数，用来判断一个对象是不是某种类型，返回 True 或 False，用法：isinstance(对象, 类型)
            sampling_params = [sampling_params] * len(prompts)   # 如过传入的sampling_params不是list[SamplingParams]，而是SamplingParams这种单个参数对象，那么就构造一个与prompts等长的列表。例如：[sampling_params] * 3会变成[sampling_params, sampling_params, sampling_params]
        for prompt, sp in zip(prompts, sampling_params):         # zip() 将两个列表按位置配对：prompts[0]  ↔ sampling_params[0] ， prompts[1]  ↔ sampling_params[1]     # 每次循环，prompt 是一条输入，sp 是它对应的采样参数    # 额外插一嘴zip的用法，当然这里不存在这个问题，因为第68行他们长度一定是一样的。两个列表长度不同时，zip() 只遍历较短的一方，这里没有额外校验长度
            self.add_request(prompt, sp)                         # 逐条创建请求并加入调度器
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
