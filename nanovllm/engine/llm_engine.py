import atexit
from dataclasses import fields            # fields() 用来获取一个 dataclass 类定义的字段信息，包括字段名、类型、默认值等。Config类就是dataclass类定义的
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:                          # LLMEngine 这个类负责组织整个推理流程
                                          # kwargs 只是惯用变量名，真正起作用的是前面的 **，写成 **options 也可以
    def __init__(self, model, **kwargs):  # 在函数定义里，**kwargs 会把额外的“参数名=值”收集成一个字典。可以像普通字典一样读取：像example.py中传入了enforce_eager=True, tensor_parallel_size=1。故可以：kwargs["tensor_parallel_size"]  # 得到 1
        config_fields = {field.name for field in fields(Config)}                # 收集 Config 定义的字段名，这里取得的是字段的描述信息，并没有创建 Config 实例。语法：{表达式 for 变量 in 可迭代对象}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields} # 只保留 Config 支持的参数。这是一条字典推导式，可以拆成三部分：{k: v  # 把保留下来的键值对放入新字典  for k, v in kwargs.items() # 遍历原字典中的键值对  if k in config_fields   # 只保留字段名合法的项}
        config = Config(model, **config_kwargs)                                 # **config_kwargs 表示：把字典展开成关键字参数
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
        self.scheduler.add(seq)                                 # 加入scheduler.waiting。这里调用了scheduler类的add函数，而这个函数的函数就是将请求加入waiting队列

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()                                               # 从调度器中挑选本轮要处理的请求，schedule()函数在scheduler.py文件 第25行
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)   # 记录本轮处理量，使用正负号区分阶段。正数时，num_tokens等于本轮实际调度计算的输入 token 总数；负数时，num_tokens绝对值等于本轮处理的序列数，因为每条序列处理一个输入 token
        token_ids = self.model_runner.call("run", seqs, is_prefill)                                # 执行模型，采样下一个 token
        self.scheduler.postprocess(seqs, token_ids, is_prefill)                                    # 把结果写回请求，更新状态并检查是否完成
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]      # 收集本轮已完成请求的全部输出 token
        return outputs, num_tokens                                       

    def is_finished(self):
        return self.scheduler.is_finished()                      # 对应 Scheduler 类中的 is_finished() 函数，函数体内容是：return not self.waiting and not self.running。即没有等待处理的请求，并且没有正在运行的请求

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
        outputs = {}                                             # 创建保存完成结果的字典，这里只是先准备一个容器，用来跨多轮循环累计已完成请求的结果    # 我怎么好像发现有花括号 {} ，然后花括号里面有 键: 值 形式，就是字典呢
        prefill_throughput = decode_throughput = 0.              # 这两个变量的含义分别是最近一轮 prefill 和 decode 的处理速度，单位都是 token/秒，在这里我们都初始化为0
        while not self.is_finished():                            # 对应 第57行 的is_finished()函数，这里是意思是：只要还有请求未完成，就......
            t = perf_counter()                                   # perf_counter() 是用于测量耗时的高精度计时函数  #保存开始时间后，稍后计算：perf_counter() - t，就能得到从这里开始经过的秒数，用于计算本轮速度
            output, num_tokens = self.step()                     # 这个step()函数是在 第49行 定义的函数，这里表示让推理过程向前推进一轮
            if num_tokens > 0:                                   # prefill阶段
                prefill_throughput = num_tokens / (perf_counter() - t)         # prefill 速度 = 本轮计算的 token 数 ÷ 本轮耗时
            else:                                                # decode阶段
                decode_throughput = -num_tokens / (perf_counter() - t)         # 先用 -num_tokens 把计数恢复为正数，再除以耗时
            pbar.set_postfix(                                    # 调用 set_postfix()，开始传入一个字典，设置进度条末尾的附加信息
                "Prefill": f"int(prefill_throughput)}tok/s",     # 设置 Prefill 的显示内容  # 其中，int() 去掉小数部分，f-string 将数值放进字符串。例如 6400.8 会显示成：6400tok/s
                "Decode": f"int(decode_throughput)}tok/s",       # 设置 Decode 的显示内容  
            })
            for seq_id, token_ids in output:                     # 逐个取出已完成请求，并把每个二元组解包为：seq_id    请求的序列 ID ，token_ids   这个请求生成的全部输出 token      # 这里的 token_ids 来自 seq.completion_token_ids，只包含输出部分，不包含 prompt
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()                                             # 所有请求处理完毕后关闭进度条。注意缩进：这一行已经位于 while 循环之外，因此，它在全部请求结束后执行一次，用于完成进度条显示并释放相关显示资源
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]       # 按序列 ID 排序，将结果恢复为输入顺序
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]  # 将 token ID 解码成文本，构造最终结果
        return outputs                                           # 返回全部生成结果，这是一个字典
