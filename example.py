import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")            # expanduser() 把 ~ 替换为当前用户的主目录。这个目录需要提前存放模型权重、配置和分词器文件，是从hugging face下载的
    tokenizer = AutoTokenizer.from_pretrained(path)                   # AutoTokenizer.from_pretrained(path) 加载分词器，创建分词器对象
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)       # LLM(path, ...) 初始化推理引擎、加载模型。LLM 明明是一个类，为什么能这样当函数用？Python 中，类名(...) 就是在创建这个类的对象，叫“实例化”，别跟C++的写法搞混了
                                                                      # 右边 LLM(...)：创建对象。  左边 llm：给这个对象起的变量名
    # LLM 没有自己的 __init__，所以使用父类的这个方法。此处参数是这样接收的：path：第一个位置的参数，传给 model，两边名字不用相同。后两个带名字的参数：被 **kwargs 收集成一个字典：kwargs = { "enforce_eager": True,"tensor_parallel_size": 1} 然后程序把这些配置交给 Config 
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [                                        # 这里的语法是：新列表 = [处理表达式 for 元素 in 原列表]
        tokenizer.apply_chat_template(                 # apply_chat_template是调用分词器对象的方法，使用加载的模板整理聊天消息。同一个 apply_chat_template() 方法，加载不同模型的分词器后，可能输出不同的聊天格式
            [{"role": "user", "content": prompt}],     # 这一行提供聊天消息，先看里面的字典 {}：第一组是指这条消息的说话者是用户，第二组是指这条消息的内容是变量 prompt 当前保存的字符串。注意："content" 是带引号的固定键名，prompt 不带引号，所以会读取变量的值
                                                       # 比如第一次循环时，prompt 是 "introduce yourself"，所以相当于：{"role": "user","content": "introduce yourself"}
            tokenize=False,                            # False 表示关闭这里的 token 编码。因为我们把编码的工作交给了引擎，后面的 llm.generate() 会进行编码，所以这里关闭token 编码。
            add_generation_prompt=True,                # add_generation_prompt=True：加上“轮到助手回答”的标记
        )                                              # 这里的apply_chat_template()处理完之后，prompts是已经套好聊天格式的字符串。套用过程就像：“introduce yourself” → “用户：introduce yourself\n助手：” 。总的来说，主要是为了让输入符合模型训练时使用的对话格式，让模型知道“谁在说话、现在轮到谁回答”
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
