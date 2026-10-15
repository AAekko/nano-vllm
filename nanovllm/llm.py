from nanovllm.engine.llm_engine import LLMEngine
# 这个文件主要是为后续维护预留空间，当前这层的实际功能收益确实很小

class LLM(LLMEngine):    # LLM 类继承了 LLMEngine 类
    pass                 # pass 是占位语句，本身什么也不做，表示这里暂时没有额外的实现。继承来的功能仍然可以正常使用
                         # 因为 Python 要求 class ...: 后面必须有一个缩进的代码块
