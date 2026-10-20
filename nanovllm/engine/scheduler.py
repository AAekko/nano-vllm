from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):         # 调度器维护两个双端队列（deque）和若干预算约束
        self.max_num_seqs = config.max_num_seqs                       # 一轮最多几条序列
        self.max_num_batched_tokens = config.max_num_batched_tokens   # 一次批量处理的 token 预算
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()                       # 新来的、还没 prefill 的序列
        self.running: deque[Sequence] = deque()                       # 已 prefill、正在逐 token decode 的序列
    
    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)
    # schedule() 的逻辑是优先 prefill：只要 waiting 里有序列且预算允许，就先做 prefill；waiting 空了，才去 running 做 decode。注意：这里是不可能将 prefill 和 decode 混批的
    def schedule(self) -> tuple[list[Sequence], bool]:  # 返回值用元组很合适，因为它表达的是位置和含义固定的一组信息。其中请求列表可以有很多个请求，但外层元组始终只有这两个元素   # 最后返回两个信息：1.哪些请求要参与本轮计算：scheduled_seqs。 2.本轮执行 Prefill 还是 Decode：is_prefill。
        scheduled_seqs = []                             # 保存本轮选中的请求
        num_batched_tokens = 0                          # 统计本轮 Prefill 已经安排了多少个 token

        # prefill（从 waiting 队头依次取序列，往一个 batch 里塞，直到撞上任一预算上限）
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:  # 只要还有等待请求，而且本轮请求数没有达到上限，就继续尝试
            seq = self.waiting[0]                                        # self.waiting[0] 只是查看队首，没有立即取出，因为这个请求可能无法调度，也可能需要分多轮完成 Prefill
            remaining = self.max_num_batched_tokens - num_batched_tokens # 这批剩余的 token 预算
            if remaining == 0:
                break
            if not seq.block_table:                                      # 如果这条请求是首次调度，先判断是否有足够的块供它分配，如果够的话还要探测有几个块可以复用
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:                              # can_allocate() 返回值为 -1，说明空闲块不足，无法分配
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size  # 减去可复用缓存块上的 token，还有多少 token
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens      # 并非首次调度，chunked prefill 续算：还有多少 token 待处理
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:                                      # 如果这条请求是首次调度，现在给这条首次调度的请求正式分配块
                self.block_manager.allocate(seq, num_cached_blocks)      
            seq.num_scheduled_tokens = min(num_tokens, remaining)        # 允许对「本轮第一条」做 chunked prefill：只算预算内的部分
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True                  # Python 中，逗号分隔的多个返回值会打包成一个 tuple（元组）  # 这里等价于 return (scheduled_seqs, True) ，调用方再通过“解包”分别接收：例如 seqs, is_prefill = self.scheduler.schedule()

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
