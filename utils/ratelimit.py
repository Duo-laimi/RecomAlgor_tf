import time
from collections import deque
from functools import wraps
from threading import Lock


# 定义一个限速器装饰器类
class RateLimitDecorator:
    def __init__(self, max_calls, period):
        # 初始化限速器，指定最大调用次数和时间周期
        self.max_calls = max_calls  # 每个时间周期内允许的最大调用次数
        self.period = period  # 时间周期长度（秒）
        self.calls = deque()  # 使用双端队列来存储每次调用的时间戳
        self.lock = Lock()  # 创建一个锁对象，用于同步线程之间的访问

    def __call__(self, func):
        # 定义装饰器，当装饰的函数被调用时执行
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.lock:
                current_time = time.time()  # 获取当前时间戳

                # 移除在当前时间周期之外的调用记录
                while self.calls and self.calls[0] <= current_time - self.period:
                    self.calls.popleft()

                # 检查当前的调用次数是否在限制之内
                if len(self.calls) < self.max_calls:
                    # 如果没有超过限制，记录这次调用的时间戳
                    self.calls.append(current_time)
                else:
                    # 如果超过限制，计算需要等待的时间并进行睡眠
                    sleep_time = self.calls[0] + self.period - current_time
                    time.sleep(sleep_time)
                    # 记录这次调用的时间戳（包括睡眠时间后的时间）
                    self.calls.append(current_time + sleep_time)

            # 调用被装饰的实际函数
            return func(*args, **kwargs)

        return wrapper  # 返回包装后的函数