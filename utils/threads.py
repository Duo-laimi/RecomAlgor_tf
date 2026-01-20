from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from tqdm import tqdm


# 有序的线程池
def submit_tasks(tasks, handler, max_workers=10, **kwargs):
    """Submit tasks to a thread pool and return the results."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        processor = partial(handler, **kwargs)
        results = list(executor.map(processor, tasks))
    return results


# 使用tqdm显示进度条，结果无序
def submit_tasks_unordered(tasks, handler, max_workers=10, **kwargs):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        processor = partial(handler, **kwargs)
        # 提交所有任务
        futures = {executor.submit(processor, task): task for task in tasks}

        # 使用 tqdm 显示进度条
        for future in tqdm(as_completed(futures), total=len(tasks)):
            result = future.result()
            results.append(result)
    return results


def submit_tasks_ordered(tasks, handler, max_workers=10, **kwargs):
    """
    使用线程池并行处理任务，并保持输出结果的顺序与输入任务的顺序一致。

    参数:
        tasks: 要处理的任务列表
        handler: 处理单个任务的函数
        max_workers: 线程池的最大工作线程数
        **kwargs: 传递给handler的额外参数

    返回:
        与输入tasks顺序一致的处理结果列表
    """
    results = [None] * len(tasks)  # 预先分配结果列表，保持顺序
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        processor = partial(handler, **kwargs)

        # 提交所有任务，并记录每个任务的索引
        futures = {
            executor.submit(processor, task): idx
            for idx, task in enumerate(tasks)
        }

        # 使用tqdm显示进度条
        for future in tqdm(as_completed(futures), total=len(tasks)):
            idx = futures[future]  # 获取任务的原始索引
            result = future.result()  # 获取任务结果
            results[idx] = result  # 按原始索引放入结果列表

    return results


