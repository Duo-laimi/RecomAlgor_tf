import tensorflow as tf
import numpy as np


class CosineWarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, warmup_steps, total_steps):
        """
        initial_lr: 预热结束后的峰值学习率
        warmup_steps: 预热步数
        total_steps: 总训练步数 (epochs * steps_per_epoch)
        """
        super(CosineWarmupSchedule, self).__init__()
        self.initial_lr = tf.cast(initial_lr, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.total_steps = tf.cast(total_steps, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)

        # 1. 计算线性预热阶段的学习率
        warmup_lr = self.initial_lr * (step / self.warmup_steps)

        # 2. 计算余弦退火阶段的学习率
        # 余弦衰减公式: 0.5 * initial_lr * (1 + cos(pi * current_decay_step / total_decay_steps))
        completed_fraction = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cosine_lr = 0.5 * self.initial_lr * (1.0 + tf.cos(np.pi * completed_fraction))

        # 3. 使用 tf.where 根据当前步数选择对应的逻辑
        return tf.where(step < self.warmup_steps, warmup_lr, tf.maximum(0.0, cosine_lr))

    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps
        }