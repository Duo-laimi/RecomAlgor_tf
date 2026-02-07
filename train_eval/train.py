import importlib
import logging as log
import os

import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy

from dataset.dien import DienDatasetLoader
from dataset.utils import load_dataset
from utils.config import Config
from .utils import set_all_seeds

logger = log.getLogger(__name__)

def train_from_config(config: Config):
    train_dataset = load_dataset(**config.data_config["train"])
    eval_dataset = load_dataset(**config.data_config["eval"])
    set_all_seeds(config["seed"])
    train_batch_size = config["train_batch_size"]
    eval_batch_size = config["eval_batch_size"]
    _train_loader = DienDatasetLoader(train_dataset, True, config["train_limit"])
    _eval_loader = DienDatasetLoader(eval_dataset, True, config["eval_limit"])
    train_dataset_tf = tf.data.Dataset.from_generator(
        _train_loader,
        output_signature=_train_loader.get_output_signature()
    ).shuffle(buffer_size=1024).batch(train_batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    eval_dataset_tf = tf.data.Dataset.from_generator(
        _eval_loader,
        output_signature=_eval_loader.get_output_signature()
    ).batch(eval_batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    model_class_path = config["model_class"]
    parent_path, class_name = model_class_path.rsplit(".", 1)
    parent_module = importlib.import_module(parent_path)
    MODEL_CLASS = getattr(parent_module, class_name)
    model = MODEL_CLASS(**config.model_config)

    save_path = config["save_path"]
    ckpt_path = config["ckpt_path"]

    name = config["name"]
    save_path = os.path.join(save_path, name)

    from_scratch = config["from_scratch"]
    if os.path.exists(ckpt_path) and not from_scratch:
        # model.build(input_shape=(None, model.embedding_dim))
        # model.load_weights(ckpt_path)
        model = tf.keras.models.load_model(ckpt_path)
        logger.info(f"Training based on existing weights: {ckpt_path}.")
    else:
        opt_name = config["optimizer"]
        lr = config["learning_rate"]
        weight_decay = config["weight_decay"]
        opt = tf.keras.optimizers.get({"class_name": opt_name, "config": {"learning_rate": lr, "weight_decay": weight_decay}})
        loss = tf.keras.losses.get({"class_name": config['loss'], "config": {"from_logits": False}})
        met = [tf.keras.metrics.get(m) for m in config['metrics']]
        model.compile(optimizer=opt, loss=loss, metrics=met)

    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=f"logs/{name}", histogram_freq=1, update_freq=10)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True)
    history = model.fit(
        train_dataset_tf,
        epochs=config["num_epochs"],
        validation_data=eval_dataset_tf,
        callbacks=[tensorboard_cb, checkpoint_cb]
    )

    val_metrics = {}
    for name in history.history.keys():
        # name_split = name.split("_")
        # name = "_".join(name_split[:2])
        if name.startswith('val_'):
            val_metrics[name] = history.history[name][-1]

    # model.save(save_path)
    model.export(save_path)
    # 清除后端会话
    tf.keras.backend.clear_session()
    return val_metrics


# 目标是直接生成所有配置下的实验结果，构建成表

