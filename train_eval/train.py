import os
import logging as log
from typing import Type
from datetime import datetime

import tensorflow as tf

from dataset.dien import DienDatasetLoader
from dataset.utils import load_dataset
from main import set_all_seeds
from model.din import Din
from utils.config import Config

from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import binary_accuracy, AUC, Recall, Precision

logger = log.getLogger(__name__)


def train(
        model,
        optimizer,
        criterion,
        metrics,
        num_epochs,
        train_dataset,
        eval_dataset,
        callbacks=None,
        from_scratch=True,
        ckpt_path=None,
        export_path=None
):
    if os.path.exists(ckpt_path) and not from_scratch:
        model = tf.keras.models.load_model(ckpt_path)
        logger.info(f"Training based on existing weights: {ckpt_path}.")
    else:
        model.compile(
            optimizer=optimizer,
            loss=criterion,
            metrics=metrics
        )
    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=eval_dataset,
        callbacks=callbacks
    )
    if os.path.exists(export_path):
        model.export(export_path)
        logger.info(f"Model exported to: {export_path}.")


def train_from_config(
        config: Config,
        MODEL_CLASS: Type[tf.keras.Model] = Din
):
    train_dataset = load_dataset(**config.data_config["train"])
    eval_dataset = load_dataset(**config.data_config["eval"])
    training_args = config.training_config
    set_all_seeds(training_args["seed"])
    train_batch_size = training_args["train_batch_size"]
    eval_batch_size = training_args["eval_batch_size"]
    _train_loader = DienDatasetLoader(train_dataset, True)
    _eval_loader = DienDatasetLoader(eval_dataset, True)
    train_dataset_tf = tf.data.Dataset.from_generator(
        _train_loader,
        output_signature=_train_loader.get_output_signature()
    ).shuffle(buffer_size=1024).batch(train_batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    eval_dataset_tf = tf.data.Dataset.from_generator(
        _eval_loader,
        output_signature=_eval_loader.get_output_signature()
    ).batch(eval_batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    model = MODEL_CLASS(**config.model_config)
    optimizer_args = training_args["optimizer_args"]

    save_path = config.training_config["save_path"]
    ckpt_path = config.training_config["ckpt_path"]
    from_scratch = config.training_config["from_scratch"]
    if os.path.exists(ckpt_path) and not from_scratch:
        # model.build(input_shape=(None, model.embedding_dim))
        # model.load_weights(ckpt_path)
        model = tf.keras.models.load_model(ckpt_path)
        logger.info(f"Training based on existing weights: {ckpt_path}.")
    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(**optimizer_args),
            loss=binary_crossentropy,
            metrics=[
                binary_accuracy,
                Precision(name="precision"),
                Recall(name="recall"),
                AUC(name="auc")
            ]
        )
    now = datetime.now()
    date_time = now.strftime("%Y%m%d%H%M")
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=f"logs/{MODEL_CLASS}/{date_time}", histogram_freq=1, update_freq=10)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True)
    model.fit(
        train_dataset_tf,
        epochs=training_args["num_epochs"],
        validation_data=eval_dataset_tf,
        callbacks=[tensorboard_cb, checkpoint_cb]
    )
    # model.save(save_path)
    model.export(save_path)


# 目标是直接生成所有配置下的实验结果，构建成表

