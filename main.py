import os.path
import logging as log

from parse_args import parse_args
from utils.config import Config
from utils.common import setup_logging
from dataset.amazon_book import load_amazon_book as load_dataset
from dataset.base import DienDatasetLoader
from  model.din import Din

import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import binary_accuracy, AUC, Recall, Precision

logger = log.getLogger(__name__)

def main(args):
    setup_logging(args.log_config)
    config = Config(args.config)
    dataset = load_dataset(**config.data_config)
    training_args = config.training_config
    train_batch_size = training_args["train_batch_size"]
    eval_batch_size = training_args["eval_batch_size"]
    _data_loader = DienDatasetLoader(dataset, 0.1, True)
    train_dataset_tf = tf.data.Dataset.from_generator(
        _data_loader.train_call,
        output_signature=_data_loader.get_output_signature()
    ).shuffle(buffer_size=1024).batch(train_batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    eval_dataset_tf = tf.data.Dataset.from_generator(
        _data_loader.eval_call,
        output_signature=_data_loader.get_output_signature()
    ).batch(eval_batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    model_args = {
        "num_users": dataset.num_users,
        "num_items": dataset.num_items,
        "num_categories": dataset.num_categories
    }
    config.model_config.update(model_args)
    print(config.model_config)

    model = Din(**config.model_config)
    optimizer_args = training_args["optimizer_args"]

    save_path = config.config["save_path"]
    ckpt_path = config.config["ckpt_path"]
    # if os.path.exists(ckpt_path):
    #     model.build(input_shape=(None, model.embedding_dim))
    #     # model.load_weights(ckpt_path)
    #     model = tf.keras.models.load_model(ckpt_path)
    #     logger.info(f"Training based on existing weights: {ckpt_path}.")
    # else:
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
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir="logs/din", histogram_freq=1, update_freq=10)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True)
    model.fit(
        train_dataset_tf,
        epochs=training_args["num_epochs"],
        validation_data=eval_dataset_tf,
        callbacks=[tensorboard_cb, checkpoint_cb]
    )
    model.save(save_path)



if __name__ == "__main__":
    argv = parse_args()
    main(argv)