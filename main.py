from parse_args import parse_args
from utils.config import Config
from utils.common import setup_logging
from dataset.amazon_book import load_amazon_book as load_dataset
from dataset.base import DienDatasetLoader
from  model.din import Din

import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import binary_accuracy, AUC, Recall, Precision


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
    ).batch(train_batch_size).prefetch(5)
    eval_dataset_tf = tf.data.Dataset.from_generator(
        _data_loader.eval_call,
        output_signature=_data_loader.get_output_signature()
    ).batch(eval_batch_size).prefetch(5)
    model_args = {
        "num_users": dataset.num_users,
        "num_items": dataset.num_items,
        "num_categories": dataset.num_categories
    }
    config.update_config(model_args, "model_config")

    model = Din(**config.model_config)
    optimizer_args = training_args["optimizer_args"]
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
    tf_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/din", histogram_freq=1)
    model.fit(
        train_dataset_tf,
        epochs=training_args["num_epochs"],
        validation_data=eval_dataset_tf,
        callbacks=[tf_callback]
    )



if __name__ == "__main__":
    argv = parse_args()
    main(argv)