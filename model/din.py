from typing import Union, List, Tuple, Callable, Optional

import tensorflow as tf


class Din(tf.keras.Model):
    def __init__(
            self,
            num_users: int,
            num_items: int,
            num_categories: int,
            hidden_size: int = 1024,
            use_negative: bool = True,
            activation: Optional[Union[str, Callable]] = "relu"
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_categories = num_categories
        self.hidden_size = hidden_size
        self.use_negative = use_negative

        self.user_embed = tf.keras.layers.Embedding(num_users, hidden_size)
        tf.summary.histogram("user_embed", self.user_embed, step=0)
        self.item_embed = tf.keras.layers.Embedding(num_items, hidden_size)
        tf.summary.histogram("item_embed", self.item_embed, step=0)
        self.category_embed = tf.keras.layers.Embedding(num_categories, hidden_size)
        tf.summary.histogram("category_embed", self.category_embed, step=0)

        self.dense1 = tf.keras.layers.Dense(hidden_size, activation=activation)
        self.dense2 = tf.keras.layers.Dense(hidden_size, activation=activation)
        self.dense3 = tf.keras.layers.Dense(1,)

    def din_attention(
            self,
            target_embedding: tf.Tensor,
            item_seq_embedding: Union[tf.Tensor, Tuple[tf.Tensor], List[tf.Tensor]],
            sequence_mask: tf.Tensor
    ):
        if not isinstance(item_seq_embedding, tf.Tensor):
            n = len(item_seq_embedding)
            item_seq_embedding = tf.concat(item_seq_embedding, axis=-1)
            target_embedding = tf.tile(target_embedding, [1, n])


    def call(
            self,
            user_ids: tf.Tensor,
            item_ids: tf.Tensor,
            category_ids: tf.Tensor,
            item_history: tf.Tensor,
            category_history: tf.Tensor,
            sequence_mask: tf.Tensor,
            negative_item_history: tf.Tensor,
            negative_category_history: tf.Tensor
    ):
        # ["label", "uid", "mid", "cat", "mid_history", "cat_history", "positive_mask", "negative_mid_history", "negative_cat_history"]
        pass