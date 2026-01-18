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
            activation: Optional[Union[str, Callable]] = "relu",
            softmax_logits: bool = False
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

        self.softmax_logits = softmax_logits

    def din_attention(
            self,
            target_embedding: tf.Tensor,
            item_seq_embedding: Union[tf.Tensor, Tuple[tf.Tensor], List[tf.Tensor]],
            sequence_mask: tf.Tensor
    ):
        """
        :param target_embedding: B, D
        :param item_seq_embedding: B, L, D
        :param sequence_mask: B, L
        :return:
        """
        if not isinstance(item_seq_embedding, tf.Tensor):
            n = len(item_seq_embedding)
            item_seq_embedding = tf.concat(item_seq_embedding, axis=-1)
            target_embedding = tf.tile(target_embedding, [1, n])

        target_embedding = tf.expand_dims(target_embedding, axis=1) # B, 1, D
        target_embedding = tf.broadcast_to(target_embedding, item_seq_embedding.shape)
        combined_embedding = tf.concat([
            target_embedding, item_seq_embedding,
            target_embedding - item_seq_embedding, target_embedding * item_seq_embedding
        ], axis=-1)
        logits = self.dense3(self.dense2(self.dense1(combined_embedding))) # B, L, 1
        # logits = tf.squeeze(logits, axis=-1) # B, L
        sequence_mask = tf.cast(sequence_mask, tf.bool)
        sequence_mask = tf.expand_dims(sequence_mask, axis=-1)  # B, L, 1
        # 掩码
        if self.softmax_logits:
            logits = tf.where(sequence_mask, logits, -1e9)
            logits = tf.nn.softmax(logits, axis=1)
        else:
            logits = tf.where(sequence_mask, logits, 0)
        output = logits * item_seq_embedding
        output = tf.reduce_sum(output, axis=1)
        return output

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