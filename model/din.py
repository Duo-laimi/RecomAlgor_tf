from typing import Union, List, Tuple, Callable, Optional

import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable


@register_keras_serializable("din")
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
        self.item_embed = tf.keras.layers.Embedding(num_items, hidden_size)
        self.category_embed = tf.keras.layers.Embedding(num_categories, hidden_size)
        self.softmax_logits = softmax_logits

        self.mlp1 = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_size // 2, activation=activation),
            tf.keras.layers.Dense(hidden_size // 4, activation=activation),
            tf.keras.layers.Dense(1, )
        ])

        self.mlp2 = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_size // 2, activation=activation),
            tf.keras.layers.Dense(hidden_size // 4, activation=activation),
            tf.keras.layers.Dense(1, )
        ])

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
        # target_embedding = tf.broadcast_to(target_embedding, item_seq_embedding.shape())
        seq_len = tf.shape(item_seq_embedding)[1]
        target_embedding = tf.tile(target_embedding, [1, seq_len, 1])
        combined_embedding = tf.concat([
            target_embedding, item_seq_embedding,
            target_embedding - item_seq_embedding, target_embedding * item_seq_embedding
        ], axis=-1)
        logits = self.mlp1(combined_embedding) # B, L, 1
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
            inputs
    ):
        user_ids, item_ids, category_ids, \
            item_history, category_history, \
            sequence_mask, negative_item_history, negative_category_history = inputs
        user_embed = self.user_embed(user_ids)
        item_embed = self.item_embed(item_ids)
        category_embed = self.category_embed(category_ids)

        item_history_embed = self.item_embed(item_history)
        category_history_embed = self.category_embed(category_history)

        # negative_item_history_embed = self.item_embed(negative_item_history)
        # negative_category_history_embed = self.category_embed(negative_category_history)

        item_embed = item_embed + category_embed
        item_history_embed = item_history_embed + category_history_embed

        combined_item_embed = self.din_attention(item_embed, item_history_embed, sequence_mask)
        combined_embed = tf.concat([user_embed, item_embed, combined_item_embed, item_embed * combined_item_embed], axis=-1)
        logits = self.mlp2(combined_embed) # B, 1
        return logits