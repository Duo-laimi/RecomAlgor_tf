from typing import Union, List, Tuple, Optional

import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable

from .activations import ACT_FUNC


@register_keras_serializable()
class Din(tf.keras.Model):
    def __init__(
            self,
            num_users: int,
            num_items: int,
            num_categories: int,
            embedding_dim: int = 256,
            hidden_size: int = 1024,
            use_negative: bool = True,
            activation: Optional[str] = "relu",
            softmax_logits: bool = False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.num_users = num_users
        self.num_items = num_items
        self.num_categories = num_categories
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.use_negative = use_negative
        self.activation = activation
        self.user_embed = tf.keras.layers.Embedding(num_users, embedding_dim)
        self.item_embed = tf.keras.layers.Embedding(num_items, embedding_dim)
        self.category_embed = tf.keras.layers.Embedding(num_categories, embedding_dim)
        self.softmax_logits = softmax_logits

        ACT_CLASS = ACT_FUNC[activation]

        # TODO: 基于配置构建mlp
        self.mlp1 = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_size),
            tf.keras.layers.BatchNormalization(),
            ACT_CLASS(),
            tf.keras.layers.Dense(hidden_size // 2),
            tf.keras.layers.BatchNormalization(),
            ACT_CLASS(),
            tf.keras.layers.Dense(1, )
        ])

        self.mlp2 = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_size),
            tf.keras.layers.BatchNormalization(),
            ACT_CLASS(),
            tf.keras.layers.Dense(hidden_size // 2),
            tf.keras.layers.BatchNormalization(),
            ACT_CLASS(),
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
        # logits = tf.nn.dropout(logits, rate=0.5)
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
        # combined_item_embed = tf.nn.dropout(combined_item_embed, rate=0.2)
        combined_embed = tf.concat([user_embed, item_embed, combined_item_embed, item_embed * combined_item_embed], axis=-1)
        logits = self.mlp2(combined_embed) # B, 1
        logits = tf.nn.sigmoid(logits)
        return logits

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_users" : self.num_users,
            "num_items" : self.num_items,
            "num_categories" : self.num_categories,
            "embedding_dim" : self.embedding_dim,
            "hidden_size" : self.hidden_size,
            "use_negative" : self.use_negative,
            "activation" : self.activation,
            "softmax_logits" : self.softmax_logits,

        })
        return config