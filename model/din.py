import tensorflow as tf


class Din(tf.keras.Model):
    def __init__(
            self,
            num_users: int,
            num_items: int,
            num_categories: int,
            embedding_dim: int = 1024,
            use_negative: bool = True

    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_categories = num_categories
        self.embedding_dim = embedding_dim
        self.use_negative = use_negative

        self.user_embed = tf.keras.layers.Embedding(num_users, embedding_dim)
        tf.summary.histogram("user_embed", self.user_embed, step=0)
        self.item_embed = tf.keras.layers.Embedding(num_items, embedding_dim)
        tf.summary.histogram("item_embed", self.item_embed, step=0)
        self.category_embed = tf.keras.layers.Embedding(num_categories, embedding_dim)
        tf.summary.histogram("category_embed", self.category_embed, step=0)

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