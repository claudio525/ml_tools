import tensorflow as tf


def get_min_max_scaling_fn(target_min: float, target_max: float):
    def min_max_scaling(X: tf.Tensor):
        X_min, X_max = tf.reduce_min(X, axis=0), tf.reduce_max(X, axis=0)

        X_std = (X - X_min) / (X_max - X_min)
        return X_std * (target_max - target_min) + target_min

    return tf.function(min_max_scaling)