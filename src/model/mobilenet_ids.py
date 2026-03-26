import tensorflow as tf
from tensorflow.keras.layers import (
    Conv1D, DepthwiseConv1D, LayerNormalization, ReLU,
    GlobalAveragePooling1D, Dropout, Dense, Input, Add
)
from tensorflow.keras import models
import numpy as np
from typing import List, Tuple, Dict


def _make_divisible(v: float, divisor: int = 8, min_value: int = None) -> int:
    """Ensures value is divisible by divisor (MobileNet convention)."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _inverted_residual_block(x, filters, stride, expansion, block_id):
    in_channels = x.shape[-1]
    expanded_channels = in_channels * expansion
    prefix = f'block_{block_id}_'

    shortcut = x

    if expansion != 1:
        x = Conv1D(expanded_channels, 1, padding='same', use_bias=False, name=prefix + 'expand')(x)
        x = LayerNormalization(epsilon=1e-6, name=prefix + 'expand_bn')(x)
        x = ReLU(6.0, name=prefix + 'expand_relu')(x)

    x = DepthwiseConv1D(kernel_size=3, strides=stride, padding='same',
                         use_bias=False, name=prefix + 'depthwise')(x)
    x = LayerNormalization(epsilon=1e-6, name=prefix + 'depthwise_bn')(x)
    x = ReLU(6.0, name=prefix + 'depthwise_relu')(x)

    x = Conv1D(filters, 1, padding='same', use_bias=False, name=prefix + 'project')(x)
    x = LayerNormalization(epsilon=1e-6, name=prefix + 'project_bn')(x)

    # Add residual connection when input/output shapes match
    if stride == 1 and in_channels == filters:
        x = Add(name=prefix + 'add')([shortcut, x])

    return x


def build_mobilenetv2_1d(
    input_shape: Tuple[int, int],
    num_classes: int,
    alpha: float = 0.75,
) -> tf.keras.Model:
    """
    Builds a 1D MobileNetV2-style model for tabular/sequence intrusion detection.

    Args:
        input_shape: (num_features, 1)
        num_classes: Number of output classes.
        alpha: Width multiplier. 0.75 for edge deployment.

    Returns:
        Uncompiled tf.keras.Model.
    """
    # Initial conv stem
    first_filters = _make_divisible(32 * alpha)
    inputs = Input(shape=input_shape, name='input')
    x = Conv1D(first_filters, kernel_size=3, strides=1, padding='same',
               use_bias=False, name='conv_stem')(inputs)
    x = LayerNormalization(epsilon=1e-6, name='conv_stem_bn')(x)
    x = ReLU(6.0, name='conv_stem_relu')(x)

    # Inverted residual blocks: (expansion, filters, stride)
    block_configs = [
        (1, 16,  1),
        (6, 24,  1),
        (6, 24,  1),
        (6, 32,  1),
        (6, 32,  1),
        (6, 64,  1),
        (6, 64,  1),
        (6, 96,  1),
    ]

    for block_id, (expansion, filters, stride) in enumerate(block_configs):
        adjusted_filters = _make_divisible(filters * alpha)
        x = _inverted_residual_block(x, adjusted_filters, stride, expansion, block_id)

    # Head
    last_filters = _make_divisible(1280 * alpha)
    x = Conv1D(last_filters, 1, use_bias=False, name='conv_last')(x)
    x = LayerNormalization(epsilon=1e-6, name='conv_last_bn')(x)
    x = ReLU(6.0, name='conv_last_relu')(x)
    x = GlobalAveragePooling1D(name='global_pool')(x)
    x = Dropout(0.3)(x)

    if num_classes == 2:
        outputs = Dense(1, activation='sigmoid', name='predictions')(x)
    else:
        outputs = Dense(num_classes, activation='softmax', name='predictions')(x)

    return models.Model(inputs=inputs, outputs=outputs, name='MobileNetV2_1D')


class MobileNetV2IDS:
    def __init__(
        self,
        input_shape: Tuple[int, int],
        num_classes: int = 4,
        learning_rate: float = 0.001,
        dp_enabled: bool = False,
        l2_norm_clip: float = 1.0,
        noise_multiplier: float = 1.1,
        num_microbatches: int = 1,
        alpha: float = 0.75,
    ):
        """
        1D MobileNetV2 for network intrusion detection with optional DP-SGD training.
        Drop-in replacement for IntrusionDetectionCNN.

        Args:
            input_shape: Tuple of (num_features, 1) for the 1D conv input.
            num_classes: Number of output classes (4 default for this variant).
            learning_rate: Optimizer learning rate.
            dp_enabled: If True, replaces Adam with DPKerasAdamOptimizer.
            l2_norm_clip: Per-example gradient clipping bound (C). Only used when dp_enabled=True.
            noise_multiplier: Ratio of Gaussian noise std to l2_norm_clip. Only used when dp_enabled=True.
            num_microbatches: Number of microbatches for per-example gradient computation.
                             Must divide batch_size evenly. Set to 1 for batch-level DP
                             (faster, weaker guarantee) or batch_size for example-level DP
                             (slower, stronger guarantee). Only used when dp_enabled=True.
            alpha: MobileNetV2 width multiplier. 0.75 for edge deployment.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.dp_enabled = dp_enabled
        self.l2_norm_clip = l2_norm_clip
        self.noise_multiplier = noise_multiplier
        self.num_microbatches = num_microbatches
        self.alpha = alpha
        self.model = self._build_model()

    def _build_model(self) -> tf.keras.Model:
        """
        Builds and compiles the 1D MobileNetV2.
        When dp_enabled=True, compiles with DPKerasAdamOptimizer (TF Privacy 0.9.x).
        The loss function uses per-example reduction (tf.keras.losses.Reduction.NONE)
        when DP is active — required by DPKerasAdamOptimizer's microbatch mechanism.
        """
        model = build_mobilenetv2_1d(self.input_shape, self.num_classes, self.alpha)

        if self.num_classes == 2:
            if self.dp_enabled:
                loss_fn = tf.keras.losses.BinaryCrossentropy(
                    from_logits=False,
                    reduction=tf.keras.losses.Reduction.NONE  # Required for DP microbatching
                )
            else:
                loss_fn = 'binary_crossentropy'
        else:
            if self.dp_enabled:
                loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=False,
                    reduction=tf.keras.losses.Reduction.NONE  # Required for DP microbatching
                )
            else:
                loss_fn = 'sparse_categorical_crossentropy'

        if self.dp_enabled:
            from tensorflow_privacy import DPKerasAdamOptimizer
            optimizer = DPKerasAdamOptimizer(
                l2_norm_clip=self.l2_norm_clip,
                noise_multiplier=self.noise_multiplier,
                num_microbatches=self.num_microbatches,
                learning_rate=self.learning_rate
            )
            print(f"[MobileNetV2] DP-SGD enabled: l2_clip={self.l2_norm_clip}, "
                  f"noise_mult={self.noise_multiplier}, microbatches={self.num_microbatches}")
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            print("[MobileNetV2] Standard Adam optimizer (no DP).")

        model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
        return model

    def train_on_stream(self, tf_dataset: tf.data.Dataset, epochs: int = 1) -> Dict:
        """
        Trains the model on the streaming dataset.

        Args:
            tf_dataset: Batched tf.data.Dataset with drop_remainder=True.
            epochs: Number of local training epochs.

        Returns:
            Training history dict including loss and accuracy per epoch.
        """
        history = self.model.fit(tf_dataset, epochs=epochs, verbose=1)
        return history.history

    def get_weights(self) -> List[np.ndarray]:
        """Returns model weight arrays. When DP is enabled these weights reflect
        DP-SGD trained parameters — noise has already been applied during training."""
        return self.model.get_weights()

    def set_weights(self, weights: List[np.ndarray]) -> None:
        """Applies aggregated weights received from the FL server."""
        self.model.set_weights(weights)

    def evaluate_stream(self, tf_dataset: tf.data.Dataset) -> Dict:
        """Evaluates model on a streaming dataset."""
        loss, accuracy = self.model.evaluate(tf_dataset, verbose=0)
        return {"loss": float(loss), "accuracy": float(accuracy)}


if __name__ == "__main__":
    m = MobileNetV2IDS(input_shape=(39, 1), num_classes=7)
    m.model.summary()
    print(f"Total params: {m.model.count_params():,}")
