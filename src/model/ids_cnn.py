import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import numpy as np
from typing import List, Tuple, Dict, Optional

class IntrusionDetectionCNN:
    def __init__(
        self,
        input_shape: Tuple[int, int],
        num_classes: int = 6,
        learning_rate: float = 0.001,
        dp_enabled: bool = False,
        l2_norm_clip: float = 1.0,
        noise_multiplier: float = 1.1,
        num_microbatches: int = 1,
    ):
        """
        1D-CNN for network intrusion detection with optional DP-SGD training.

        Args:
            input_shape: Tuple of (num_features, 1) for the 1D conv input.
            num_classes: Number of output classes (6 for ToN-IoT).
            learning_rate: Optimizer learning rate.
            dp_enabled: If True, replaces Adam with DPKerasAdamOptimizer.
            l2_norm_clip: Per-example gradient clipping bound (C). Only used when dp_enabled=True.
            noise_multiplier: Ratio of Gaussian noise std to l2_norm_clip. Only used when dp_enabled=True.
            num_microbatches: Number of microbatches for per-example gradient computation.
                             Must divide batch_size evenly. Set to 1 for batch-level DP
                             (faster, weaker guarantee) or batch_size for example-level DP
                             (slower, stronger guarantee). Only used when dp_enabled=True.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.dp_enabled = dp_enabled
        self.l2_norm_clip = l2_norm_clip
        self.noise_multiplier = noise_multiplier
        self.num_microbatches = num_microbatches
        self.model = self._build_model()

    def _build_model(self) -> tf.keras.Model:
        """
        Builds and compiles the 1D-CNN.
        When dp_enabled=True, compiles with DPKerasAdamOptimizer (TF Privacy 0.9.x).
        The loss function uses per-example reduction (tf.keras.losses.Reduction.NONE)
        when DP is active — required by DPKerasAdamOptimizer's microbatch mechanism.
        """
        model = models.Sequential([
            layers.InputLayer(input_shape=self.input_shape),
            layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            layers.Flatten(),
            layers.Dense(64, activation='relu', kernel_regularizer=None if self.dp_enabled else regularizers.l2(0.01)),
            layers.Dropout(0.4)
        ])

        if self.num_classes == 2:
            model.add(layers.Dense(1, activation='sigmoid'))
            if self.dp_enabled:
                loss_fn = tf.keras.losses.BinaryCrossentropy(
                    from_logits=False,
                    reduction=tf.keras.losses.Reduction.NONE  # Required for DP microbatching
                )
            else:
                loss_fn = 'binary_crossentropy'
        else:
            model.add(layers.Dense(self.num_classes, activation='softmax'))
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
            print(f"[CNN] DP-SGD enabled: l2_clip={self.l2_norm_clip}, "
                  f"noise_mult={self.noise_multiplier}, microbatches={self.num_microbatches}")
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            print("[CNN] Standard Adam optimizer (no DP).")

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