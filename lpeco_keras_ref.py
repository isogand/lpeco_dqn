"""Verbatim copy of the Keras LPECO class from the original lpeco.py.

Extracted unchanged (class body only) so it can be imported without running the
whole benchmark script. Used only by test_lpeco_torch.py::KerasReferenceTest.
"""
from typing import Union

import keras
import tensorflow as tf


class LionPredictiveErrorCorrectionOptimizer(keras.optimizers.Optimizer):
    """
    Lion with Predictive Error Correction (LPECO) optimizer.
    
    A Lion variant that integrates predictive error correction using principles
    from control theory to improve training stability on complex loss surfaces.
    """
    
    def __init__(
        self,
        learning_rate: Union[float, keras.optimizers.schedules.LearningRateSchedule] = 0.0001,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        error_decay: float = 0.999,
        correction_strength: float = 0.1,
        name: str = "LionPredictiveErrorCorrectionOptimizer",
        **kwargs
    ):
        super().__init__(learning_rate=learning_rate, name=name, **kwargs)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.error_decay = error_decay
        self.correction_strength = correction_strength
        self._built = False
        self._variable_map = {}

    def build(self, var_list):
        """Build optimizer state variables."""
        if self._built:
            return
        super().build(var_list)
        for var in var_list:
            safe_var_name = var.name.replace(':', '_').replace('/', '_')
            self._variable_map[var.name] = {
                'm': self.add_variable_from_reference(var, f"lpec_m_{safe_var_name}"),
                'error_ema': self.add_variable_from_reference(var, f"lpec_ee_{safe_var_name}")
            }
        self._built = True

    def update_step(self, gradient, variable, learning_rate=None):
        """Apply the predictive error correction update logic."""
        if gradient is None:
            return

        var_name = variable.name
        if var_name not in self._variable_map:
            safe_var_name = variable.name.replace(':', '_').replace('/', '_')
            self._variable_map[var_name] = {
                'm': self.add_variable_from_reference(variable, f"lpec_m_{safe_var_name}"),
                'error_ema': self.add_variable_from_reference(variable, f"lpec_ee_{safe_var_name}")
            }

        state = self._variable_map[var_name]
        m, error_ema = state['m'], state['error_ema']

        var_dtype = variable.dtype
        # Rule 5
        lr = self.learning_rate if learning_rate is None else learning_rate
        lr_t = tf.cast(lr, var_dtype)

        beta_1 = tf.cast(self.beta_1, var_dtype)
        beta_2 = tf.cast(self.beta_2, var_dtype)
        error_decay = tf.cast(self.error_decay, var_dtype)
        correction_strength = tf.cast(self.correction_strength, var_dtype)
        grad = tf.convert_to_tensor(gradient, var_dtype)

        # Lion Predictive Error Correction Logic
        # Calculate prediction error (momentum predicts -grad)
        error = grad + m

        # Update exponential moving average of the error
        new_error_ema = error_decay * error_ema + (1.0 - error_decay) * error

        # Lion-style interpolation
        interpolated = beta_1 * m + (1.0 - beta_1) * grad

        # Correct the interpolated update with the error EMA
        corrected_update = interpolated - correction_strength * new_error_ema

        # Final update direction is the sign of the corrected update
        final_update = tf.sign(corrected_update)

        # Update momentum state
        new_momentum = beta_2 * m + (1.0 - beta_2) * grad

        # Apply the update
        variable.assign_sub(lr_t * final_update)

        # Update state variables
        m.assign(new_momentum)
        error_ema.assign(new_error_ema)

    def get_config(self):
        config = super().get_config()
        config.update({
            "beta_1": self.beta_1, "beta_2": self.beta_2,
            "error_decay": self.error_decay, "correction_strength": self.correction_strength
        })
        return config

