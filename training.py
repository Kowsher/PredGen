
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer
import inspect
class GenTrainer(Trainer):
    def __init__(self, *args, max_sampling_prob=1.0, min_sampling_prob=0.0, max_steps_for_sampling=500, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_sampling_prob = max_sampling_prob
        self.min_sampling_prob = min_sampling_prob
        self.max_steps_for_sampling = max_steps_for_sampling
        self.current_step = 0

    def training_step(self, model, inputs, num_items=None):
        # Increment the current step
        self.current_step += 1

        # Calculate the sampling probability
        if self.current_step <= self.max_steps_for_sampling:
            sampling_prob = self.min_sampling_prob + (self.max_sampling_prob - self.min_sampling_prob) * (self.current_step / self.max_steps_for_sampling)
        else:
            sampling_prob = self.max_sampling_prob

        # Add the sampling_prob to the inputs
        inputs['sampling_prob'] = sampling_prob
        inputs['is_inference'] = False  # Indicate training mode

        # Check if the base class method expects 'num_items'
        base_training_step = super().training_step
        signature = inspect.signature(base_training_step)
        if 'num_items' in signature.parameters:
            return base_training_step(model, inputs, num_items)
        else:
            return base_training_step(model, inputs)


    def evaluation_step(self, model, inputs):
        # During evaluation, no need for sampling_prob, so pass it as inference
        inputs['is_inference'] = True  # Evaluation mode
        return super().evaluation_step(model, inputs)
