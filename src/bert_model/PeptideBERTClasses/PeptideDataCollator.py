from transformers import DataCollatorForLanguageModeling
import numpy as np

class PeptideCurriculumDataCollator(DataCollatorForLanguageModeling):
    """
    This implementation should provide a Curriculum Learning approach to the MLM task.
    The update strategy can be adjusted this is just a naive implementation. Im not sure if different
    strategies yield better results.
    Strategy could be increasing the mlm probability if current epoch deviates from the mean loss of the last n epochs.
    This could be even used to decrease the mlm probability if the model is overfitting or suddenly loosing too much accuracy.
    """
    def __init__(self, tokenizer, initial_prob=0.01, max_prob=0.90, increase_step=0.1):
        super().__init__(tokenizer=tokenizer, mlm=True, mlm_probability=initial_prob)
        self.current_prob = initial_prob
        self.max_prob = max_prob
        self.increase_step = increase_step

    def __call__(self, examples, **kwargs):
        batch = self.tokenizer.pad(examples, return_tensors="pt")  # Use PyTorch tensors

        if self.mlm:
            """
            torch_mask_tokens is a method from DataCollatorForLanguageModeling that masks tokens in the input batch.
            For 80% of the data it replaces the token with [MASK], for 10% it replaces it with a random token and for the
            remaining 10% it keeps the original token.
            The Masking probability for 80% of the data is set initial to 15% but with curriculum learning this can be
            increased.
            """
            inputs, labels = self.torch_mask_tokens(batch["input_ids"])


            # Update batch with masked input & labels
            batch["input_ids"], batch["labels"] = inputs, labels

        return batch


    def update_probability(self):
        if self.current_prob < self.max_prob:
            self.current_prob += self.increase_step
            self.mlm_probability = min(self.current_prob, self.max_prob)
            print(f"Updated MLM probability: {self.mlm_probability}")