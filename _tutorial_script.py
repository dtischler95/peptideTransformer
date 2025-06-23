from torch.utils.data import Dataset


class TutorialDataset(Dataset):
    def __init__(self, peptides, tokenizer, max_length=36):
        self.peptides = [' '.join(seq) for seq in peptides]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.peptides)

    def __getitem__(self, idx):
        peptide = self.peptides[idx]

        # Tokenize the peptide sequence
        encoding = self.tokenizer(peptide, padding='max_length', truncation=True, max_length=self.max_length,
                                  return_tensors='pt')

        # Flatten the tensors (remove the batch dimension)
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # For binary classification, return labels; otherwise, ignore
        item = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()
        }

        return item


def compute_metrics(eval_preds) -> dict:
    """
    computes accuracy for mlm task. Ignore -100 labels and calculate accuracy only on predictions of masked tokens

    :param eval_preds: predictions and labels

    :return: dictionary with the metrics
    """
    import numpy as np
    import evaluate

    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_preds

    # Get predicted labels from logits
    preds = np.argmax(predictions, axis=-1)

    # Ignore the -100 labels
    mask = labels != -100  # Create a mask for valid labels
    masked_preds = preds[mask]  # Filter predictions using the mask
    masked_labels = labels[mask]  # Filter labels using the mask

    # Calculate accuracy only on valid predictions
    return {'accuracy': accuracy.compute(predictions=masked_preds.flatten(), references=masked_labels.flatten())["accuracy"]}


def main():


    from transformers import BertForMaskedLM, BertTokenizer, BertConfig
    # Pfad zum Hugging Face model repository
    model_repository_path = "Rostlab/prot_bert_bfd"
    config = BertConfig.from_pretrained(model_repository_path)
    model = BertForMaskedLM.from_pretrained(model_repository_path, config=config)
    tokenizer = BertTokenizer.from_pretrained(model_repository_path)



    # model_inputs = tokenizer("G L P A L I S W I K R K R G G", return_tensors="pt")
    # outputs = model(**model_inputs)
    #
    # # Output_embedding für jede Aminosäure des Peptids
    # last_hidden_states = outputs.last_hidden_state
    #
    # # Repräsentation der Eingabesequenz
    # cls_token = last_hidden_states[:, 0, :]
    #
    # # CLS-Token welches durch eine lineare Schicht und Tanh-Aktivierung verarbeitet wurde
    # pooled_output = outputs.pooler_output



    import yaml
    from transformers import TrainingArguments
    arguments_file = "./src/bert_model/peptideBERT_configs/BERT_config.yaml"
    with open(arguments_file, 'r') as file:
        arguments = yaml.safe_load(file)
    training_args = TrainingArguments(**arguments)




    from transformers import DataCollatorForLanguageModeling

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )


    import pandas as pd
    from sklearn.model_selection import train_test_split

    train_file = "./data/train_data/tutorial.csv"
    df = pd.read_csv(train_file, sep=';')
    df_train, df_eval = train_test_split(df, test_size=0.2, random_state=42)
    train_data = TutorialDataset(df_train['sequence'].tolist(), tokenizer)
    eval_data = TutorialDataset(df_eval['sequence'].tolist(), tokenizer)



    from transformers import Trainer
    trainer = Trainer(
        model=model,  # Hier sollte dein Modell eingefügt werden
        args=training_args,
        train_dataset=train_data,  # Hier sollte dein Trainingsdatensatz eingefügt werden
        eval_dataset=eval_data,  # Hier sollte dein Evaluationsdatensatz eingefügt werden
        data_collator=data_collator, # Hier sollte dein Data Collator eingefügt werden
        compute_metrics=compute_metrics  # Hier sollte deine Metrik-Funktion eingefügt werden
    )
    trainer.train()


if __name__ == '__main__':
    main()
