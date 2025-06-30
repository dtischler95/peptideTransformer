import evaluate
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import BertForMaskedLM, BertTokenizer, BertConfig, DataCollatorForLanguageModeling, Trainer, \
    TrainingArguments, pipeline

"""
Dies ist ein Tutorial-Skript, welches als Beispielhafte Implementierung für das Training eines BERT-Modells auf einem
selbst erstellten Datensatz dient. 
Ziel ist es, den Umgang mit dem Hugging Face Transformers Framework zu demonstrieren und die Grundlagen des Trainings
eines BERT-Modells auf einem spezifischen Datensatz zu vermitteln.
Für eigene Projekte sollte dieses Skript als Vorlage dienen, muss aber an die spezifischen Anforderungen
und den Datensatz angepasst werden.
"""


class TutorialDataset(Dataset):
    """
    Die Dataset-Klasse von PyTorch ist sehr hilfreich, um Daten für das Training und die Evaluation von Modellen
    bereitzustellen. Diese Klasse ist speziell für das Training von BERT-Modellen auf Peptidsequenzen konzipiert.
    Die Methoden `__init__`, `__len__` und `__getitem__` sind essenziell und müssen entsprechend den Anforderungen
    des Modells und des Datensatzes implementiert werden.

    """

    def __init__(self, peptides, tokenizer, max_length=36):
        """
        Initialisiert das Dataset mit Peptidsequenzen, einem Tokenizer und der maximalen Länge der Sequenzen.
        Was hier übergeben wird, hängt stark von der Trainingsaufgabe ab.

        :param peptides: Liste von Peptidsequenzen, die trainiert werden sollen.
        :param tokenizer: Tokenizer, der verwendet wird, um die Peptidsequenzen in Token-IDs umzuwandeln.
        :param max_length: Maximale Länge der Sequenzen, die für das Padding und Truncation verwendet wird.
        """
        self.peptides = [' '.join(seq) for seq in peptides]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """
        `__len__` gibt die Anzahl der Peptide im Datensatz zurück. Dies ist wichtig für die Trainings- und Evaluationsschleifen,
        um zu wissen, wie viele Datenpunkte verarbeitet werden müssen.
        """
        return len(self.peptides)

    def __getitem__(self, idx):
        """
        `__getitem__` sollte hierbei immer den verarbeiteten Input als Dictionary zurückgeben, welches die
        `input_ids`, `attention_mask` und optional `labels` enthält. Außerdem lassen sich hier auch, wenn vorhanden,
        zusätzliche Features wie Konzentrationen oder andere numerische Werte hinzufügen.

        :param idx: Index des Peptids, das abgerufen werden soll.
        """
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
    Beispielhafte berechnung von Metriken für das Modelltraining.
    Diese Funktion kann beliebig in abhängigkeit von der Trainingsaufgabe angepasst werden.

    :param eval_preds: predictions and labels

    :return: dictionary with the metrics
    """

    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_preds

    # Get predicted labels from logits
    preds = np.argmax(predictions, axis=-1)

    # Ignore the -100 labels
    mask = labels != -100  # Create a mask for valid labels
    masked_preds = preds[mask]  # Filter predictions using the mask
    masked_labels = labels[mask]  # Filter labels using the mask

    # Calculate accuracy only on valid predictions
    return {'accuracy': accuracy.compute(predictions=masked_preds.flatten(), references=masked_labels.flatten())[
        "accuracy"]}


def main():
    """
    Simple Implementierung eines Trainings-Skripts für ein BERT-Modell auf einem selbst erstellten Datensatz.
    Dieses Skript demonstriert die grundlegenden Schritte des Trainings, der Evaluation und der Vorhersage mit einem
    BERT-Modell unter Verwendung des Hugging Face Transformers Frameworks.
    Dieses Skript ist als Tutorial gedacht und sollte an die spezifischen Anforderungen und den Datensatz angepasst werden.
    Hierbei sollte nur die Modelklasse und das Datenpreprozessing am stärksten verändert werden müssen.
    Callbacks sind hier nicht implementiert, sollten aber in der Praxis nicht fehlen.
    """
    # Pfad zum Hugging Face model repository
    model_repository_path = "Rostlab/prot_bert_bfd"
    config = BertConfig.from_pretrained(model_repository_path)
    model = BertForMaskedLM.from_pretrained(model_repository_path, config=config)
    tokenizer = BertTokenizer.from_pretrained(model_repository_path)

    arguments_file = "./src/bert_model/peptideBERT_configs/BERT_config.yaml"
    with open(arguments_file, 'r') as file:
        arguments = yaml.safe_load(file)
    training_args = TrainingArguments(**arguments)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    # --------------------------- Data Preparation ---------------------------
    """
    Stärkster Variabler Teil des Skripts. Data Preprocessing und Data Split hängt stark von der Trainingsaufgabe ab.
    """
    train_file = "./data/train_data/tutorial.csv"
    df = pd.read_csv(train_file, sep=';')
    # Data Split und Preprocessing hängt von Trainingsaufgabe ab. Stark individuell!
    df_train, df_eval = train_test_split(df, test_size=0.2, random_state=42)

    train_data = TutorialDataset(df_train['sequence'].tolist(), tokenizer)
    eval_data = TutorialDataset(df_eval['sequence'].tolist(), tokenizer)

    # ------------------------------------------------------------------------

    trainer = Trainer(
        model=model,  # Hier sollte dein Modell eingefügt werden
        args=training_args,  # Hier sollten deine Trainingsargumente eingefügt werden
        train_dataset=train_data,  # Hier sollte dein Trainingsdatensatz eingefügt werden
        eval_dataset=eval_data,  # Hier sollte dein Evaluationsdatensatz eingefügt werden
        data_collator=data_collator,  # Hier sollte dein Data Collator eingefügt werden
        compute_metrics=compute_metrics,  # Hier sollte deine Metrik-Funktion eingefügt werden
        # callbacks=[] # Hier sollten deine Callbacks eingefügt werden *OPTIONAL*
    )
    trainer.train()

    predict_file = "./data/train_data/tutorial_predict.csv"
    test_df = pd.read_csv(predict_file, sep=';')
    test_dataset = TutorialDataset(test_df['sequence'].tolist(), tokenizer)

    preds = trainer.predict(test_dataset=test_dataset)

    # Returned den Index des Tokens mit der höchsten Wahrscheinlichkeit
    pred_token_ids = np.argmax(preds.predictions, axis=-1)

    # Tokenizer benötigt die Token IDs nicht die Logits, wie sie das BERT Modell zurückgibt
    decoded_sequences = [
        tokenizer.decode(token_ids, skip_special_tokens=True)
        for token_ids in pred_token_ids
    ]

    print(decoded_sequences)


def predict():
    """
    Diese Funktion ist ein Beispiel für eine Vorhersagefunktion, die auf der Huggingface Pipeline basiert.
    """

    # Pfad zum Hugging Face model repository
    model_repository_path = "Rostlab/prot_bert_bfd"
    # Vortrainierte Modelle haben idr einen vordefinierten Task, dieser muss dann in der Pipeline nicht angegeben werden.
    # Sonst lässt sich die Pipeline auch mit dem Task-Argument initialisieren, z.B. pipeline(task="fill-mask", model=model_repository_path)
    # wobei der model repository_path hier optional ist.
    pipe = pipeline(model=model_repository_path)
    sequences = ["K A K C [MASK] C", "A L [MASK] V V K"]

    predictions = pipe(sequences)

    print(predictions)


if __name__ == '__main__':
    # main()
    predict()
