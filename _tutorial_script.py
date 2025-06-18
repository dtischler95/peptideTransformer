from transformers import BertModel, BertTokenizer, BertConfig

# Pfad zum Hugging Face model repository
model_repository_path = "Rostlab/prot_bert_bfd"

config = BertConfig.from_pretrained(model_repository_path)
model = BertModel.from_pretrained(model_repository_path, config=config)
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



import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

train_file = "./data/train_data/tutorial.csv"
df = pd.read_csv(train_file, sep=';')
encodings = tokenizer(df['sequence'].tolist(), padding=True, truncation=True, max_length=36, return_tensors='pt')
train_data, eval_data = train_test_split(encodings, test_size=0.2, random_state=42)
train_dataset = Dataset.from_dict({
    'input_ids': train_data['input_ids'],
    'attention_mask': train_data['attention_mask'],
    'labels': df['label'].tolist()[:len(train_data['input_ids'])]  # Assuming labels are in the same order
})









from transformers import Trainer

trainer = Trainer(
    model=model,  # Hier sollte dein Modell eingefügt werden
    args=training_args,
    train_dataset=train_data,  # Hier sollte dein Trainingsdatensatz eingefügt werden
    eval_dataset=eval_data,  # Hier sollte dein Evaluationsdatensatz eingefügt werden
)

