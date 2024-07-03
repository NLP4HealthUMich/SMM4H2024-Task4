# Task 4 LHS712NV @ @SMM4H’24
## System Description
The system leverages the BERT (Bidirectional Encoder Representations from Transformers) model to meet the requirements of the Named Entity Recognition (NER) task. BERT models are highly effective in capturing contextual information from both the left and right contexts in text data. The task involves the identification and classification of pre-annotated Reddit posts3, specifically focusing on clinical and social impacts. The process commences with data preprocessing, where the dataset is divided into training and test sets, followed by tokenization and encoding of labels for model input. The model is then fine-tuned using the Hugging Face Transformers library, with training parameters such as batch size, learning rate, and the number of epochs specified. During training, the metrics precision, recall, F1 score, and accuracy are computed to assess the model's performance. Subsequently, the model is evaluated on a separate validation dataset to gauge its effectiveness. Predictions are generated, and metrics such as precision, recall, F1 score, and accuracy are calculated and displayed, along with a visualization of the confusion matrix. The selection of BERT for this task is underscored by its ability to comprehend the contextual nuances of words, enabling accurate identification of named entities. Fine-tuning pre-trained BERT models often results in enhanced performance on downstream tasks such as NER, making it a popular choice in natural language processing applications. Finally, the trained model is tested on a separate dataset, and the predictions are saved for further analysis or submission.

## Installation

To get started, install the necessary libraries:

```bash
!pip install datasets
!pip install transformers
!pip install seqeval
!pip install transformers[torch]
```

## Data Preparation

### Cleaning the Data

The data is pre-tokenized and saved in a TSV file. We will clean and organize the data into posts and labels.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NUMBER_OF_POSTS = 848

# Import the data
clean_data = []
full_data = []
with open("reddit-impacts-train.tsv", "r") as file:
    for line in file:
        if not line.startswith("#Text"):
            clean_data.append(line)
        else:
            full_data.append(line)

# Create the posts and labels lists
posts = [[] for _ in range(NUMBER_OF_POSTS)]
labels = [[] for _ in range(NUMBER_OF_POSTS)]
for line in clean_data:
    parts = line.split("\t")
    if len(parts) == 6:
        post_number, token_number = parts[0].split("-")
        word = parts[2]
        label = parts[4]
        posts[int(post_number)].append(word)
        labels[int(post_number)].append(label)
```

### Checking Unique Values

Identify the unique labels present in the dataset.

```python
unique_label = set()
for label in labels:
    for value in label:
        unique_label.add(value)

label_list = list(unique_label)
print(label_list)
```

### Label Encoding

Create a dictionary to encode the labels.

```python
label_encoding_dict = {'_': 0, 'Clinical Impacts': 1, 'Social Impacts': 2}
```

### Splitting Data

Split the data into training and test sets.

```python
from sklearn.model_selection import train_test_split

Xt, Xv, yt, yv = train_test_split(posts, labels, test_size=0.20, train_size=0.80)

train_df = pd.DataFrame({'tokens': Xt, 'labels': yt})
test_df = pd.DataFrame({'tokens': Xv, 'labels': yv})

from datasets import Dataset

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
```

## Model Training with Transformers

### Loading Necessary Libraries

```python
import os
import itertools
from datasets import load_metric
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import torch
```

### Setting Parameters

```python
task = "labels"
model_checkpoint = "distilbert-base-uncased"
batch_size = 16

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
```

### Tokenize and Align Labels

```python
def tokenize_and_align_labels(examples):
    label_all_tokens = True
    tokenized_inputs = tokenizer(list(examples["tokens"]), truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"{task}"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif label[word_idx] == '0':
                label_ids.append(0)
            elif word_idx != previous_word_idx:
                label_ids.append(label_encoding_dict[label[word_idx]])
            else:
                label_ids.append(label_encoding_dict[label[word_idx]] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

train_tokenized_datasets = train_dataset.map(tokenize_and_align_labels, batched=True)
test_tokenized_datasets = test_dataset.map(tokenize_and_align_labels, batched=True)
```

### Model Initialization

```python
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))

args = TrainingArguments(
    f"test-{task}",
    evaluation_strategy = "epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=1e-5,
)

data_collator = DataCollatorForTokenClassification(tokenizer)
metric = load_metric("seqeval")
```

### Compute Metrics

```python
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"], "accuracy": results["overall_accuracy"]}
```

### Training the Model

```python
trainer = Trainer(
    model,
    args,
    train_dataset=train_tokenized_datasets,
    eval_dataset=test_tokenized_datasets,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()
trainer.save_model('un-ner.model')
```

## Validation

Load the tokenizer and model, and predict labels for the validation dataset.

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import pandas as pd

NUMBER_OF_POSTS_V = 262

tokenizer = AutoTokenizer.from_pretrained('./un-ner.model/')
model = AutoModelForTokenClassification.from_pretrained('./un-ner.model/', num_labels=len(label_list))

clean_data_test = []
full_data_test = []
with open("reddit-impacts-dev.tsv", "r") as file:
    for line in file:
        if not line.startswith("#Text"):
            clean_data_test.append(line)
        else:
            full_data_test.append(line)

paragraph = [[] for _ in range(NUMBER_OF_POSTS_V)]
index = [[] for _ in range(NUMBER_OF_POSTS_V)]
span = [[] for _ in range(NUMBER_OF_POSTS_V)]
labels_v = [[] for _ in range(NUMBER_OF_POSTS_V)]
for line in clean_data_test:
    parts = line.split("\t")
    if len(parts) == 6:
        post_number, token_number = parts[0].split("-")
        ind = parts[0]
        sp = parts[1]
        word = parts[2]
        label = parts[4]
        paragraph[int(post_number)].append(word)
        index[int(post_number)].append(ind)
        span[int(post_number)].append(sp)
        labels_v[int(post_number)].append(label)

output_lines = []
prediction_label = []
for i, post_tokens in enumerate(paragraph):
    inputs = tokenizer(" ".join(post_tokens), return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()

    post_prediction_label = []
    post_text = " ".join(post_tokens)
    output_lines.append(f"#Text = {post_text}")
    for j, (token, pred_label_id) in enumerate(zip(post_tokens, predictions)):
        pred_label = label_list[pred_label_id] if pred_label_id != 0 else "_"
        has_label = "*" if pred_label_id != 0 else "_"
        current_index = index[i][j]
        current_span = span[i][j]
        output_lines.append(f"{current_index}\t{current_span}\t{token}\t{has_label}\t{pred_label}")
        post_prediction_label.append(pred_label)
    prediction_label.append(post_prediction_label)
    output_lines.append("")

with open("output.tsv", "w") as file:
    for line in output_lines:
        file.write(line + "\n")
```

### Writing Predictions to CSV

```python
import csv

def flatten_list_of_lists(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

flat_predictions = flatten_list_of_lists(prediction_label)
flat_labels = flatten_list_of_lists(labels_v)

def write_csv(predictions, labels, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['Labels', 'Predictions']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for label, prediction in zip(labels, predictions):
            writer.writerow({'Labels': label, 'Predictions': prediction})

filename = 'predictions_labels.csv'
write_csv(flat_predictions, flat_labels, filename)
```

### Performance Evaluation

Evaluate the model's performance using precision, recall, accuracy, F1 score, and confusion