import argparse
import torch
from torch import nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, classification_report
import numpy as np
import pandas as pd
import os
import json
import time
import sys
import random
import wandb

torch.cuda.empty_cache()
wandb.login()

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
else:
    print("CUDA is not available. Using CPU.")
    # exit since there is not GPU support
    # sys.exit(1) 

def parse_arguments():
    parser = argparse.ArgumentParser(description='Get predictions from a dual transformer model')
    parser.add_argument('--model_dir', type=str, required=True, help='Path to the directory containing the model. The model should be saved as model.pt')
    parser.add_argument('--model_name_or_path', type=str, required=True, help='Hugging face transformer model name')
    parser.add_argument('--test_file', type=str, default='./datasets/train_val_test_sets/df_test.csv', help='Path to the test data')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size used during training')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for converting probabilities to binary predictions')
    parser.add_argument('--feature_prefixes', type=str, nargs='*', default=[], help='List of prefixes for the engineered features to include')
    parser.add_argument('--fc_after_features', type=lambda x: (str(x).lower() == 'true'), choices=[True, False], default='False', help='whether to add some fully connected layers after the engineered features to then concatenate with the transformer outputs')
    parser.add_argument('--modified_classification_head', type=lambda x: (str(x).lower() == 'true'), choices=[True, False], default='False', help='whether to add some fully connected layers to the classification head')
    parser.add_argument('--dropout_between_fc', type=lambda x: (str(x).lower() == 'true'), choices=[True, False], default='False', help='whether to add dropout between fully connected layers to the classification head')
    return parser.parse_args()

class CustomDataset(Dataset):
    def __init__(self, texts1, texts2, labels, features, tokenizer, max_length1, max_length2):
        self.texts1 = texts1
        self.texts2 = texts2
        self.labels = labels
        self.features = features
        self.tokenizer = tokenizer
        self.max_length1 = max_length1
        self.max_length2 = max_length2

    def __len__(self):
        return len(self.texts1)

    def __getitem__(self, idx):
        text1 = self.texts1[idx]
        text2 = self.texts2[idx]
        labels = self.labels[idx]
        features = self.features[idx]

        inputs1 = self.tokenizer(text1, truncation=True, padding='max_length', max_length=self.max_length1, return_tensors='pt')
        inputs2 = self.tokenizer(text2, truncation=True, padding='max_length', max_length=self.max_length2, return_tensors='pt')

        input_ids1 = inputs1['input_ids'].squeeze()
        attention_mask1 = inputs1['attention_mask'].squeeze()
        input_ids2 = inputs2['input_ids'].squeeze()
        attention_mask2 = inputs2['attention_mask'].squeeze()

        return input_ids1, attention_mask1, input_ids2, attention_mask2, torch.tensor(features, dtype=torch.float), torch.tensor(labels, dtype=torch.float)

class MultiTransformerModel(nn.Module):
    def __init__(self, num_labels, features_dim):
        super(MultiTransformerModel, self).__init__()
        self.transformer1 = AutoModel.from_pretrained(args.model_name_or_path)
        self.transformer2 = AutoModel.from_pretrained(args.model_name_or_path)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        
        # Use (or not) hidden layers after engineered features
        if args.fc_after_features:
            self.fc_features1 = nn.Linear(features_dim, 64)
            self.fc_features2 = nn.Linear(64, 32)
            features_dim = 32
        
        hidden_size = self.transformer1.config.hidden_size + self.transformer2.config.hidden_size + features_dim
        
        # Define the classification head fully connected layers conditionally
        if args.modified_classification_head:
            if args.dropout_between_fc:
                self.fc1 = nn.Linear(hidden_size, 128)
                self.fc1_dropout = nn.Dropout(0.1)
                self.fc2 = nn.Linear(128, 16)
                self.fc2_dropout = nn.Dropout(0.1)
                self.classifier = nn.Linear(16, num_labels)
            else:
                self.fc1 = nn.Linear(hidden_size, 128)
                self.fc2 = nn.Linear(128, 16)
                self.classifier = nn.Linear(16, num_labels)
        else:
            self.classifier = nn.Linear(hidden_size, num_labels)
        

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2, features):
        # Get the outputs from the transformers
        outputs1 = self.transformer1(input_ids=input_ids1, attention_mask=attention_mask1)
        outputs2 = self.transformer2(input_ids=input_ids2, attention_mask=attention_mask2)

        # Some models do not provide a pooler_output, use last_hidden_state if that is the case
        if hasattr(outputs1, 'last_hidden_state'):
            pooled_output1 = outputs1.last_hidden_state.mean(dim=1)
        else:
            pooled_output1 = outputs1.pooler_output

        if hasattr(outputs2, 'last_hidden_state'):
            pooled_output2 = outputs2.last_hidden_state.mean(dim=1)
        else:
            pooled_output2 = outputs2.pooler_output

        # Apply fully connected layers after engineered features (or not)
        if args.fc_after_features and features.size(1) > 0:
            features = self.relu(self.fc_features1(features))
            features = self.relu(self.fc_features2(features))

        # Concatenate outputs conditionally based on features_dim
        if features.size(1) > 0: 
            concatenated_output = torch.cat((pooled_output1, pooled_output2, features), dim=1)
        else:
            concatenated_output = torch.cat((pooled_output1, pooled_output2), dim=1)
        concatenated_output = self.dropout(concatenated_output)
        
        if args.modified_classification_head:
            if args.dropout_between_fc:
                x = self.relu(self.fc1(concatenated_output))
                x = self.fc1_dropout(x)
                x = self.relu(self.fc2(x))
                x = self.fc2_dropout(x)
                logits = self.classifier(x)
            else:
                x = self.relu(self.fc1(concatenated_output))
                x = self.relu(self.fc2(x))
                logits = self.classifier(x)
        else:
            logits = self.classifier(concatenated_output)

        return logits
    
def main(args):

    print('---------------------------------')
    print('Arguments------------------------')
    print('---------------------------------')
    print(args)

    def predicted_labels_from_logits(logits, threshold=args.threshold):
        """
        Converts logits to binary predictions using a specified threshold.
        
        Args:
            logits (array-like): The raw output from the model before applying any activation function.
            threshold (float): The threshold for converting probabilities to binary predictions.
            
        Returns:
            np.ndarray: Binary predictions based on the specified threshold.
        """
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(logits))
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= threshold)] = 1
        return y_pred

    def multi_label_metrics(logits, labels, threshold=args.threshold):
        y_pred = predicted_labels_from_logits(logits=logits, threshold=threshold)
        y_true = labels
        precision, recall, _, _ = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)

        metrics = {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy
        }
        return metrics
    
    def concatenate_columns(row, columns):
        """
        Concatenate the values of specified columns in a pandas DataFrame row into a single string,
        treating NaN values as empty strings.

        Args:
            row (pandas.Series): A single row from a pandas DataFrame.
            columns (list of str): A list of column names whose values will be concatenated.

        Returns:
            str: A single string containing the concatenated values from the specified columns of the row.
        """
        concatenated = ''
        for column in columns:
            concatenated += f"{row[column] if pd.notna(row[column]) else ''}"
        return concatenated
    
    def evaluate_model(model, dataloader, device, criterion):
        model = model.to(device)
        model.eval()

        total_loss = 0
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids1, attention_mask1, input_ids2, attention_mask2, features, labels = [item.to(device) for item in batch]

                outputs = model(input_ids1, attention_mask1, input_ids2, attention_mask2, features)
                all_logits.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                loss = criterion(outputs, labels)
                total_loss += loss.item()


        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        avg_loss = total_loss / len(dataloader)

        metrics = multi_label_metrics(logits=all_logits, labels=all_labels)
        y_pred = predicted_labels_from_logits(logits=all_logits, threshold=args.threshold)
        return metrics, all_labels, y_pred, avg_loss

    # Load the dataset
    df_test = pd.read_csv(args.test_file)      
    
    # Get context text
    df_test['context'] = df_test.apply(lambda row: concatenate_columns(row, columns=['previous_context','main_tweet','posterior_context']), axis=1)

    # List of fallacies
    fallacies = ['Ad Hominem', 'Appeal to Fear', 'Appeal to Ridicule', 'False Dilemma', 'Hasty Generalization', 'Loaded Language', 'None of the above']

    # Engineered features (if any)
    if args.feature_prefixes:
        print("Using the following engineered features:", args.feature_prefixes)
        feature_columns = []
        for prefix in args.feature_prefixes:
            feature_columns.extend([col for col in df_test.columns if col.startswith(prefix)])
        test_features = df_test[feature_columns].values
        features_dim = test_features.shape[1]
    else:
        # If no features are specified, use features_dim as 0
        test_features = np.empty((len(df_test), 0))
        features_dim = 0


    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Create the custom datasets
    test_dataset = CustomDataset(texts1=df_test['main_tweet'].tolist(), 
                                  texts2=df_test['context'].tolist(),
                                  features=test_features, 
                                  labels=df_test[fallacies].values, 
                                  tokenizer=tokenizer, 
                                  max_length1=128,
                                  max_length2=512)

    # Create the DataLoaders
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Set the device
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the best model
    model = MultiTransformerModel(num_labels=len(fallacies), features_dim=features_dim)
    best_model_path = os.path.join(args.model_dir, 'model.pt')
    model.load_state_dict(torch.load(best_model_path, map_location=torch.device(device)))

    test_metrics, test_labels, test_predictions, _ = evaluate_model(model, test_dataloader, device, criterion=nn.BCEWithLogitsLoss())
    print(test_metrics)

    print(classification_report(y_true=test_labels, y_pred=test_predictions, target_names=fallacies, output_dict=False, zero_division='warn'))

    # Save predictions along with true labels, text, and context
    with open(os.path.join(args.model_dir, 'true_labels_and_predictions.jsonl'), 'w') as jsonl_file:
        for i in range(len(df_test)):
            record = {
                'main_tweet': df_test['main_tweet'].iloc[i],
                'context': df_test['context'].iloc[i],
                'true_labels': test_labels[i].tolist(),  # Convert numpy array to list
                'predictions': test_predictions[i].tolist()  # Convert numpy array to list
            }
            # Write each record as a JSON object on a new line
            jsonl_file.write(json.dumps(record) + '\n')

    print('---------------------------------')
    print('Done!----------------------------')
    print('---------------------------------')

if __name__ == "__main__":
    args = parse_arguments()
    main(args)

