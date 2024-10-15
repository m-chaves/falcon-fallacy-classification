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
    sys.exit(1) 

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a baseline transformer model')
    parser.add_argument('--model_name_or_path', type=str, required=True, help='Hugging face transformer model name')
    parser.add_argument('--train_file', type=str, default='./datasets/train_val_test_sets/df_train.csv', help='Path to the training data')
    parser.add_argument('--val_file', type=str, default='./datasets/train_val_test_sets/df_val.csv', help='Path to the validation data')
    parser.add_argument('--test_file', type=str, default='./datasets/train_val_test_sets/df_test.csv', help='Path to the test data')
    parser.add_argument('--results_dir', type=str, default='./results/models/', help='Path to the output directory')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--seed', type=int, default=1105, help='Random seed')
    parser.add_argument('--num_runs', type=int, default=3, help='Number of runs')
    parser.add_argument('--main_metric', type=str, default='f1', help='Main metric to monitor performance')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for converting probabilities to binary predictions')
    parser.add_argument('--wandb_project', type=str, default='twitter_fallacy_classification', help='Weights and Biases project name')
    # parser.add_argument('--wandb_log_model', type=str, default='checkpoint', help='Weights and Biases log model')
    return parser.parse_args()

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]

        inputs = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')

        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()

        return input_ids, attention_mask, torch.tensor(labels, dtype=torch.float)

class TransformerModel(nn.Module):
    def __init__(self, num_labels):
        super(TransformerModel, self).__init__()
        self.transformer = AutoModel.from_pretrained(args.model_name_or_path)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        hidden_size = self.transformer.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        

    def forward(self, input_ids, attention_mask):
        # Get the outputs from the transformers
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)

        # Some models do not provide a pooler_output, use last_hidden_state if that is the case
        if hasattr(outputs, 'last_hidden_state'):
            pooled_output = outputs.last_hidden_state.mean(dim=1)
        else:
            pooled_output = outputs.pooler_output

        dropout_output = self.dropout(pooled_output)
        logits = self.classifier(dropout_output)
        return logits
    
def ensure_directory_exists(directory_path):
    """
    Ensures that the specified directory exists.
    If the directory does not exist, it is created.

    :param directory_path: Path to the directory to check/create.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
                input_ids, attention_mask, labels = [item.to(device) for item in batch]

                outputs = model(input_ids, attention_mask)
                all_logits.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                loss = criterion(outputs, labels)
                total_loss += loss.item()

            # total_loss += loss.item()

        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        avg_loss = total_loss / len(dataloader)

        metrics = multi_label_metrics(logits=all_logits, labels=all_labels)
        y_pred = predicted_labels_from_logits(logits=all_logits, threshold=args.threshold)
        return metrics, all_labels, y_pred, avg_loss

    def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, device, num_epochs, best_model_path):
        model = model.to(device)
        criterion = nn.BCEWithLogitsLoss()
        
        best_metric = -1.0
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_f1": [],
            "val_accuracy": [],
            "val_precision": [],
            "val_recall": []
        }

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for batch in train_dataloader:
                input_ids, attention_mask, labels = [item.to(device) for item in batch]

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_dataloader)
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')
            wandb.log({"Epoch": epoch + 1, "Train Loss": avg_loss})

            # Evaluate the model on validation set
            val_metrics, _, _, val_loss = evaluate_model(model, val_dataloader, device, criterion)
            print(f"Epoch {epoch + 1} Validation Metrics: {val_metrics}")
            wandb.log({"Epoch": epoch + 1, "Validation F1": val_metrics['f1'], "Validation Precision": val_metrics['precision'], "Validation Recall": val_metrics['recall'], "Validation Accuracy": val_metrics['accuracy'], "Validation Loss": val_loss})

            # Save the historic metrics
            history["train_loss"].append(avg_loss)
            history["val_loss"].append(val_loss)
            history["val_f1"].append(val_metrics['f1'])
            history["val_accuracy"].append(val_metrics['accuracy'])
            history["val_precision"].append(val_metrics['precision'])
            history["val_recall"].append(val_metrics['recall'])

            # Save the best model
            if val_metrics[args.main_metric] > best_metric:
                best_metric = val_metrics[args.main_metric]
                torch.save(model.state_dict(), best_model_path+'/model.pt')
                print(f"Best model saved with F1 score: {best_metric:.4f}")
                wandb.save(best_model_path+'/model.pt')
        
        # Save the history to json
        with open(best_model_path + "/history.json", 'w') as json_file:
            json.dump(history, json_file, indent=4)


    # Load the datasets
    # UNCOMMENT LATER!!!!
    df_train = pd.read_csv(args.train_file)  
    df_val = pd.read_csv(args.val_file)     
    df_test = pd.read_csv(args.test_file)      
    
    # # DELETE LATER!!!!
    # df_train = pd.read_csv(args.train_file).sample(n=50, random_state=42)  
    # df_val = pd.read_csv(args.val_file).sample(n=10, random_state=42)       
    # df_test = pd.read_csv(args.test_file).sample(n=10, random_state=42)  

    # List of fallacies
    fallacies = ['Ad Hominem', 'Appeal to Fear', 'Appeal to Ridicule', 'False Dilemma', 'Hasty Generalization', 'Loaded Language', 'None of the above']

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Create the custom datasets
    train_dataset = CustomDataset(texts=df_train['main_tweet'].tolist(), 
                                  labels=df_train[fallacies].values, 
                                  tokenizer=tokenizer, 
                                  max_length=128)
    val_dataset = CustomDataset(texts=df_val['main_tweet'].tolist(), 
                                  labels=df_val[fallacies].values, 
                                  tokenizer=tokenizer, 
                                  max_length=128)
    test_dataset = CustomDataset(texts=df_test['main_tweet'].tolist(), 
                                  labels=df_test[fallacies].values, 
                                  tokenizer=tokenizer, 
                                  max_length=128)

    # Create the DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Create the model directory
    my_model_name = args.model_name_or_path.replace('/', '_') + "_finetuned_" + str(args.num_epochs) + "_epochs"
    model_dir = os.path.join(args.results_dir, my_model_name)
    ensure_directory_exists(model_dir)
    
    # Train the model for multiple runs
    for run in range(args.num_runs):

        # Set the device
        torch.cuda.empty_cache()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create the run directory
        run_dir = os.path.join(args.results_dir, my_model_name, f"run_{run + 1}")
        ensure_directory_exists(run_dir)        

        # Set the seed
        print(f"Run {run + 1}/{args.num_runs}")
        run_seed = args.seed + run
        set_seed(run_seed)

        # Create the model
        model = TransformerModel(num_labels=len(fallacies))
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        total_steps = len(train_dataloader) * args.num_epochs  
        
        # Weights and Biases
        wandb.init(project=args.wandb_project, name= my_model_name + '_run' + str(run + 1), reinit=True)
        wandb.watch(model, log="all")
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        # Train the model
        start_time = time.time()
        train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, device, num_epochs=args.num_epochs, best_model_path=run_dir)
        elapsed_time = time.time() - start_time
        print("Train time:", elapsed_time/60, "minutes")

        # Load the best model
        best_model_path = os.path.join(run_dir, 'model.pt')
        model.load_state_dict(torch.load(best_model_path))

        # Evaluate the model on the validation and test sets
        val_metrics, _, _, _ = evaluate_model(model, val_dataloader, device, criterion=nn.BCEWithLogitsLoss())
        print(val_metrics)
        with open(os.path.join(run_dir, "val_metrics.json"), 'w') as json_file:
            json.dump(val_metrics, json_file, indent=4)

        test_metrics, test_labels, test_predictions, _ = evaluate_model(model, test_dataloader, device, criterion=nn.BCEWithLogitsLoss())
        print(test_metrics)
        with open(os.path.join(run_dir, "test_metrics.json"), 'w') as json_file:
            json.dump(test_metrics, json_file, indent=4)
    
        classification_report_dict = classification_report(y_true=test_labels, y_pred=test_predictions, target_names=fallacies, output_dict=True, zero_division='warn')
        classification_report_df = pd.DataFrame(classification_report_dict).transpose()
        classification_report_df.to_csv(os.path.join(run_dir, 'classification_report.csv'), index=True)
        print(classification_report(y_true=test_labels, y_pred=test_predictions, target_names=fallacies, output_dict=False, zero_division='warn'))

        # Finish the run
        wandb.finish()

    print('---------------------------------')
    print('Done!----------------------------')
    print('---------------------------------')

if __name__ == "__main__":
    args = parse_arguments()
    main(args)

