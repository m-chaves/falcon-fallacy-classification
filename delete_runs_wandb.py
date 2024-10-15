# Description: This script is used to delete runs from a W&B project based on specific criteria.
# It groups runs by name and deletes runs whose names contain specific substrings (e.g., "1_epochs", "2_epochs", "3_epochs").
# It keeps the most recent run for each group of runs with the same name and deletes the others. 

import wandb
from datetime import datetime

# Authenticate to W&B
wandb.login()
api = wandb.Api()

# Define your project details
entity = "m-chaves-org"  # replace with your W&B entity
project = "twitter_fallacy_classification"  # replace with your W&B project name

# Fetch all runs in the project
runs = api.runs(f"{entity}/{project}")
print('Number of runs available before deletion:', len(runs))

# Group runs by name
runs_by_name = {}

# List of substrings to check for in run names
substr_to_delete = ["1_epochs", "2_epochs", "3_epochs"]

for run in runs:
    if any(substr in run.name for substr in substr_to_delete):
        # Delete runs whose names contain specific substrings
        print(f"Deleting run: {run.name}, Created at: {run.created_at} (matches deletion criteria)")
        run.delete()
    else:
        # Group other runs by name
        if run.name not in runs_by_name:
            runs_by_name[run.name] = []
        runs_by_name[run.name].append(run)

# Iterate over each group of runs with the same name
for run_name, run_list in runs_by_name.items():
    # Sort runs by creation date in descending order
    run_list.sort(key=lambda r: datetime.strptime(r.created_at, "%Y-%m-%dT%H:%M:%S"), reverse=True)
    
    # Keep the most recent run and delete the others
    most_recent_run = run_list[0]
    print(f"Keeping run: {most_recent_run.name}, Created at: {most_recent_run.created_at}")
    
    for run_to_delete in run_list[1:]:
        print(f"Deleting run: {run_to_delete.name}, Created at: {run_to_delete.created_at}")
        run_to_delete.delete()

print('Number of runs available after deletion:', len(runs))

