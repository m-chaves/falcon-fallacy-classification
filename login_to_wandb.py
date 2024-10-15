# Perform the initial login to store the authentication token
# Run this script once to authenticate the user and store the token in your machine or server
# The token is stored in the ~/.netrc file
# After running this script, you can run the training script several times without the need to authenticate each time
import wandb
wandb.login()