import argparse
import pandas as pd
import json
import os

fallacies = ['Ad Hominem', 'Appeal to Fear', 'Appeal to Ridicule', 
             'False Dilemma', 'Hasty Generalization', 'Loaded Language']

fallacy_map = {
    'Ad Hominem': 'Ad Hominem',
    'Appeal to Fear': 'Fear',
    # 'Appeal to Fear': 'Ad Metum',
    'Appeal to Ridicule': 'Ridicule',
    # 'Appeal to Ridicule': 'Ad Ridiculum',
    'False Dilemma': 'False Dilemma',
    'Hasty Generalization': 'Hasty Generalization',
    'Loaded Language': 'Loaded Language'
}

fallacy_definitions = {
    'Ad Hominem': "Attacking the person or some aspect of the person making the argument rather than addressing the argument itself. It can include Abusive Ad Hominem (A pure attack on the character of the opponent. It often takes the form of 'My opponent is not a good person or has a bad trait, therefore his or her argument should not be accepted'), Tu Quoque Ad Hominem (Suggesting that because someone does not practice what they preach, their argument is invalid. It's analogous to a 'You did it first' 'You are just the same' or 'You are just as bad' kind of attack.), Bias Ad Hominem (Implying that the opponent is personally benefiting from his stance in the argument. That is, questioning the impartiality of the arguer), and Name-calling (Using derogatory language or offensive labels against the opponent to undermine an argument instead of addressing the content of the argument.)",
    'Fear': "Eliciting fear to support a claim.",
    'Ridicule': "Presenting an opponent's argument as absurd, ridiculous, or humorous. Mocking the opponent's point of view. A common instance is to use the expression 'That's crazy!' to dismiss an argument.",
    'False Dilemma': "Presenting a situation as having only two alternatives, when in reality there are more options available. It oversimplifies a complex issue by reducing it to only two possible outcomes or choices, often in a way that excludes other possibilities, nuances, or middle-ground.",
    'Hasty Generalization': "Making a broad statement about a group or population based on a limited or unrepresentative sample. It usually follows the form: X is true for A, X is also true for B, therefore, X is true for C, D and E.",
    'Loaded Language': "The use of words and phrases with strong connotations (either positive or negative) to influence an audience and invoke an emotional response."
}

prompts = {
    'with_fallacy_definition': "Given the following tweet and fallacy definitions, which of the fallacies defined below occurs in the tweet? More than one can occur :\n\nDefinitions:\n{fallacy_def_text}\n\nTweet: {main_tweet}",
    'with_fallacy_definition_with_NotA': "Given the following tweet and fallacy definitions, which of the fallacies defined below occurs in the tweet? More than one can occur. (If none of them occur, you can answer 'none of the above') :\n\nDefinitions:\n{fallacy_def_text}\n\nTweet: {main_tweet}",
    'without_fallacy_definition': "Given the tweet below, which of the following fallacies occurs in the tweet: {fallacies}? More than one can occur. \nTweet: {main_tweet}",
    'without_fallacy_definition_with_NotA': "Given the tweet below, which of the following fallacies occurs in the tweet: {fallacies}? More than one can occur. (If none of them occur, you can answer 'none of the above') \nTweet: {main_tweet}"
}

def process_csv_to_jsonlines(csv_file, jsonl_file, prompt_type):
    df = pd.read_csv(csv_file)

    # Precompute the fallacy definition text and fallacies list
    fallacy_def_text = "\n".join([f"{fallacy}: {definition}" for fallacy, definition in fallacy_definitions.items()])
    fallacies_list = ', '.join([fallacy_map[fallacy] for fallacy in fallacies])

    with open(jsonl_file, 'w') as jsonl_fp:
        for _, row in df.iterrows():
            # Format the prompt with the relevant information
            prompt = prompts[prompt_type].format(
                fallacy_def_text=fallacy_def_text, 
                fallacies=fallacies_list, 
                main_tweet=row['main_tweet']
            )

            # Determine the fallacies detected in the tweet
            fallacies_detected = [fallacy_map[fallacy] for fallacy in fallacies if row[fallacy] == 1]
            target = ', '.join(fallacies_detected) if fallacies_detected else "none of the above"

            json_line = json.dumps({
                "translation": {
                    "input": prompt,
                    "target": target,
                    "binary_target": [row[fallacy] for fallacy in fallacies + ['None of the above']]
                }
            })
            jsonl_fp.write(json_line + '\n')


def main():
    parser = argparse.ArgumentParser(description="Convert CSV files to JSONLINES for text classification training.")
    parser.add_argument('--train_csv', type=str, required=True, help="Path to the training CSV file.")
    parser.add_argument('--val_csv', type=str, required=True, help="Path to the validation CSV file.")
    parser.add_argument('--test_csv', type=str, required=True, help="Path to the test CSV file.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the JSONLINES files.")
    parser.add_argument('--prompt_type', type=str, choices=['with_fallacy_definition', 'with_fallacy_definition_with_NotA', 'without_fallacy_definition', 'without_fallacy_definition_with_NotA'], default='without_fallacy_definition', help="Type of prompt to include in the input.")

    args = parser.parse_args()

    # Create a directory to save the JSONLINES files
    dir_name = 'prompt_type_' + args.prompt_type
    output_path = os.path.join(args.output_dir, dir_name)
    os.makedirs(output_path, exist_ok=True)

    datasets = {
        'train': args.train_csv,
        'val': args.val_csv,
        'test': args.test_csv
    }
    
    for split_name, csv_file in datasets.items():
        jsonl_filename = f"{split_name}.jsonl"
        jsonl_path = os.path.join(output_path, jsonl_filename)
        process_csv_to_jsonlines(csv_file, jsonl_path, args.prompt_type)

if __name__ == '__main__':
    main()
