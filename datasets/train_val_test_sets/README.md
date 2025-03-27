# FALCON dataset (Fallacies in COVID-19 Network-based)

FALCON is a multi-label, graph-based dataset containing COVID-19-related tweets.
This dataset includes expert annotations for six fallacy types—loaded language, appeal to fear, appeal to ridicule, hasty generalization, ad hominem, and false dilemma—and allows for the detection of multiple fallacies in a single tweet.
The dataset's graph structure enables analysis of the relationships between fallacies and their progression in conversations.


# Train Validation Test Sets

The annotated dataset can be found at [```datasets/train_val_test_sets```](https://github.com/m-chaves/falcon-fallacy-classification/tree/master/datasets/train_val_test_sets).
The dataset was split by components of the graph, with 60% allocated to training, 20% to validation, and 20% to testing.
The dataset contains 2916 annotated tweets.

# Graphs

The dataset graph can be found in [G_dataset_sample_with_attributes.pkl](https://github.com/m-chaves/falcon-fallacy-classification/blob/master/results/G_dataset_sample_with_attributes.pkl) or [G_dataset_sample_with_attributes.graphml.gz](https://github.com/m-chaves/falcon-fallacy-classification/blob/master/results/G_dataset_sample_with_attributes.graphml.gz).
Both files contain the same information, but in different formats.
You can load the graph using:
```python
        with gzip.open("results/G_dataset_sample_with_attributes.graphml.gz", "rb") as f:
        G = nx.read_graphml(f)
```

Additionally, we provide the larger graph ```G_modified_with_attributes.graphml.gz``` which contains 382126 tweets (nodes) and 256269 arcs as well as attributes (text, creation date, anonymized user ID) for each node.
Note that this larger graph is not annotated with fallacies.


# Variables

The annotated dataset contains the following variables:

* **new\_id**: our unique identifier for each tweet. This identifier is different from the tweet's ID provided by the Twitter API. It was created after the removal of duplicates.
* **component\_id**: unique identifier for each component in the graph. This variables helps to identify the conversation or thread to which the tweet belongs.
* **main\_tweet**: the text of the tweet that was annotated.
* **previous\_context**: the text of the context tweets that preceded the main tweet. That is its in-neighbors of order 1 or 2.
* **posterior\_context**: the text of the context tweets that followed the main tweet. That is its out-neighbors of order 1 or 2.
* **Ad Hominem**, * **Appeal to Fear**, * **Appeal to Ridicule**, * **False Dilemma**, * **Hasty Generalization**, * **Loaded Language**, * **None of the above**: Binary variables indicating the presence of each class in the main tweet.
* **created\_at**: the time when the \textit{main tweet} was posted.
* **followers**: number of followers of the user who posted the \textit{main tweet}.
* **tweet\_count**: number of tweets the user who posted the \textit{main tweet} had posted.
* **hashtags**: the hashtags present in the \textit{main tweet}.
* **cashtags**: the cashtags present in the \textit{main tweet}.
* **mentions**: the mentions present in the \textit{main tweet}.
* **retweet\_count**: number of retweets of the \textit{main tweet}.
* **reply\_count**: number of replies to the \textit{main tweet}.
* **like\_count**: number of likes of the \textit{main tweet}.
* **quote\_count**: number of quotes of the \textit{main tweet}.
* **emojis**: the emojis present in the \textit{main tweet}.
Variables with the prefix * **hashtags\_**: binary variables indicating the presence of a specific hashtag in the \textit{main tweet}. For instance **hashtags_ivermectin** indicates the presence of the hashtag "ivermectin." Only the most frequent hashtags were included in this list.
Variables with the prefix * **mentions\_**: binary variables indicating the presence of a specific mention in the \textit{main tweet}. For instance **mentions_user1** indicates the presence of the mention "user1." Only the most frequent mentions were included in this list.
Variables with the prefix * **emojis\_**: binary variables indicating the presence of a specific emoji in the \textit{main tweet}.
* **VADER\_neg**, * **VADER\_neu**, * **VADER\_pos**, * **VADER\_compound**: the negative, neutral, positive, and compound sentiment scores of the \textit{main tweet} calculated by the VADER sentiment analysis tool.
* **VAD\_valence**, * **VAD\_arousal**, * **VAD\_dominance**: the valence, arousal, and dominance scores of the \textit{main tweet} calculated with the NRC-VAD lexicon. We averaged the scores of all words in the tweet to obtain these values.
Variables with the prefix * **POS\_**: discrete variables indicating the number of words with a specific part-of-speech tag in the \textit{main tweet}. For instance, * **POS\_NOUN** indicates the number of nouns in the tweet. The part-of-speech tags were obtained using the spaCy library.
