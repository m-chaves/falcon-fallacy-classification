## FALCON: A Multi-Label Graph-Based Dataset for Fallacy Classification in the COVID-19 Infodemic

### Description
This repository contains the code and data associated with the paper **"FALCON: A Multi-Label Graph-Based Dataset for Fallacy Classification in the COVID-19 Infodemic."**
In this paper we introduce FALCON, a multi-label, graph-based dataset containing COVID-19-related tweets.
This dataset includes expert annotations for six fallacy types—loaded language, appeal to fear, appeal to ridicule, hasty generalization, ad hominem, and false dilemma—and allows for the detection of multiple fallacies in a single tweet.
The dataset's graph structure enables analysis of the relationships between fallacies and their progression in conversations.
We also evaluate the performance of different large language models (LLMs) on this dataset and propose some transformer-based architectures.

<!-- Pending to add links to the following -->
<!-- Publish paper | Extended paper (with appendices and additional results) -->

### Installation

1. To clone this repository:
   ```bash
   git clone https://github.com/m-chaves/falcon-fallacy-classification.git
   cd falcon-fallacy-classification
   ```

2. Set up a virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

### Usage

1. **Dataset**

   The final annotated dataset can be found in [```datasets/train_val_test_sets```](https://github.com/m-chaves/falcon-fallacy-classification/tree/master/datasets/train_val_test_sets).
   The corresponding graph can be found in [G_dataset_sample_with_attributes.pkl](https://github.com/m-chaves/falcon-fallacy-classification/blob/master/results/G_dataset_sample_with_attributes.pkl) or [G_dataset_sample_with_attributes.graphml.gz](https://github.com/m-chaves/falcon-fallacy-classification/blob/master/results/G_dataset_sample_with_attributes.graphml.gz). You can load the graph using:
   ```python
         with gzip.open("results/G_dataset_sample_with_attributes.graphml.gz", "rb") as f:
            G = nx.read_graphml(f)
   ```

   If you wish the replicate the creation of the dataset you will need to download the raw data (ai4media-sample-v2.zip) from https://docs.hpai.cloud/s/dteJywq4xXWJJsX, extract the files and place them in   the ```datasets``` folder. Then check the following notebooks:

   * [data_extraction.ipynb](https://github.com/m-chaves/falcon-fallacy-classification/blob/master/data_extraction.ipynb) contains the code to extract the data from the raw files, perform the data cleaning and graph-based preprocessing.
   * [topic_modelling.ipynb](https://github.com/m-chaves/falcon-fallacy-classification/blob/master/topic_modeling.ipynb) contains the code to perform topic modelling on the data and select tweets that should be excluded from the dataset because they are not related to COVID-19 topics.
   * [data_for_annotation.ipynb](https://github.com/m-chaves/falcon-fallacy-classification/blob/master/data_for_annotations.ipynb) contains the code to prepare the data for annotation.
   * [inter_annotator_agreement.ipynb](https://github.com/m-chaves/falcon-fallacy-classification/blob/master/inter_annotator_agreement.ipynb) computes the inter-annotator agreement for the annotations.
   * [train_val_test.ipynb](https://github.com/m-chaves/falcon-fallacy-classification/blob/master/train_val_test.ipynb) merges the annotations information with the tweets data, add extra engineered features and split the data into train, validation and test sets.
   * [corpus_statistics_and_analysis.ipynb](https://github.com/m-chaves/falcon-fallacy-classification/blob/master/corpus_statistics_and_analysis.ipynb) gathers different statistical analysis of the dataset. In particular, it studies the transitivity of the fallacies in the dataset.

2. **Annotation Guidelines**

   The annotation guidelines can be found in the [annotation_guidelines.pdf](https://github.com/m-chaves/falcon-fallacy-classification/blob/master/annotation_guidelines/annotation_guidelines.pdf) file.


3. **Encoder-based Models**

   We fine-tuned several transformer models from Hugging Face’s Transformers library using the FALCON dataset. The task is a multi-label classification problem, where each tweet can be associated with multiple fallacies. The code for training and evaluating these models can be found in the following files:

   * [train_baseline.py](https://github.com/m-chaves/falcon-fallacy-classification/blob/master/train_baseline.py) allows to fine-tune a transformer model. The input of the model is a tweet's text. [train_baseline.sh](https://github.com/m-chaves/falcon-fallacy-classification/blob/master/scheduler_scripts/train_baseline.sh) shows how to pass the arguments to the python script.
   * [train_context_model.py](https://github.com/m-chaves/falcon-fallacy-classification/blob/master/train_context_model.py) allows to fine-tune a dual transformer model, which consist of two instances of the same type of pre-trained transformers, one for processing the main tweet and the other for the context information. The context data consists of the concatenation of the context tweets and the main tweet in chronological order. This model can also take as input the engineered features present on the dataset (POS tags, sentiment scores, emojis, etc). [train_context_model.sh](https://github.com/m-chaves/falcon-fallacy-classification/blob/master/scheduler_scripts/train_context_model.sh) shows how to pass the arguments to the python script.
   * [predict_context_model.py](https://github.com/m-chaves/falcon-fallacy-classification/blob/master/predict_context_model.py) and [predict_context_model.sh](https://github.com/m-chaves/falcon-fallacy-classification/blob/master/scheduler_scripts/predict_context_model.sh) allow to predict the labels of the test set using the fine-tuned context model.
   * [results_from_models.ipynb](https://github.com/m-chaves/falcon-fallacy-classification/blob/master/results_from_models.ipynb) gathers the results from the models and performs a comparison between them. It also studies the error analysis of the best model and conducts statistical tests to understand the effect of different features in model performance.

4. **T5 models**
   We fine-tunned Text-To-Text Transfer Transformer (T5) model using the ```t5-large``` check-point. Specifically, we used the T5 model settings proposed by [Alhindi et al](https://aclanthology.org/2022.emnlp-main.560/).

   * [generate_JSONLINES.py](https://github.com/m-chaves/falcon-fallacy-classification/blob/master/generate_JSONLINES.py) and [generate_JSONLINES.sh](https://github.com/m-chaves/falcon-fallacy-classification/blob/master/scripts/generate_JSONLINES.sh) allow to generate the JSONLINES files needed to train the T5 models. Each file in the JSONLINES includes the prompt and the target for the model.
   * [train_t5_alhindi_model.py](https://github.com/m-chaves/falcon-fallacy-classification/blob/master/train_t5_alhindi_model.py) allows to fine-tune and evaluate the T5 model. See [train_t5_alhindi.sh](https://github.com/m-chaves/falcon-fallacy-classification/blob/master/scheduler_scripts/train_t5_alhindi.sh) for an example of how to pass the arguments to the python script.
   * [results_from_t5_models.ipynb](https://github.com/m-chaves/falcon-fallacy-classification/blob/master/results_from_t5_models.ipynb) gathers the results from the T5 models.

5. **Results folder**

   * An example of the results obtained by the models can be found in ```results/models```. Note that the ```model.pt``` files (i.e. the file containing the fine-tunned model) are not included in the repository due to their size.
   * Files starting with ```df``` contain dataframes derived from the data-processing and annotation.
   * Files starting with ```G``` contain graphs. The most important one to consider is ```G_modified_with_attributes.graphml.gz``` which contains 382126 tweets (nodes) and 256269 arcs between as well as attributes (text, creation date, anonymized user ID) for each node.To load it use:
      ```python
         with gzip.open("results/G_modified_with_attributes.graphml.gz", "rb") as f:
            G = nx.read_graphml(f)
      ```


### Acknowledgements

This work was supported by the European Union's Horizon 2020 research and innovation programme under grant agreement No 951911 ([AI4Media](https://ai4media.eu/)).


<!-- ### Citation
If you use this code or the dataset in your research, please cite the following paper:

```bibtex
@inproceedings{falcon2024,
  title={FALCON: A Multi-Label Graph-Based Dataset for Fallacy Classification in the COVID-19 Infodemic},
  author={Author, Firstname and Coauthor, Secondname},
  booktitle={Proceedings of the Conference},
  year={2024},
  url={https://link-to-paper}
}
``` -->

<!-- ### License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. -->
