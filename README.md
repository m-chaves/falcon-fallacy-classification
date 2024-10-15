## FALCON: A Multi-Label Graph-Based Dataset for Fallacy Classification in the COVID-19 Infodemic

### Description
This repository contains the code and data associated with the paper **"FALCON: A Multi-Label Graph-Based Dataset for Fallacy Classification in the COVID-19 Infodemic."**
In this paper we introduce FALCON, a multi-label, graph-based dataset containing COVID-19-related tweets.
This dataset includes expert annotations for six fallacy types—loaded language, appeal to fear, appeal to ridicule, hasty generalization, ad hominem, and false dilemma—and allows for the detection of multiple fallacies in a single tweet.
The dataset's graph structure enables analysis of the relationships between fallacies and their progression in conversations.
We also evaluate the performance of different large language models (LLMs) on this dataset and propose some transformer-based architectures.

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
   The corresponding graph can be found in G_dataset_sample_with_attributes.pkl or G_dataset_sample_with_attributes.graphml.gz.

   If you wish the replicate the creation of the dataset you will need to download the raw data (ai4media-sample-v2.zip) from https://docs.hpai.cloud/s/dteJywq4xXWJJsX, extract the files and place them in   the ```datasets``` folder. Then check the following notebooks:

   * [data_extraction.ipynb](https://github.com/m-chaves/falcon-fallacy-classification/blob/master/data_extraction.ipynb) contains the code to extract the data from the raw files, perform the data cleaning and graph-based preprocessing.
   * [topic_modelling.ipynb](https://github.com/m-chaves/falcon-fallacy-classification/blob/master/topic_modeling.ipynb) contains the code to perform topic modelling on the data and select tweets that should be excluded from the dataset because they are not related to COVID-19 topics.
   * [data_for_annotation.ipynb](https://github.com/m-chaves/falcon-fallacy-classification/blob/master/data_for_annotations.ipynb) contains the code to prepare the data for annotation.
   * [inter_annotator_agreement.ipynb](https://github.com/m-chaves/falcon-fallacy-classification/blob/master/inter_annotator_agreement.ipynb) computes the inter-annotator agreement for the annotations.
   * [train_val_test.ipynb](https://github.com/m-chaves/falcon-fallacy-classification/blob/master/train_val_test.ipynb) merges the annotations information with the tweets data, add extra engineered features and split the data into train, validation and test sets.
   * [corpus_statistics_and_analysis.ipynb](https://github.com/m-chaves/falcon-fallacy-classification/blob/master/corpus_statistics_and_analysis.ipynb) gathers different statistical analysis of the dataset. In particular, it studies the transitivity of the fallacies in the dataset.


2. **Encoder-based Models**

   * [train_baseline.py](https://github.com/m-chaves/falcon-fallacy-classification/blob/master/train_baseline.py) and scheduler_scripts/train_baseline.sh
   * [train_context_model.py](https://github.com/m-chaves/falcon-fallacy-classification/blob/master/train_context_model.py) and scheduler_scripts/train_context_model.sh
   * [predict_context_model.py](https://github.com/m-chaves/falcon-fallacy-classification/blob/master/predict_context_model.py) and scheduler_scripts/predict_context_model.sh
   * [results_from_models.ipynb](https://github.com/m-chaves/falcon-fallacy-classification/blob/master/results_from_models.ipynb)

3. **T5 models**

   * [generate_JSONLINES.py](https://github.com/m-chaves/falcon-fallacy-classification/blob/master/generate_JSONLINES.py) and scripts/generate_JSONLINES.sh
   * [train_t5_alhindi_model.py](https://github.com/m-chaves/falcon-fallacy-classification/blob/master/train_t5_alhindi_model.py) and scheduler_scripts/train_t5_alhindi.sh
   * [results_from_t5_models.ipynb](https://github.com/m-chaves/falcon-fallacy-classification/blob/master/results_from_t5_models.ipynb)

4. **Results folder**

   * An example of the results obtained by the models can be found in ```results/models```. Note that the ```model.pt``` files (i.e. the file containing the fine-tunned model) are not included in the repository due to their size.
   * Files starting with ```df``` contain dataframes derived from the data-processing and annotation.
   * Files starting with ```G``` contain graphs. The most important one to consider is ```G_modified_with_attributes.graphml.gz``` which contains 382126 tweets (nodes) and 256269 arcs between as well as attributes (text, creation date, anonymized user ID) for each node.To load it use:
      ```python
         with gzip.open("results/G_modified_with_attributes.graphml.gz", "rb") as f:
            G = nx.read_graphml(f)
      ```

4. **Others**

   * [login_to_wand.py](https://github.com/m-chaves/falcon-fallacy-classification/blob/master/login_to_wandb.py)


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
