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

   The final annotated dataset can be found in ```datasets/train_val_test_sets```.

   If you wish the replicate the creation of the dataset you will need to download the raw data (ai4media-sample-v2.zip) from https://docs.hpai.cloud/s/dteJywq4xXWJJsX, extract the files and place them in   the ```datasets``` folder. Then check the following notebooks:

   * data_extraction.ipynb contains the code to extract the data from the raw files, perform the data cleaning and graph-based preprocessing.
   * topic_modelling.ipynb contains the code to perform topic modelling on the data and select tweets that should be excluded from the dataset because they are not related to COVID-19 topics.
   * data_for_annotation.ipynb contains the code to prepare the data for annotation.
   * inter_annotator_agreement.ipynb computes the inter-annotator agreement for the annotations.
   * train_val_test.ipynb
   * corpus_statistics_and_analysis.ipynb


2. **Encoder-based Models**

   * train_baseline.py and scheduler_scripts/train_baseline.sh
   * train_context_model.py and scheduler_scripts/train_context_model.sh
   * predict_context_model.py and scheduler_scripts/predict_context_model.sh
   * results_from_models.ipynb

   <!-- The code for the encoder-based models can be found in the ```models``` folder. The models are implemented using PyTorch and Hugging Face's Transformers library.

   * train_model.py contains the code to train the models.
   * evaluate_model.py contains the code to evaluate the models.
   * predict_model.py contains the code to make predictions using the models.

   To train a model, run the following command:
   ```bash
   python train_model.py --model bert --epochs 10 --batch_size 32
   ```
   Available models: `bert`, `xgboost`, `svm`, and more. -->

3. **T5 models**

   * generate_JSONLINES.py and scripts/generate_JSONLINES.sh
   * train_t5_alhindi_model.py and scheduler_scripts/train_t5_alhindi.sh
   * results_from_t5_models.ipynb

4. **Others**

   * login_to_wand.py

5. **Results folder**

   * An example of the results obtained by the models can be found in ```results/models```. Note that the ```model.pt``` files (i.e. the file containing the fine-tunned model) are not included in the repository due to their size.
   * Files starting with ```df``` contain dataframes derived from the data-processing and annotation.
   * Files starting with ```G``` contain graphs. The most important one to consider is ```G_modified_with_attributes.graphml.gz``` which contains 382126 tweets (nodes) and 256269 arcs between as well as attributes (text, creation date, anonymized user ID) for each node.To load it use:
```python
   with gzip.open("results/G.graphml.gz", "rb") as f:
      G = nx.read_graphml(f)
```


<!-- 5. **Dataset Creation**:
   Run the script to generate the dataset:
   ```bash
   python create_dataset.py
   ```

1. **Model Training**:
   Train a classification model on the dataset:
   ```bash
   python train_model.py --model bert --epochs 10 --batch_size 32
   ```
   Available models: `bert`, `xgboost`, `svm`, and more. -->


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
