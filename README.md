# bert-crf-entity-extraction-pytorch
This repository is for the entity extraction task using the pre-trained **BERT**[[1]](#1) and the additional **CRF**(Conditional Random Field)[[2]](#2) layer.

Originally, this project has been conducted for dialogue datasets, so it contains both <u>single-turn</u> setting and <u>multi-turn</u> setting.

The single-turn setting is the same as the basic entity extraction task, but the multi-turn one is a little bit different since it considers the dialogue contexts(previous histories) to conduct the entity extraction task to current utterance.

The multi-turn context application is based on **ReCoSa**(the Relevant Contexts with Self-attention)[[3]](#3) structure.

You can see the details of each model in below descriptions.

<img src="https://user-images.githubusercontent.com/16731987/97841292-d981ec80-1d28-11eb-9e85-5275a8412b7c.png" alt="The structure of BERT+CRF entity extraction model in the single-turn / multi-turn setting.">

<br/>

---

### Arguments

**Arguments for data pre-processing**

| Argument              | Type     | Description                                                  | Default                |
| --------------------- | -------- | ------------------------------------------------------------ | ---------------------- |
| `seed`        | `int`   | The random seed.                                             | `0`                   |
| `data_dir`    | `str`   | The parent data directory.                                   | `"data"`              |
| `raw_dir`     | `str`   | The directory which contains the raw data json files.        | `"raw"`               |
| `save_dir`    | `str`   | The directory which will contain the parsed data pickle files. | `"processed"`         |
| `bert_type`   | `str`   | The BERT type to load.                                       | `"bert-base-uncased"` |
| `train_ratio` | `float` | The ratio of train set to the total number of dialogues in each file. | `0.8`                 |

<br/>

**Arguments for training/evaluating**

| Argument             | Type    | Description                                                  | Default               |
| -------------------- | ------- | ------------------------------------------------------------ | --------------------- |
| `seed`               | `int`   | The random seed.                                             | `0`                   |
| `turn_type`          | `str`   | The turn type setting. (`"single"` or `"multi"`)             | *YOU SHOULD SPECIFY*  |
| `bert_type`          | `str`   | The BERT type to load.                                       | `"bert-base-uncased"` |
| `pooling`            | `str`   | The pooling policy when using the multi-turn setting.        | `"cls"`               |
| `data_dir`           | `str`   | The parent data directory.                                   | `"data"`              |
| `processed_dir`      | `str`   | The directory which contains the parsed data pickle files.   | `"processed"`         |
| `ckpt_dir`           | `str`   | The path for saved checkpoints.                              | `"saved_models"`      |
| `gpu`                | `int`   | The index of a GPU to use.                                   | `0`                   |
| `sp1_token`          | `str`   | The speaker1(USER) token.                                    | `"[USR]"`             |
| `sp2_token`          | `str`   | The speaker2(SYSTEM) token.                                  | `"[SYS]"`             |
| `max_len`            | `int`   | The max length of each utterance.                            | `128`                 |
| `max_turns`          | `int`   | The maximum number of the dialogue history to be attended in the multi-turn setting. | `5`                   |
| `dropout`            | `float` | The dropout rate.                                            | `0.1`                 |
| `context_d_ff`       | `int`   | The size of intermediate hidden states in the feed-forward layer. | `2048`                |
| `context_num_heads`  | `int`   | The number of heads for the multi-head attention.            | `8`                   |
| `context_dropout`    | `float` | The dropout rate for the context encoder.                    | `0.1`                 |
| `context_num_layers` | `int`   | The number of layers in the context encoder.                 | `2`                   |
| `learning_rate`      | `float` | The initial learning rate.                                   | `5e-5`                |
| `warmup_ratio`       | `float` | The ratio of warmup steps to the total training steps.       | `0.1`                 |
| `batch_size`         | `int`   | The batch size.                                              | `8`                   |
| `num_workers`        | `int`   | The number of sub-processes for data loading.                | `4`                   |
| `num_epochs`         | `int`   | The number of training epochs.                               | `10`                  |

<br/>

<hr style="background: transparent; border: 0.5px dashed;"/>

### Dataset

This repository uses the Google's Taskmaster-2[[4]](#4) dataset for entity extraction task.

You should first download the data (`"TM-2-2020"`), and get all json files in `"TM-2-2020/data"` directory to properly run this project.

You can see the detailes for using the Taskmaster-2 dataset in the next section.

<br/>

<hr style="background: transparent; border: 0.5px dashed;"/>

### How to run

1. Install all required packages.

   ```shell
   pip install -r requirements.txt
   ```

   <br/>

2. Make the directory `{data_dir}/{raw_dir}` and put the json files, as mentioned in the previous section.

   In default setting, the structure of whole data directory should be like below.

   ```
   data
   └--raw
       └--flight.json
       └--food-ordering.json
       └--hotels.json
       └--movies.json
       └--music.json
       └--restaurant-search.json
       └--sports.json
   ```

   <br/>

3. Run the data processing script.

   ```shell
   sh exec_data_processing.sh
   ```

   After running it, you will get the processed files like below in the default setting.

   ```
   data
   └--raw
       └--flight.json
       └--food-ordering.json
       └--hotels.json
       └--movies.json
       └--music.json
       └--restaurant-search.json
       └--sports.json
   └--processed
       └--class_dict.json
       └--train_tokens.pkl
       └--train_tags.pkl
       └--valid_tokens.pkl
       └--valid_tags.pkl
       └--test_tokens.pkl
       └--test_tags.pkl
   ```

   <br/>

4. Run the main script and check the results.

   ```shell
   sh exec_main.sh
   ```

<br/>

---

### Results

| Turn type | Pooling | Validation F1 | Test F1    |
| --------- | ------- | ------------- | ---------- |
| Single    | -       | 0.6719        | 0.6755     |
| Multi     | CLS     | **0.7148**    | **0.7118** |
| Multi     | Mean    | 0.7132        | 0.7095     |
| Multi     | Max     | 0.7116        | 0.7104     |

<br/>

---

### References

<a id="1">[1]</a> Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. *arXiv preprint arXiv:1810.04805*. ([https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805))

<a id="2">[2]</a> Lafferty, J., McCallum, A., & Pereira, F. C. (2001). Conditional random fields: Probabilistic models for segmenting and labeling sequence data. ([https://repository.upenn.edu/cis_papers/159/](https://repository.upenn.edu/cis_papers/159/))

<a id="3">[3]</a> Zhang, H., Lan, Y., Pang, L., Guo, J., & Cheng, X. (2019). Recosa: Detecting the relevant contexts with self-attention for multi-turn dialogue generation. *arXiv preprint arXiv:1907.05339*. ([https://arxiv.org/abs/1907.05339](https://arxiv.org/abs/1907.05339))

<a id="4">[4]</a> Taskmaster-2 . (2020). ([https://research.google/tools/datasets/taskmaster-2/](https://research.google/tools/datasets/taskmaster-2/))
