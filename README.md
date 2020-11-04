# bert-crf-entity-recognition-pytorch
This repository is for the entity recognition task using the pre-trained **BERT**[[1]](#1) and the additional **CRF**(Conditional Random Field)[[2]](#2) layer.

Originally, this project has been conducted for dialogue datasets, so it contains both <u>single-turn</u> setting and <u>multi-turn</u> setting.

The single-turn setting is the same as the basic entity extraction task, but the multi-turn one is a little bit different since it considers the dialogue contexts(previous histories) to conduct the entity recognition task to current utterance.

The multi-turn context application is based on **ReCoSa**(the Relevant Contexts with Self-attention)[[3]](#3) structure.

You can see the details of each model in below descriptions.

<img src="https://user-images.githubusercontent.com/16731987/97841292-d981ec80-1d28-11eb-9e85-5275a8412b7c.png" alt="The structure of BERT+CRF entity extraction model in the single-turn / multi-turn setting.">

<br/>

---

### Configuarations

You can set various arguments by modifying `config.json` in top directory.

The description of each variable is as follows. (Those not introduced in below table are set automatically and should not be changed.)

| Argument              | Type     | Description                                                  | Default                |
| --------------------- | -------- | ------------------------------------------------------------ | ---------------------- |
| `turn_type` | `String` | The turn type setting. (`"single"` or `"multi"`) | `"single"` |
| `sentence_embedding` | `String` | The sentence embedding policy when using the multi-turn setting. (`"cls"`: Using [CLS] token. `"max"`: Max pooling. `"Mean"`: Mean pooling.) | `"cls"` |
| `data_dir`            | `String` | The name of the parent directory where data files are stored. | `"data"`               |
| `original_dir`        | `String` | The name of the directory under `data_dir` which contains the original data files before pre-processing. | `"original"`           |
| `entity_dir`          | `String` | The name of the directory under `data_dir` which contains the processed data with inputs & labels. | `"entity"`             |
| `utter_split`         | `String` | The string symbol for splitting each utterance in one dialogue. | `"[END OF UTTERANCE]"` |
| `dialogue_split_line` | `String` | The line for splitting each dialogue in the preprocessed data files. | `"[END OF DIALOGUE]`"  |
| `train_frac`          | `Number`(`float`) | The ratio of the conversations to be included in the train set. | `0.8`                  |
| `valid_frac`          | `Number`(`float`) | The ratio of the conversations to be included in the validation set. (The remaining portion except the train set and the validation set would become the test set.) | `0.1`                  |
| `train_name`          | `String` | The prefix of the train data files' name.                    | `"train"`              |
| `valid_name`          | `String` | The prefix of the validation data files' name.               | `"valid"`              |
| `test_name`           | `String` | The prefix of the test data files' name.                     | `"test"`               |
| `tags_name`           | `String` | The prefix of the dictionary which has all class names & ids. | `"class_dict"`           |
| `outer_split_symbol`  | `String` | The symbol splitting each entity information in one utterance. | `"$$$"`                |
| `inner_split_symbol`  | `String` | The symbol splitting the entity name and the tag in one entity. | `"$$"`                 |
| `max_len`             | `Number`(`int`) | The maximum length of a sentence.                            | `128`                  |
| `max_time`            | `Number`(`int`) | The maximum length of the dialogue history to be attended in the multi-turn setting. | `10`                   |
| `pad_token`           | `String` | The padding token.                                           | `"[PAD]"`              |
| `cls_token`           | `String` | The CLS token for BERT.                                      | `"[CLS]"`              |
| `sep_token`           | `String` | The SEP token for BERT.                                      | `"[SEP]"`              |
| `speaker1_token`      | `String` | The token indicating the speaker 1.                          | `"[ASSISTANT]"`        |
| `speaker2_token`      | `String` | The token indicating the speaker 2.                          | `"[USER]"`             |
| `o_tag`               | `String` | The label indicating the outer entity.                       | `"O"`                  |
| `bert_name`           | `String` | The BERT model type.                                         | `"bert-base-cased"`    |
| `dropout`             | `Number`(`float`) | The dropout rate.                                            | `0.1`                  |
| `context_d_ff`        | `Number`(`int`) | The size of intermediate hidden states in the feed-forward layer. | `2048`                 |
| `context_num_heads`   | `Number`(`int`) | The number of heads for Multi-head attention.                | `8`                    |
| `context_dropout`     | `Number`(`int`) | The dropout rate for the context encoder.                    | `0.1`                  |
| `context_num_layers`  | `Number`(`int`) | The number of layers in the context encoder.                 | `2`                    |
| `ckpt_dir`            | `String` | The path for saved checkpoints.                              | `"saved_models"`       |
| `device`              | `String` | The device type. (`"cuda"` or `"cpu"`) If this is set to `"cuda"`, then the device configuration is set to `torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')`. If this variable is `"cpu"`, then the setting becomes just `torch.devcie('cpu')`. | `"cuda"`               |
| `num_epochs`          | `Number`(`int`) | The total number of iterations.                              | `10`                   |
| `batch_size`          | `Number`(`int`) | The batch size.                                              | `8`                    |
| `learning_rate`       | `Number`(`float`) | The learning rate.                                           | `5e-5`                 |

<br/>

<hr style="background: transparent; border: 0.5px dashed;"/>

### Dataset

This repository includes the Google's Taskmaster-2[[4]](#4) dataset which is processed for entity recognition task beforehand.

The `samples` directory has the files for each domain which has the speaker identity, utterances and entities.

If you want to use these samples, you just have to move all files in `samples` to `{data_dir}/{original_dir}`.

<br/>

If you want to use your own data, please make the data files fit into the formats of the samples I mentioned above.

The description of these formats is as follows. (Make sure that all symbols are compatible with the configurations above.)

<img src="https://user-images.githubusercontent.com/16731987/97839217-121fc700-1d25-11eb-8688-0f3b8b2a63be.png" alt="The description for data processing when using dialogue datasets.">

<br/>

The case that you don't want the multi-turn setting(only with the single-turn) is not different since you just make `{dialogue_split_line}` split each utterance line.

Additionally, if you don't need the speaker information, all you need to do is just putting a dummy string into the speaker position.

<br/>

<hr style="background: transparent; border: 0.5px dashed;"/>

### How to run

1. Install all required packages.

   ```shell
   pip install -r requirements.txt
   ```

   <br/>

2. Make the directory `{data_dir}/{original_dir}` and put the sample files or your own data processed like in the previous section.

   In default setting, the structure of whole data directory should be like below.

   - `data`
     - `original`
       - `flights.txt`
       - `food-ordering.txt`
       - `hotels.txt`
       - `movies.txt`
       - `music.txt`
       - `restaurant-search.txt`
       - `sports.txt`

   <br/>

3. Run the data processing codes.

   ```shell
   python src/data_process.py --config_path=PATH_TO_CONFIGURATION_FILE
   ```

   - `--config_path`: This indicates the path to the configuration file. (default: `config.json`)

   <br/>

4. Run the below command to train the model you want.

   ```shell
   python src/main.py --mode='train' --config_path=PATH_TO_CONFIGURATION_FILE --ckpt_name=CHECKPOINT_NAME
   ```

   - `--mode`: You have to specify the mode among two options, `'train'` or `'test'`.
   - `--ckpt_name`: This specify the checkpoint file name. This would be the name of trained checkpoint and you can continue your training with this model in the case of resuming training. If you want to conduct training from the beginning, this parameter should be omitted. When testing, this would be the name of the checkpoint you want to test. (default: `None`)
   
   <br/>

5. After training, you can test your model as follows.

   ```shell
   python src/main.py --mode='test' --config_path=PATH_TO_CONFIGURATION_FILE --ckpt_name=CHECKPOINT_NAME
   ```

<br/>

---

### References

<a id="1">[1]</a> Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. *arXiv preprint arXiv:1810.04805*. ([https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805))

<a id="2">[2]</a> Lafferty, J., McCallum, A., & Pereira, F. C. (2001). Conditional random fields: Probabilistic models for segmenting and labeling sequence data. ([https://repository.upenn.edu/cis_papers/159/](https://repository.upenn.edu/cis_papers/159/))

<a id="3">[3]</a> Zhang, H., Lan, Y., Pang, L., Guo, J., & Cheng, X. (2019). Recosa: Detecting the relevant contexts with self-attention for multi-turn dialogue generation. *arXiv preprint arXiv:1907.05339*. ([https://arxiv.org/abs/1907.05339](https://arxiv.org/abs/1907.05339))

<a id="4">[4]</a> Taskmaster-2 . (2020). ([https://research.google/tools/datasets/taskmaster-2/](https://research.google/tools/datasets/taskmaster-2/))
