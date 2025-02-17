---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:156
- loss:MatryoshkaLoss
- loss:MultipleNegativesRankingLoss
base_model: Snowflake/snowflake-arctic-embed-l
widget:
- source_sentence: What significant multi-modal models were released by major vendors
    in 2024?
  sentences:
  - 'The boring yet crucial secret behind good system prompts is test-driven development.
    You don’t write down a system prompt and find ways to test it. You write down
    tests and find a system prompt that passes them.


    It’s become abundantly clear over the course of 2024 that writing good automated
    evals for LLM-powered systems is the skill that’s most needed to build useful
    applications on top of these models. If you have a strong eval suite you can adopt
    new models faster, iterate better and build more reliable and useful product features
    than your competition.

    Vercel’s Malte Ubl:'
  - 'In 2024, almost every significant model vendor released multi-modal models. We
    saw the Claude 3 series from Anthropic in March, Gemini 1.5 Pro in April (images,
    audio and video), then September brought Qwen2-VL and Mistral’s Pixtral 12B and
    Meta’s Llama 3.2 11B and 90B vision models. We got audio input and output from
    OpenAI in October, then November saw SmolVLM from Hugging Face and December saw
    image and video models from Amazon Nova.

    In October I upgraded my LLM CLI tool to support multi-modal models via attachments.
    It now has plugins for a whole collection of different vision models.'
  - 'Those US export regulations on GPUs to China seem to have inspired some very
    effective training optimizations!

    The environmental impact got better

    A welcome result of the increased efficiency of the models—both the hosted ones
    and the ones I can run locally—is that the energy usage and environmental impact
    of running a prompt has dropped enormously over the past couple of years.

    OpenAI themselves are charging 100x less for a prompt compared to the GPT-3 days.
    I have it on good authority that neither Google Gemini nor Amazon Nova (two of
    the least expensive model providers) are running prompts at a loss.'
- source_sentence: How did the construction of railways in the 1800s impact the environment?
  sentences:
  - 'Intuitively, one would expect that systems this powerful would take millions
    of lines of complex code. Instead, it turns out a few hundred lines of Python
    is genuinely enough to train a basic version!

    What matters most is the training  data. You need a lot of data to make these
    things work, and the quantity and quality of the training data appears to be the
    most important factor in how good the resulting model is.

    If you can gather the right data, and afford to pay for the GPUs to train it,
    you can build an LLM.'
  - 'An interesting point of comparison here could be the way railways rolled out
    around the world in the 1800s. Constructing these required enormous investments
    and had a massive environmental impact, and many of the lines that were built
    turned out to be unnecessary—sometimes multiple lines from different companies
    serving the exact same routes!

    The resulting bubbles contributed to several financial crashes, see Wikipedia
    for Panic of 1873, Panic of 1893, Panic of 1901 and the UK’s Railway Mania. They
    left us with a lot of useful infrastructure and a great deal of bankruptcies and
    environmental damage.

    The year of slop'
  - 'OpenAI made GPT-4o free for all users in May, and Claude 3.5 Sonnet was freely
    available from its launch in June. This was a momentus change, because for the
    previous year free users had mostly been restricted to GPT-3.5 level models, meaning
    new users got a very inaccurate mental model of what a capable LLM could actually
    do.

    That era appears to have ended, likely permanently, with OpenAI’s launch of ChatGPT
    Pro. This $200/month subscription service is the only way to access their most
    capable model, o1 Pro.

    Since the trick behind the o1 series (and the future models it will undoubtedly
    inspire) is to expend more compute time to get better results, I don’t think those
    days of free access to the best available models are likely to return.'
- source_sentence: How does the analogy of a broken ASML machine relate to the importance
    of evals, models, and UX in prompts?
  sentences:
  - 'An interesting point of comparison here could be the way railways rolled out
    around the world in the 1800s. Constructing these required enormous investments
    and had a massive environmental impact, and many of the lines that were built
    turned out to be unnecessary—sometimes multiple lines from different companies
    serving the exact same routes!

    The resulting bubbles contributed to several financial crashes, see Wikipedia
    for Panic of 1873, Panic of 1893, Panic of 1901 and the UK’s Railway Mania. They
    left us with a lot of useful infrastructure and a great deal of bankruptcies and
    environmental damage.

    The year of slop'
  - 'When @v0 first came out we were paranoid about protecting the prompt with all
    kinds of pre and post processing complexity.

    We completely pivoted to let it rip. A prompt without the evals, models, and especially
    UX is like getting a broken ASML machine without a manual'
  - 'The environmental impact got much, much worse

    The much bigger problem here is the enormous competitive buildout of the infrastructure
    that is imagined to be necessary for these models in the future.

    Companies like Google, Meta, Microsoft and Amazon are all spending billions of
    dollars rolling out new datacenters, with a very material impact on the electricity
    grid and the environment. There’s even talk of spinning up new nuclear power stations,
    but those can take decades.

    Is this infrastructure necessary? DeepSeek v3’s $6m training cost and the continued
    crash in LLM prices might hint that it’s not. But would you want to be the big
    tech executive that argued NOT to build out this infrastructure only to be proven
    wrong in a few years’ time?'
- source_sentence: Why does the author believe that gullibility may hinder the development
    of AI agents?
  sentences:
  - 'Terminology aside, I remain skeptical as to their utility based, once again,
    on the challenge of gullibility. LLMs believe anything you tell them. Any systems
    that attempts to make meaningful decisions on your behalf will run into the same
    roadblock: how good is a travel agent, or a digital assistant, or even a research
    tool if it can’t distinguish truth from fiction?

    Just the other day Google Search was caught serving up an entirely fake description
    of the non-existant movie “Encanto 2”. It turned out to be summarizing an imagined
    movie listing from a fan fiction wiki.'
  - 'DeepSeek v3 is a huge 685B parameter model—one of the largest openly licensed
    models currently available, significantly bigger than the largest of Meta’s Llama
    series, Llama 3.1 405B.

    Benchmarks put it up there with Claude 3.5 Sonnet. Vibe benchmarks (aka the Chatbot
    Arena) currently rank it 7th, just behind the Gemini 2.0 and OpenAI 4o/o1 models.
    This is by far the highest ranking openly licensed model.

    The really impressive thing about DeepSeek v3 is the training cost. The model
    was trained on 2,788,000 H800 GPU hours at an estimated cost of $5,576,000. Llama
    3.1 405B trained 30,840,000 GPU hours—11x that used by DeepSeek v3, for a model
    that benchmarks slightly worse.'
  - 'A lot of people are excited about AI agents—an infuriatingly vague term that
    seems to be converging on “AI systems that can go away and act on your behalf”.
    We’ve been talking about them all year, but I’ve seen few if any examples of them
    running in production, despite lots of exciting prototypes.

    I think this is because of gullibility.

    Can we solve this? Honestly, I’m beginning to suspect that you can’t fully solve
    gullibility without achieving AGI. So it may be quite a while before those agent
    dreams can really start to come true!

    Code may be the best application

    Over the course of the year, it’s become increasingly clear that writing code
    is one of the things LLMs are most capable of.'
- source_sentence: What are the hardware requirements mentioned for running models
    like GPT-4?
  sentences:
  - 'This remains astonishing to me. I thought a model with the capabilities and output
    quality of GPT-4 needed a datacenter class server with one or more $40,000+ GPUs.

    These models take up enough of my 64GB of RAM that I don’t run them often—they
    don’t leave much room for anything else.

    The fact that they run at all is a testament to the incredible training and inference
    performance gains that we’ve figured out over the past year. It turns out there
    was a lot of low-hanging fruit to be harvested in terms of model efficiency. I
    expect there’s still more to come.'
  - 'The two main categories I see are people who think AI agents are obviously things
    that go and act on your behalf—the travel agent model—and people who think in
    terms of LLMs that have been given access to tools which they can run in a loop
    as part of solving a problem. The term “autonomy” is often thrown into the mix
    too, again without including a clear definition.

    (I also collected 211 definitions on Twitter a few months ago—here they are in
    Datasette Lite—and had gemini-exp-1206 attempt to summarize them.)

    Whatever the term may mean, agents still have that feeling of perpetually “coming
    soon”.'
  - 'So far, I think they’re a net positive. I’ve used them on a personal level to
    improve my productivity (and entertain myself) in all sorts of different ways.
    I think people who learn how to use them effectively can gain a significant boost
    to their quality of life.

    A lot of people are yet to be sold on their value! Some think their negatives
    outweigh their positives, some think they are all hot air, and some even think
    they represent an existential threat to humanity.

    They’re actually quite easy to build

    The most surprising thing we’ve learned about LLMs this year is that they’re actually
    quite easy to build.'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- cosine_accuracy@1
- cosine_accuracy@3
- cosine_accuracy@5
- cosine_accuracy@10
- cosine_precision@1
- cosine_precision@3
- cosine_precision@5
- cosine_precision@10
- cosine_recall@1
- cosine_recall@3
- cosine_recall@5
- cosine_recall@10
- cosine_ndcg@10
- cosine_mrr@10
- cosine_map@100
model-index:
- name: SentenceTransformer based on Snowflake/snowflake-arctic-embed-l
  results:
  - task:
      type: information-retrieval
      name: Information Retrieval
    dataset:
      name: Unknown
      type: unknown
    metrics:
    - type: cosine_accuracy@1
      value: 0.9166666666666666
      name: Cosine Accuracy@1
    - type: cosine_accuracy@3
      value: 1.0
      name: Cosine Accuracy@3
    - type: cosine_accuracy@5
      value: 1.0
      name: Cosine Accuracy@5
    - type: cosine_accuracy@10
      value: 1.0
      name: Cosine Accuracy@10
    - type: cosine_precision@1
      value: 0.9166666666666666
      name: Cosine Precision@1
    - type: cosine_precision@3
      value: 0.3333333333333333
      name: Cosine Precision@3
    - type: cosine_precision@5
      value: 0.20000000000000004
      name: Cosine Precision@5
    - type: cosine_precision@10
      value: 0.10000000000000002
      name: Cosine Precision@10
    - type: cosine_recall@1
      value: 0.9166666666666666
      name: Cosine Recall@1
    - type: cosine_recall@3
      value: 1.0
      name: Cosine Recall@3
    - type: cosine_recall@5
      value: 1.0
      name: Cosine Recall@5
    - type: cosine_recall@10
      value: 1.0
      name: Cosine Recall@10
    - type: cosine_ndcg@10
      value: 0.9692441461309548
      name: Cosine Ndcg@10
    - type: cosine_mrr@10
      value: 0.9583333333333334
      name: Cosine Mrr@10
    - type: cosine_map@100
      value: 0.9583333333333334
      name: Cosine Map@100
---

# SentenceTransformer based on Snowflake/snowflake-arctic-embed-l

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [Snowflake/snowflake-arctic-embed-l](https://huggingface.co/Snowflake/snowflake-arctic-embed-l). It maps sentences & paragraphs to a 1024-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [Snowflake/snowflake-arctic-embed-l](https://huggingface.co/Snowflake/snowflake-arctic-embed-l) <!-- at revision d8fb21ca8d905d2832ee8b96c894d3298964346b -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 1024 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 1024, 'pooling_mode_cls_token': True, 'pooling_mode_mean_tokens': False, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'What are the hardware requirements mentioned for running models like GPT-4?',
    'This remains astonishing to me. I thought a model with the capabilities and output quality of GPT-4 needed a datacenter class server with one or more $40,000+ GPUs.\nThese models take up enough of my 64GB of RAM that I don’t run them often—they don’t leave much room for anything else.\nThe fact that they run at all is a testament to the incredible training and inference performance gains that we’ve figured out over the past year. It turns out there was a lot of low-hanging fruit to be harvested in terms of model efficiency. I expect there’s still more to come.',
    'So far, I think they’re a net positive. I’ve used them on a personal level to improve my productivity (and entertain myself) in all sorts of different ways. I think people who learn how to use them effectively can gain a significant boost to their quality of life.\nA lot of people are yet to be sold on their value! Some think their negatives outweigh their positives, some think they are all hot air, and some even think they represent an existential threat to humanity.\nThey’re actually quite easy to build\nThe most surprising thing we’ve learned about LLMs this year is that they’re actually quite easy to build.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 1024]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Information Retrieval

* Evaluated with [<code>InformationRetrievalEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.InformationRetrievalEvaluator)

| Metric              | Value      |
|:--------------------|:-----------|
| cosine_accuracy@1   | 0.9167     |
| cosine_accuracy@3   | 1.0        |
| cosine_accuracy@5   | 1.0        |
| cosine_accuracy@10  | 1.0        |
| cosine_precision@1  | 0.9167     |
| cosine_precision@3  | 0.3333     |
| cosine_precision@5  | 0.2        |
| cosine_precision@10 | 0.1        |
| cosine_recall@1     | 0.9167     |
| cosine_recall@3     | 1.0        |
| cosine_recall@5     | 1.0        |
| cosine_recall@10    | 1.0        |
| **cosine_ndcg@10**  | **0.9692** |
| cosine_mrr@10       | 0.9583     |
| cosine_map@100      | 0.9583     |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 156 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 156 samples:
  |         | sentence_0                                                                         | sentence_1                                                                           |
  |:--------|:-----------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                               |
  | details | <ul><li>min: 12 tokens</li><li>mean: 20.97 tokens</li><li>max: 36 tokens</li></ul> | <ul><li>min: 43 tokens</li><li>mean: 135.04 tokens</li><li>max: 214 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                         | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
  |:-----------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>What significant advancements in AI were made in 2023, particularly regarding Large Language Models (LLMs)?</code>           | <code>Stuff we figured out about AI in 2023<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>Simon Willison’s Weblog<br>Subscribe<br><br><br><br><br><br><br>Stuff we figured out about AI in 2023<br>31st December 2023<br>2023 was the breakthrough year for Large Language Models (LLMs). I think it’s OK to call these AI—they’re the latest and (currently) most interesting development in the academic field of Artificial Intelligence that dates back to the 1950s.<br>Here’s my attempt to round up the highlights in one place!</code> |
  | <code>How does the development of LLMs in 2023 relate to the historical context of Artificial Intelligence since the 1950s?</code> | <code>Stuff we figured out about AI in 2023<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>Simon Willison’s Weblog<br>Subscribe<br><br><br><br><br><br><br>Stuff we figured out about AI in 2023<br>31st December 2023<br>2023 was the breakthrough year for Large Language Models (LLMs). I think it’s OK to call these AI—they’re the latest and (currently) most interesting development in the academic field of Artificial Intelligence that dates back to the 1950s.<br>Here’s my attempt to round up the highlights in one place!</code> |
  | <code>What are some potential applications of Large Language Models (LLMs) mentioned in the context?</code>                        | <code>Large Language Models<br>They’re actually quite easy to build<br>You can run LLMs on your own devices<br>Hobbyists can build their own fine-tuned models<br>We don’t yet know how to build GPT-4<br>Vibes Based Development<br>LLMs are really smart, and also really, really dumb<br>Gullibility is the biggest unsolved problem<br>Code may be the best application<br>The ethics of this space remain diabolically complex<br>My blog in 2023</code>                                                                                                                           |
* Loss: [<code>MatryoshkaLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss) with these parameters:
  ```json
  {
      "loss": "MultipleNegativesRankingLoss",
      "matryoshka_dims": [
          768,
          512,
          256,
          128,
          64
      ],
      "matryoshka_weights": [
          1,
          1,
          1,
          1,
          1
      ],
      "n_dims_per_step": -1
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 10
- `per_device_eval_batch_size`: 10
- `num_train_epochs`: 5
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 10
- `per_device_eval_batch_size`: 10
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 5
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch | Step | cosine_ndcg@10 |
|:-----:|:----:|:--------------:|
| 1.0   | 16   | 0.9692         |


### Framework Versions
- Python: 3.13.0
- Sentence Transformers: 3.4.1
- Transformers: 4.48.3
- PyTorch: 2.6.0
- Accelerate: 1.3.0
- Datasets: 3.2.0
- Tokenizers: 0.21.0

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MatryoshkaLoss
```bibtex
@misc{kusupati2024matryoshka,
    title={Matryoshka Representation Learning},
    author={Aditya Kusupati and Gantavya Bhatt and Aniket Rege and Matthew Wallingford and Aditya Sinha and Vivek Ramanujan and William Howard-Snyder and Kaifeng Chen and Sham Kakade and Prateek Jain and Ali Farhadi},
    year={2024},
    eprint={2205.13147},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->