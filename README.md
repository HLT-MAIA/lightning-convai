# State-of-the-art Conversational AI

This code is based on the [transfer-learning-conv-ai](https://github.com/huggingface/transfer-learning-conv-ai) repo from Hugging Face. Please check the accompanying blog post [How to build a State-of-the-Art Conversational AI with Transfer Learning](https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313).

The major difference is that we use [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/0.9.0/) instead of [Ignite](https://pytorch.org/ignite/) and a more "up to date" version of [Transformers](https://huggingface.co/transformers/). We also made an effort to make everything well documented and "easy" to understand.

# Model Architecture

<div style="text-align:center"><img src="resources/convai_model.png" alt="architecture"></div>

Our model is built on top of a pretrained GPT2 model and its is trained in a multi-task setting where we minimize the following losses:
- Language modeling: we project the hidden-state on the word embedding matrix to get logits and apply a cross-entropy loss on the portion of the target corresponding to the gold reply (green labels on the above figure).
- Next-sentence prediction: we pass the hidden-state of the last token (the end-of-sequence token) through a linear layer to get a score and apply a cross-entropy loss to classify correctly a gold answer among distractors.


## Install:

```bash
virtualenv -p python3.6 convai-env
source convai-env/bin/activate

git clone https://github.com/ricardorei/lightning-convai
cd ligthning-conv-ai
pip install -r requirements.txt
```

## Command Line Interface:

### Train:

To set up your training you have to define your model configs. Take a look at the `example.yaml` in the configs folder, where all hyperparameters are briefly described.

After defining your hyperparameter run the following command:
```bash
python cli.py train -f configs/example.yaml
```

### Monitor training with Tensorboard:
Launch tensorboard with:

```
tensorboard --logdir="experiments/"
```

### Test:

To test your model ability to rank candidate answers and reply to user questions just run the following command:

```bash
python cli.py test --experiment experiments/{experiment_id}/ --test_set data/personachat_val.json
```

where `experiment_id` is the name of the experiment folder containing the model you want to test.

```
Options:
  --experiment PATH    Path to the experiment folder containing the checkpoint
                       we want to interact with.  [required]

  --test_set PATH      Path to the json file containing the testset.
                       [required]

  --cuda / --cpu       Flag that either runs inference on cuda or in cpu.
                       [default: True]

  --seed INTEGER       Seed value used during inference. This influences
                       results only when using sampling.

  --sample / --search  Flag that either runs Nucleus-Sampling or Beam search.
                       [default: True]

  --top_p FLOAT        Nucleus filtering (top-p) before sampling (<=0.0: no
                       filtering)

  --temperature FLOAT  Use temperature to decrease the sensitivity to low
                       probability candidates when sampling.

  --num_beams INTEGER  Number of beams during search.
  --to_json TEXT       Creates and exports model predictions to a JSON file.
                       [default: False]

  --help               Show this message and exit.
```


## Interact:
Fun command where we can interact with with a trained model that impersonates a Vegan that likes cooking and radical activities such as sky-diving.

```bash
python cli.py interact --experiment experiments/{experiment_id}/
```

## Benchmarks:

Training with the `example.yaml` config should result in the following:

| Metric  | Result |
| :-----: | :----: |
| Hits@1↑ | 0.8023 |
| Hits@5↑ | 0.9721 |
| Hits@10↑ | 0.9948 |
| BLEU↑ | 2.7799 |
| TER↓ | 1.0497 |
| BERTScore↑ | 0.8548 |

### Code Style:
All the code follows the same style we use Black.

