# -*- coding: utf-8 -*-
r"""
Command Line Interface
=======================
   Commands:
   - train: for Training a new model.
   - interact: Model interactive mode where we can "talk" with a trained model.
   - test: Tests the model ability to rank candidate answers and generate text.
"""
import json
import logging
import os

import bert_score
import click
import sacrebleu
import torch
import yaml
from nltk.tokenize import wordpunct_tokenize
from pytorch_lightning import seed_everything
from tqdm import tqdm

from models import AssistantGPT2, AssistantT5, AssistantMT5, GPT2DataModule, T5DataModule, MT5DataModule 
from trainer import TrainerConfig, build_trainer


@click.group()
def cli():
    pass


@cli.command(name="train")
@click.option(
    "--config",
    "-f",
    type=click.Path(exists=True),
    required=True,
    help="Path to the configure YAML file",
)
def train(config: str) -> None:
    yaml_file = yaml.load(open(config).read(), Loader=yaml.FullLoader)
    # Build Trainer
    train_configs = TrainerConfig(yaml_file)
    seed_everything(train_configs.seed)
    trainer = build_trainer(train_configs.namespace())

    # Build Model
    if yaml_file["model"] == "AssistantGPT2":
        model_config = AssistantGPT2.ModelConfig(yaml_file)
        model = AssistantGPT2(model_config.namespace())
        data = GPT2DataModule(model.hparams, model.tokenizer)
    elif yaml_file["model"] == "AssistantT5":
        model_config = AssistantT5.ModelConfig(yaml_file)
        model = AssistantT5(model_config.namespace())
        data = T5DataModule(model.hparams, model.tokenizer)
    elif yaml_file["model"] == "AssistantMT5":
        model_config = AssistantMT5.ModelConfig(yaml_file)
        model = AssistantMT5(model_config.namespace())
        data = MT5DataModule(model.hparams, model.tokenizer)
    else:
        Exception("Invalid model: {}".format(yaml_file["model"]))

    trainer.fit(model, data)



@cli.command(name="test")
@click.option(
    "--experiment",
    type=click.Path(exists=True),
    required=True,
    help="Path to the experiment folder containing the checkpoint we want to interact with.",
)
@click.option(
    "--test_set",
    type=click.Path(exists=True),
    required=True,
    help="Path to the json file containing the testset.",
)
@click.option(
    "--cuda/--cpu",
    default=True,
    help="Flag that either runs inference on cuda or in cpu.",
    show_default=True,
)
@click.option(
    "--seed",
    default=12,
    help="Seed value used during inference. This influences results only when using sampling.",
    type=int,
)
@click.option(
    "--sample/--search",
    default=True,
    help="Flag that either runs Nucleus-Sampling or Beam search.",
    show_default=True,
)
@click.option(
    "--top_p",
    default=0.9,
    help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)",
    type=float,
)
@click.option(
    "--temperature",
    default=0.9,
    help="Use temperature to decrease the sensitivity to low probability candidates when sampling.",
    type=float,
)
@click.option(
    "--num_beams",
    default=5,
    help="Number of beams during search.",
    type=int,
)
@click.option(
    "--to_json",
    default=False,
    help="Creates and exports model predictions to a JSON file.",
    show_default=True,
)
def test(
    experiment: str,
    test_set: str,
    cuda: bool,
    seed: int,
    sample: bool,
    top_p: float,
    temperature: float,
    num_beams: int,
    to_json: str,
) -> None:
    """Testing function where a trained model is tested in its ability to rank candidate
    answers and produce replies.
    """
    logging.disable(logging.WARNING)
    hparams_file = os.path.join(experiment, "hparams.yaml")
    hparams = yaml.load(open(hparams_file).read(), Loader=yaml.FullLoader)

    if "GPT2" in hparams["model"]:
        model_class = AssistantGPT2
        data_class = GPT2DataModule
    elif "MT5" in hparams["model"]:
        model_class = AssistantMT5
        data_class = MT5DataModule
    elif "T5" in hparams["model"]:
        model_class = AssistantT5
        data_class = T5DataModule
    else:
        Exception("Invalid model: {}".format(hparams["model"]))

    model = model_class.from_experiment(experiment)
    data_module = data_class(model.hparams, model.tokenizer)

    seed_everything(seed)
    cuda = cuda and torch.cuda.is_available()
    if cuda:
        model.to("cuda")

    with open(test_set, "r", encoding="utf-8") as f:
        dataset = json.loads(f.read())

    replies, rankings = [], []
    for dialog in tqdm(dataset, desc="Scoring dialogs...", dynamic_ncols=True):
        # 1) Prepare batch for Ranking:
        batch = data_module.prepare_batch([dialog], train=False)
        if cuda:
            batch = {k: v.cuda() for k, v in batch.items()}

        # 2) Run model to get Multiple Choice Logits:
        model_out = model(**batch)

        # 3) Save produced rankings
        rankings.append(
            {
                "domain": dialog["domain"],
                "history": dialog["history"][-(2 * model.hparams.max_history + 1) :],
                "candidates": dialog["candidates"],
                "ranking": torch.topk(
                    model_out.mc_logits, len(dialog["candidates"])
                ).indices.tolist()[0],
            }
        )

        # 4) Generate answer
        if "GPT2" in hparams["model"]:
            bot_input = batch["input_ids"][0, 0, :].unsqueeze(0)
        else:
            bot_input = batch["encoder_input_ids"][0, 0, :].unsqueeze(0)

        # Nucleus Sampling
        if sample:
            bot_reply_ids = model.generate(
                input_ids=bot_input,
                token_type_ids=batch["token_type_ids"][0, 0, :].unsqueeze(0)
                if "GPT2" in hparams["model"]
                else None,
                max_length=400,
                do_sample=True,
                top_p=top_p,
                temperature=0.7,
            )
        # Beam Search
        else:
            bot_reply_ids = model.generate(
                input_ids=bot_input,
                token_type_ids=batch["token_type_ids"][0, 0, :].unsqueeze(0)
                if "GPT2" in hparams["model"]
                else None,
                max_length=400,
                num_beams=num_beams,
                no_repeat_ngram_size=2,
                early_stopping=True,
            )
        bot_reply = model.tokenizer.decode(bot_reply_ids[0], skip_special_tokens=True)
        # Save generated replies
        replies.append(
            {
                "domain": dialog["domain"],
                "history": dialog["history"][-(2 * model.hparams.max_history + 1) :],
                "bot": bot_reply,
                "human": dialog["candidates"][-1],
            }
        )

    # 6) Runs Ranking Metrics
    hits_1, hits_5, hits_10 = [], [], []
    for ranks in rankings:
        hits_1.append((len(ranks["candidates"]) - 1) in ranks["ranking"][:1])
        hits_5.append((len(ranks["candidates"]) - 1) in ranks["ranking"][:5])
        hits_10.append((len(ranks["candidates"]) - 1) in ranks["ranking"][:10])

    click.secho("Hits@1: {}".format(sum(hits_1) / len(hits_1)), fg="yellow")
    click.secho("Hits@5: {}".format(sum(hits_5) / len(hits_5)), fg="yellow")
    click.secho("Hits@10: {}".format(sum(hits_10) / len(hits_10)), fg="yellow")

    # 7) Runs Generation Metrics
    refs = [[s["human"] for s in replies]]
    sys = [s["bot"] for s in replies]

    bleu = sacrebleu.corpus_bleu(sys, refs, lowercase=True, tokenize="intl").score
    click.secho(f"BLEU: {bleu}", fg="blue")
    ter = sacrebleu.corpus_ter(sys, refs, no_punct=True).score
    click.secho(f"TER: {ter}", fg="blue")

    # BERTScore returns precison, recall, f1.. we will use F1
    bertscore = float(
        bert_score.score(
            cands=sys,
            refs=refs[0],
            lang="en",
            verbose=False,
            nthreads=4,
        )[2].mean()
    )
    click.secho(f"BERTScore: {bertscore}", fg="blue")

    # 8) Saves results.
    if isinstance(to_json, str):
        data = {
            "results": {
                "BLEU": bleu,
                "TER": ter,
                "BERTScore": bertscore,
                "Hits@1": hits_1,
                "Hits@5": hits_5,
                "Hits@10": hits_10,
            },
            "generation": replies,
            "ranking": rankings,
        }
        with open(to_json, "w") as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=4)
        click.secho(f"Predictions saved in: {to_json}.", fg="yellow")


if __name__ == "__main__":
    cli()
