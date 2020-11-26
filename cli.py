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

import bert_score
import click
import sacrebleu
import torch
import yaml
from nltk.tokenize import wordpunct_tokenize
from tqdm import tqdm

from model.data_module import DataModule
from model.gpt2 import AssistantGPT2
from pytorch_lightning import seed_everything
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
    model_config = AssistantGPT2.ModelConfig(yaml_file)
    model = AssistantGPT2(model_config.namespace())
    data = DataModule(model.hparams, model.tokenizer)
    trainer.fit(model, data)


@cli.command(name="interact")
@click.option(
    "--experiment",
    type=click.Path(exists=True),
    required=True,
    help="Path to the experiment folder containing the checkpoint we want to interact with.",
)
def interact(experiment: str) -> None:
    """Interactive mode command where we can have a conversation with a trained model
    that impersonates a Vegan that likes cooking and radical activities such as sky-diving.
    """
    logging.disable(logging.WARNING)
    model = AssistantGPT2.from_experiment(experiment)
    click.secho("Hello my name is AssistantGPT2 and i'll pretend that: ", fg="yellow")
    # persona we are going to interact with:
    persona = [
        "i am a vegan and i love hummus.",
        "i love rollercoasters and sky diving.",
        "i do like watching cooking shows.",
        "i am not a good swimmer at all.",
    ]
    persona_ids = [model.tokenizer.encode(s) for s in persona]

    for sentence in persona:
        click.secho(sentence, fg="yellow")
    click.secho("Let's talk:", fg="yellow")

    history = []
    while True:
        raw_text = input(">>> ")
        while not raw_text:
            print("Prompt should not be empty!")
            raw_text = input(">>> ")

        history.append(model.tokenizer.encode(raw_text))
        bot_input = DataModule.build_input(
            tokenizer=model.tokenizer, persona=persona_ids, history=history
        )

        history_ids = model.generate(
            input_ids=torch.LongTensor([bot_input["input_ids"]]),
            token_type_ids=torch.LongTensor([bot_input["token_type_ids"]]),
            max_length=200,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
        )
        bot_reply_ids = history_ids[:, len(bot_input["input_ids"]) :][0]
        bot_reply = model.tokenizer.decode(bot_reply_ids, skip_special_tokens=True)
        print("BOT: {}".format(bot_reply))
        history.append(bot_reply_ids.tolist())


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
    model = AssistantGPT2.from_experiment(experiment)
    seed_everything(seed)

    cuda = cuda and torch.cuda.is_available()
    if cuda:
        model.to("cuda")

    with open(test_set, "r", encoding="utf-8") as f:
        dataset = json.loads(f.read())

    replies, rankings = [], []
    for dialog in tqdm(dataset, desc="Scoring dialogs...", dynamic_ncols=True):
        # 2) Saves Ground-Truth
        ground_truth_reply = dialog["candidates"][-1]

        # 3) Prepares History
        history = dialog["history"][-(2 * model.hparams.max_history + 1) :]
        history_ids = [model.tokenizer.encode(h) for h in history]

        # 4) Rank Candidates in batch:
        batch = []
        for j, candidate in enumerate(dialog["candidates"]):
            candidate_ids = model.tokenizer.encode(candidate)
            instance = DataModule.build_input(
                tokenizer=model.tokenizer,
                domain=persona_ids,
                history=history_ids,
                reply=candidate_ids,
            )
            batch.append(instance)

        # from list of dictionaries to dictionary of lists
        batch = {k: [d[k] for d in batch] for k in batch[0]}
        batch = DataModule.pad_dataset(batch)
        if cuda:
            batch = {k: torch.LongTensor(v).cuda() for k, v in batch.items()}
        else:
            batch = {k: torch.LongTensor(v) for k, v in batch.items()}

        mc_logits = model(**batch).mc_logits
        rankings.append({
            "persona": persona,
            "history": history,
                "candidates": dialog["candidates"],
                "ranking": torch.topk(
                    mc_logits, len(dialog["candidates"])
                ).indices.tolist(),
        })

        # 5) Generates Reply
        bot_input = DataModule.build_input(
            tokenizer=model.tokenizer, persona=persona_ids, history=history_ids
        )
        # Nucleus Sampling
        if sample:
            history_ids = model.generate(
                input_ids=torch.LongTensor([bot_input["input_ids"]]).cuda() if cuda else torch.LongTensor([bot_input["input_ids"]]),
                token_type_ids=torch.LongTensor([bot_input["token_type_ids"]]).cuda() if cuda else torch.LongTensor([bot_input["token_type_ids"]]),        
                max_length=200,
                do_sample=True,
                top_p=top_p,
                temperature=0.7,
            )
            # Beam Search
        else:
            history_ids = model.generate(
                input_ids=torch.LongTensor([bot_input["input_ids"]]).cuda() if cuda else torch.LongTensor([bot_input["input_ids"]]),
                token_type_ids=torch.LongTensor([bot_input["token_type_ids"]]).cuda() if cuda else torch.LongTensor([bot_input["token_type_ids"]]),
                max_length=200,
                num_beams=num_beams,
                no_repeat_ngram_size=2,
                early_stopping=True,
            )
        bot_reply_ids = history_ids[:, len(bot_input["input_ids"]) :][0]
        bot_reply = model.tokenizer.decode(bot_reply_ids, skip_special_tokens=True)

        replies.append({
            "persona": persona,
            "history": history,
            "bot": " ".join(wordpunct_tokenize(bot_reply.lower())),
            "human": ground_truth_reply,
        })

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
