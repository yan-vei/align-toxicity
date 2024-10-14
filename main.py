import wandb
import tqdm
import hydra
import json
import torch
from omegaconf import DictConfig, OmegaConf
from settings import settings
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from utils.preprocess import preprocess_hatexplain, preprocess_dataset, preprocess_mixed_irony
from utils.prompt import create_prompt


@hydra.main(config_path='configs', config_name='defaults', version_base=None)
def run_pipeline(cfg: DictConfig):
    """
    Define the run managed by hydra configs
    :param cfg: hydra config
    :return: void
    """
    # Log into wandb if required
    use_wandb = cfg.basic.use_wandb
    if use_wandb:
        cfg_copy = OmegaConf.to_container(cfg, resolve=True)
        wandb.login(key=settings.WANDB_API_KEY)

        project_name = cfg.dataset.name

        wandb.init(project=project_name, name=cfg.basic.wandb_run,
                   config=cfg_copy)

    # Load correct dataset
    dataset_name = cfg.dataset.name
    dataset_path = cfg.dataset.path

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)

    if dataset_name == 'hatexplain':
        examples = preprocess_hatexplain(dataset_path)
    elif dataset_name == 'implicit_hate':
        examples = preprocess_dataset(dataset_path)
    elif dataset_name == 'offenseval_implicit':
        examples = preprocess_dataset(dataset_path)
    elif dataset_name == 'mixed_irony':
        examples = preprocess_mixed_irony(dataset_path)

    if cfg.model.output_folder == 'llama':
        examples_with_labels = run_llama(cfg=cfg, tokenizer=tokenizer, examples=examples, device=device)
    elif cfg.model.output_folder == 'hatebert':
        examples_with_labels = run_hatebert(cfg=cfg, tokenizer=tokenizer, examples=examples, device=device)
    elif cfg.model.output_folder == 'gpt2':
        examples_with_labels = run_gpt2(cfg=cfg, tokenizer=tokenizer, examples=examples, device=device)

    # Save the processed data with labels
    if cfg.basic.is_fewshot:
        with open(cfg.basic.output_dir + "/" + cfg.model.output_folder + "/" + cfg.dataset.name + "_fewshot.json", 'w') as f:
            json.dump(examples_with_labels, f)
    else:
        with open(cfg.basic.output_dir + "/" + cfg.model.output_folder + "/" + cfg.dataset.name + ".json", 'w') as f:
            json.dump(examples_with_labels, f)

    print(f"\t Finished processing the dataset {cfg.dataset.name} with the model {cfg.model.name}.")

    if use_wandb:
        wandb.finish()


def run_llama(cfg, tokenizer, examples, device):
    """
    Run labeling pipeline using Llama model.
    :param cfg: hydra config
    :param examples: dict, unlabeled data
    :param tokenizer: tokenizer
    :return: dict, examples with predicted labels
    """

    hf_kwargs = {"device_map": cfg.basic.device_map, "offload_folder": cfg.basic.offload_folder}
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        use_auth_token=True,
        **hf_kwargs,
    )

    # Create prompts
    prepended_prompt = cfg.prompt.text
    examples_with_prompts = list(map(lambda e: create_prompt(prepended_prompt, e,
                                                             is_fewshot=cfg.basic.is_fewshot), examples))

    # Classify examples
    examples_with_labels = []
    for example in tqdm.tqdm(examples_with_prompts):
        inputs = tokenizer(example['prompt'], return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=cfg.model.max_new_tokens)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        label = result.split('Answer:')[-1].strip()
        example['predicted_label'] = label

        examples_with_labels.append(example)

    return examples_with_labels


def run_hatebert(cfg, tokenizer, examples, device):
    """
    Run labeling pipeline using HateBERT model.
    :param cfg: hydra config
    :param tokenizer: tokenizer
    :param examples: dict, unlabeled data
    :return: dict, examples with predicted labels
    """

    model = AutoModelForSequenceClassification.from_pretrained(cfg.model.name)

    # Classify examples
    examples_with_labels = []
    for example in tqdm.tqdm(examples):
        inputs = tokenizer(example['text'], padding=True, truncation=True, return_tensors="pt").to(device)
        outputs = model(**inputs)
        label = outputs.logits.argmax(dim=-1).item()
        example['predicted_label'] = label

        examples_with_labels.append(example)

    return examples_with_labels


def run_gpt2(cfg, tokenizer, examples, device):
    """
    Run labeling pipeline using GPT-3.5 free tier model.
    :param cfg: hydra config
    :param examples: dict, unlabeled data
    :param tokenizer: tokenizer
    :return: dict, examples with predicted labels
    """

    hf_kwargs = {"device_map": cfg.basic.device_map, "offload_folder": cfg.basic.offload_folder}
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        use_auth_token=True,
        **hf_kwargs,
    )

    # Create prompts
    prepended_prompt = cfg.prompt.text
    examples_with_prompts = list(map(lambda e: create_prompt(prepended_prompt, e,
                                                             is_fewshot=cfg.basic.is_fewshot), examples))

    # Classify examples
    examples_with_labels = []
    for example in tqdm.tqdm(examples_with_prompts):
        inputs = tokenizer(example['prompt'], return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=cfg.model.max_new_tokens, pad_token_id=tokenizer.eos_token_id)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        label = result.split('Answer:')[-1].strip()
        example['predicted_label'] = label

        examples_with_labels.append(example)

    return examples_with_labels


if __name__ == "__main__":
    run_pipeline()
