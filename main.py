import wandb
import tqdm
import hydra
import json
from omegaconf import DictConfig, OmegaConf
from settings import settings
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.preprocess import preprocess_hatexplain
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

    # Create model based on configs
    pretrained = cfg.model.name
    hf_kwargs = {"device_map": cfg.basic.device_map, "offload_folder": cfg.basic.offload_folder}
    model = AutoModelForCausalLM.from_pretrained(
        pretrained,
        use_auth_token=True,
        **hf_kwargs,
    )

    # Create tokenizer for the model
    tokenizer = AutoTokenizer.from_pretrained(pretrained)

    # Load correct dataset
    dataset_name = cfg.dataset.name
    dataset_path = cfg.dataset.path

    if dataset_name == 'hatexplain':
        examples = preprocess_hatexplain(dataset_path)

    # Create prompts
    prepended_prompt = cfg.prompt.text
    examples_with_prompts = list(map(lambda e: create_prompt(prepended_prompt, e), examples))

    examples_with_labels = []
    for example in tqdm.tqdm(examples_with_prompts):
        inputs = tokenizer(example['prompt'], return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=5)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        label = result.split('Answer:')[-1].strip()
        example['predicted_label'] = label

        examples_with_labels.append(example)

    # Save the processed data with labels
    with open(cfg.basic.output_dir + "/" + cfg.model.output_folder + "/" + cfg.dataset.name + ".json", 'w') as f:
        json.dump(examples_with_labels, f)

    print(f"\t Finished processing the dataset {cfg.dataset.name} with the model {cfg.model.name}.")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    run_pipeline()