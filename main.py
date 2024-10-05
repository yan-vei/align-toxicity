import wandb
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from settings import settings
from transformers import AutoModelForCausalLM


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


    if use_wandb:
        wandb.finish()