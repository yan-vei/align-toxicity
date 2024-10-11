import json


def create_prompt(prepended_prompt, example, is_fewshot=False):
    """
    Prepend the prompt used for classification based on the text given in config.

    :param prepended_prompt: str
    :param example: dict
    :param is_fewshot: bool
    :return: dict
    """
    if is_fewshot:
        with open('datasets/implicit_hate_fewshots.json', 'r') as f:
            fewshots = json.load(f)
        prompt = (f"These are examples of some types of implicit hate speech: {fewshots[0]['text']}\t{fewshots[1]['text']}"
                  f"{prepended_prompt}:\n\nText: \"{example['text']}\"\nAnswer:")
    else:
        prompt = f"{prepended_prompt}:\n\nText: \"{example['text']}\"\nAnswer:"

    example['prompt'] = prompt

    return example
