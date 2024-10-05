def create_prompt(prepended_prompt, example):
    """
    Prepend the prompt used for classification based on the text given in config.

    :param prepended_prompt: str
    :param example: dict
    :return: dict
    """
    prompt = f"{prepended_prompt}:\n\nText: \"{example['text']}\"\nAnswer:"
    example['prompt'] = prompt

    return example
