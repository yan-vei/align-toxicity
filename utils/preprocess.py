import json


def preprocess_dataset(filepath):
    """
    Preprocess Implicit Hate dataset or OffensEval; labeling is 1 for toxic and 0 for non-toxic.

    :param filepath: str, directory with the dataset (.json format)
    :return: list of dicts
    """

    with open(filepath, 'r') as file:
        data = json.load(file)

    return data


def preprocess_hatexplain(filepath):
    """
    Preprocess HateXplain dataset and label it with 1 for toxic and 0 for non-toxic tweets.

    :param filepath: str, directory with the dataset (.json format)
    :return: list of dicts
    """
    with open(filepath, 'r') as file:
        data = json.load(file)

    processed_list = []

    for post_id, post_data in data.items():
        text = " ".join(post_data["post_tokens"])
        annotator_count = len(post_data["annotators"])

        # Count how many annotators labeled the post as "hatespeech" or "offensive"
        label_counts = {"hatespeech": 0, "offensive": 0}
        for annotator in post_data["annotators"]:
            if annotator["label"] in label_counts:
                label_counts[annotator["label"]] += 1

        # Determine the final label: 1 if majority is "hatespeech" or "offensive"
        majority_label = 1 if (label_counts["hatespeech"] + label_counts["offensive"]) > (annotator_count / 2) else 0

        processed_entry = {
            "post_id": post_id,
            "text": text,
            "annotator_count": annotator_count,
            "label": majority_label,
            "original_labels": post_data["annotators"],
            "rationales": post_data["rationales"],
            "post_tokens": post_data["post_tokens"],
        }

        processed_list.append(processed_entry)

    return processed_list

