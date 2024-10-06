import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    # instance_data = {
    #     "audio": audio,
    #     "spectrogram": spectrogram,
    #     "text": text,
    #     "text_encoded": text_encoded,
    #     "audio_path": audio_path,
    # }

    batch = {}

    batch["text"] = [item["text"] for item in dataset_items]
    batch["text_encoded"] = [item["text_encoded"] for item in dataset_items]
    batch["audio_path"] = [item["audio_path"] for item in dataset_items]

    batch_len = len(dataset_items)

    max_audio_len, max_text_len, max_spectrogram_len = -1, -1, -1
    for item in dataset_items:
        max_audio_len = max(max_audio_len, item["audio"].shape[1])
        max_text_len = max(max_text_len, len(item["text"]))
        max_spectrogram_len = max(max_spectrogram_len, item["spectrogram"].shape[2])

    batch["audio"] = torch.zeros(batch_len, max_audio_len)
    batch["spectrogram"] = torch.zeros(
        batch_len, dataset_items[0]["spectrogram"].shape[1], max_spectrogram_len
    )
    batch["text_encoded"] = torch.zeros(batch_len, max_text_len)

    for ind, item in enumerate(dataset_items):
        batch["audio"][ind, : item["audio"].shape[1]] = item["audio"]
        batch["spectrogram"][ind, :, : item["spectrogram"].shape[2]] = item[
            "spectrogram"
        ]
        batch["text_encoded"][ind, : item["text_encoded"].shape[1]] = item[
            "text_encoded"
        ]

    batch["text_encoded_length"] = torch.IntTensor(
        [dataset_item["text_encoded"].shape[1] for dataset_item in dataset_items]
    )
    batch["spectrogram_length"] = torch.IntTensor(
        [dataset_item["spectrogram"].shape[2] for dataset_item in dataset_items]
    )

    return batch
