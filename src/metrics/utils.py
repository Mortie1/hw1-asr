# Based on seminar materials
import editdistance

# Don't forget to support cases when target_text == ''


def calc_cer(target_text: str, predicted_text: str) -> float:
    if not target_text:
        if not predicted_text:
            return 0
        return 1
    return editdistance.eval(target_text, predicted_text) / len(target_text)


def calc_wer(target_text: str, predicted_text: str) -> float:
    if not target_text:
        if not predicted_text:
            return 0
        return 1
    return editdistance.eval(target_text.split(), predicted_text.split()) / len(
        target_text.split()
    )
