import re
import pandas as pd
HYPHEN_TRANSLATION = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
    }
)

def normalize_mirna(value):
    text = str(value).strip().translate(HYPHEN_TRANSLATION)
    if not text or text.lower() == "nan":
        return ""
    lower = text.lower()
    if lower.startswith("hsa-"):
        return "hsa-" + text[4:]
    if lower.startswith("mirna-"):
        return "hsa-miR-" + text[6:]
    if lower.startswith("microrna-"):
        return "hsa-miR-" + text[9:]
    if lower.startswith("mir-"):
        return "hsa-miR-" + text[4:]
    if lower.startswith("let-"):
        return "hsa-" + text
    return text
