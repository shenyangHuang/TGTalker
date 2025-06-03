import csv
import numpy as np

def load_txt_from_edgelist(file_path):
    out_str = ""
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            out_str += f"{row[0]},{row[1]},{row[2]}\n"
    return out_str


def load_rows_from_edgelist(file_path):
    out_rows = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            out_rows.append([row[0],row[1],row[2]])
    return out_rows


def row2text(rows):
    out_text = ""
    for row in rows:
        # out_text += f"{row[0]},{row[1]},{row[2]}\n"
        out_text += f"({row[0]},{row[1]},{row[2]})"
    return out_text


def predict_link(query_dst: np.ndarray, llm_dst: int) -> np.ndarray:
    r"""
    convert LLM prediction into MRR format, just check if the LLM prediction is within the possible destinations
    """
    pred = np.zeros(len(query_dst))
    idx = 0
    for dst in query_dst:
        if (dst == llm_dst):
            pred[idx] = 1.0
        idx += 1
    return pred