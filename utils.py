import csv
import numpy as np
import random
import torch



def set_seed(seed: int = 42):
    """
    Set the random seed for reproducibility.
    Args:
        seed
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def readcsv2list(file_path):
    r"""
    Read a CSV file and return its content as a list of lists.
    Args:
        file_path: Path to the CSV file.
    Returns:
        List of lists containing the CSV data.
    """
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        pred_list = []
        true_list = []
        for row in reader:
            pred_list.append(int(row[0]))
            true_list.append(int(row[1]))
    return pred_list, true_list


def append2csv(file_path, data):
    r"""
    Append data to a CSV file.
    Args:
        file_path: Path to the CSV file.
        data: Data to append (list of lists).
    """
    with open(file_path, 'a', newline='\n') as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow(row)



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