import csv
import os

def create_folder(path: str):
    """
    @param path: string, name of folder to create

    Create folder if not exists
    """
    if not os.path.exists(path):
        os.makedirs(path)


def write_csv_file(fname, data, *args, **kwargs):
    """
    @param fname: string, name of file to write
    @param data: list of list of items

    Write data to file
    """
    csv_file = csv.writer(open(fname, 'w'), *args, **kwargs)

    for row in data:
        csv_file.writerow(row)

def normalize_mic(value: str):
    """
    @param fname: string, value to normalize

    Normalize value following Nguyen rules
    """
    if("/" in value):
        value = value.split("/")[0]

    if ("<=" in value or ">=" in value):
        return float(value.replace("<=", "").replace(">=", ""))

    if ("<" in value):
        return float(value.replace("<", ""))/2

    if (">" in value):
        return float(value.replace(">", "")) * 2

    return float(value)
