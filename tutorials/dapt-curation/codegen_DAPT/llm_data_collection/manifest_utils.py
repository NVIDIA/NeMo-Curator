
import os
import pandas as pd
import numpy as np
import sys

from bs4 import BeautifulSoup
from prospector_lm_download_utils import (
    decode_html,
    lang_detect,
    get_stop_list_dict,
    extract_text,
)

# Get stop lists for justext
stop_lists = get_stop_list_dict()

# Function to remove html tags from a string
# Use BeautifulSoup or jusText
# If navbar present, use jusText (e.g., nvwiki)
# if not present, use BeautifulSoup (e.g. Confluence)
def remove_html_tags(html, use_justext=False):
    global stop_lists

    if use_justext:
        lang = lang_detect(html)
        text = None
        if lang in stop_lists:
            text = extract_text(html, stop_lists[lang])

        if not text:
            return ""
        return '\n\n'.join(text)
    else:
        soup = BeautifulSoup(html, "lxml")  # Use lxml as the parser
        return soup.get_text('\n')


def bytes_to_mb(size):
    return size / 1_000_000

def bytes_to_kb(size):
    return size / 1_000

def get_extension(filename):
    special_cases = ['Makefile', 'Makeppfile', '.ep3.pp'] # multi extensions needed here
    for case in special_cases:
        if filename.upper().endswith(case.upper()):
            return case.upper()
    ext = os.path.splitext(filename)[1]
    # Special case for .gz files: use the original filename instead
    if ext.upper() == ".GZ":
        ext = get_extension(os.path.splitext(filename)[0])
    return ext.upper() if ext else 'NO_EXTENSION'

def human_readable_range(value1, value2, do_log):
    if do_log:
        value1 = 10 ** value1
        value2 = 10 ** value2

    # formatter = mpl.ticker.EngFormatter(places=2)
    # return f"{formatter(value1)} - {formatter(value2)}"

    return f"{value1:.2f} - {value2:.2f}"
    # return f"{value1:,.0f} - {value2:,.0f}"
    # return f"{value1:,.2E} - {value2:,.2E}"

def remove_outliers(data, extension_name):
    Q1 = data[extension_name].quantile(0.25)
    Q3 = data[extension_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[extension_name] >= lower_bound) & (data[extension_name] <= upper_bound)]

def manifest_on_open(data):
    # Extract file extensions, handle extension-less files
    # data['Extension'] = data['Path'].apply(lambda x: os.path.splitext(x)[1].upper() if os.path.splitext(x)[1] else 'NO_EXTENSION')
    data.loc[:,'Extension'] = data['Path'].apply(get_extension)

    # Calculate the total file size for each file extension and convert to MB
    data.loc[:,'Size_MB'] = data['Size'].apply(bytes_to_mb)
    data.loc[:,'Size_KB'] = data['Size'].apply(bytes_to_kb)
    return data

def manifest_on_close(data):
    data = data.drop(columns=['Extension'])
    data = data.drop(columns=['Size_MB', 'Size_KB'])
    return data

def print_histogram(data, extensions, extension_name, remove_outliers=False, do_log=False):
    # convert extensions to upper case
    extensions = [ext.upper() for ext in extensions]

    max_width = 50
    filtered_data = data[data['Extension'].isin(extensions)]
    if remove_outliers:
        filtered_data = remove_outliers(filtered_data, extension_name)
    human_ext_name = extension_name

    # example of how to get total size of all files
    # total_size_mb = filtered_data['Size_MB'].sum()

    if not filtered_data.empty:
        num_bins = 10

        if do_log:
            filtered_data = filtered_data[filtered_data[extension_name] > 0]
            filtered_data['Log_' + extension_name] = np.log10(filtered_data[extension_name])
            extension_name = 'Log_' + extension_name
            # # remove nan values
            # filtered_data = filtered_data[filtered_data[extension_name].notna()]
            # # remove inf values
            # filtered_data = filtered_data[filtered_data[extension_name] != float('-inf')]

        min_size = filtered_data[extension_name].min()
        max_size = filtered_data[extension_name].max()

        size_range = (max_size - min_size) / num_bins
        bins = [min_size + i * size_range for i in range(num_bins+1)]
        labels = [human_readable_range(bins[i], bins[i+1], do_log) for i in range(num_bins)]

        filtered_data['Hist_Category'] = pd.cut(filtered_data[extension_name], bins=bins, labels=labels, include_lowest=True)
        histogram_data = filtered_data['Hist_Category'].value_counts().sort_index()

        max_value = histogram_data.max()

        print(f"\nHistogram of {human_ext_name} for {', '.join(extensions)} file extension:")
        for index, value in histogram_data.items():
            num_bars = int((value / max_value) * max_width)
            # print(f"{index:<15} | {'#' * num_bars}")
            print(f"{index:<15} | {'#' * num_bars}{' ' * (50 - num_bars)} ({value})")