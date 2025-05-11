# This file is a simple automatation of `Spectral_Analysis.ipynb` "All experiment spectrum" block for all the subjects for each particular subject 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import json
import mne
import numpy as np
from tqdm import tqdm
import re

os.chdir('..')



root_dir = "Generated/Data"
valid_dirs = []

# Регулярка для S_директорий: S_ и только цифры после
s_dir_pattern = re.compile(r"^S_\d+$")
# Регулярка для Trial_директорий: Trial_ и только цифры
trial_dir_pattern = re.compile(r"^Trial_\d+$")


def collect_valid_paths():
    for s_dir in os.listdir(root_dir):
        s_path = os.path.join(root_dir, s_dir)
        if os.path.isdir(s_path) and s_dir_pattern.match(s_dir):
            for trial_dir in os.listdir(s_path):
                trial_path = os.path.join(s_path, trial_dir)
                if os.path.isdir(trial_path) and trial_dir_pattern.match(trial_dir):
                    # Проверяем, есть ли хотя бы один файл
                    if any(os.path.isfile(os.path.join(trial_path, f)) for f in os.listdir(trial_path)):
                        valid_dirs.append(trial_path)
    return valid_dirs


    
def generate_all_specrum(core_path):

    eeg_clean_path = f"{core_path}/EEG_clean.fif"
    experiment_path = f"{core_path}/Experiment.json"
    
    # Load the .fif file
    raw = mne.io.read_raw_fif(eeg_clean_path , preload=True, verbose=False)
    raw.pick_types(eeg=True, verbose=False)  # только EEG-каналы
    sr = int(raw.info['sfreq'])  # sampling rate
    
    # Load experiment sequence
    with open(experiment_path,'r') as f:
        experiment_seq = json.load(f)

    # Фильтруем все блоки кроме 300 sec и 30 sec rest
    clean_blocks = [
        (block_data['timestamp'], block_data['content']['duration'])
        for block_data in experiment_seq.values()
        if not (block_data['type'] == 'rest' and (block_data['content']['duration'] == 300 or block_data['content']['duration'] == 30))
    ]
    
    # Соберем отрезки с execution
    epochs = []
    for start_time, duration in clean_blocks:
        start_sample = int(start_time)
        stop_sample = min(int((start_time + duration)), len(raw) / sr - 0.001)
        segment = raw.copy().crop(tmin=start_sample, tmax=stop_sample)
        epochs.append(segment)
    
    # Объединяем все execution-сегменты
    raw = mne.concatenate_raws(epochs)
    
    # Путь к выходной директории
    out_dir = "./Generated/Figures/Spectral_Analysis/"
    s_name = os.path.basename(os.path.dirname(core_path))
    trial_name = os.path.basename(core_path)

    os.makedirs(out_dir, exist_ok=True)
    
    title = f"{s_name} {trial_name} PSD Spectrum for all experiment"
    
    # Рисуем усредненный PSD
    fig1 = raw.plot_psd(fmax=40, average=True, show=False)
    fig1.suptitle(title)
    fig1.savefig(os.path.join(out_dir, f"{s_name}_{trial_name}_PSD_All_avg.png"))
    
    # Рисуем PSD для всего raw
    fig2 = raw.plot_psd(fmax=40, show=False)
    fig2.suptitle(title)
    fig2.savefig(os.path.join(out_dir, f"{s_name}_{trial_name}_PSD_All.png"))
    
    # Закрываем фигуры чтобы освободить память
    plt.close(fig1)
    plt.close(fig2)




if __name__ == "__main__":
    paths = collect_valid_paths()
    print(f"Найдено {len(paths)} подходящих директорий. Starting spectrum generation...")

    # Выбираем количество процессов (по умолчанию — по числу ядер)
    with Pool(processes=cpu_count()) as pool:
        pool.map(generate_all_specrum, paths)
