# This file is a simple automatation of `Spectral_Analysis.ipynb` "All experiment spectrum" block for all the subjects averaged
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
s_dir_pattern = re.compile(r"^S_\d+$")
trial_dir_pattern = re.compile(r"^Trial_\d+$")

def collect_valid_paths():
    valid_dirs = []
    for s_dir in os.listdir(root_dir):
        s_path = os.path.join(root_dir, s_dir)
        if os.path.isdir(s_path) and s_dir_pattern.match(s_dir):
            for trial_dir in os.listdir(s_path):
                trial_path = os.path.join(s_path, trial_dir)
                if os.path.isdir(trial_path) and trial_dir_pattern.match(trial_dir):
                    if any(os.path.isfile(os.path.join(trial_path, f)) for f in os.listdir(trial_path)):
                        valid_dirs.append(trial_path)
    return valid_dirs

def load_clean_segments(core_path):
    eeg_clean_path = f"{core_path}/EEG_clean.fif"
    experiment_path = f"{core_path}/Experiment.json"

    try:
        raw = mne.io.read_raw_fif(eeg_clean_path, preload=True, verbose=False)
        raw.pick_types(eeg=True, verbose=False)
        sr = int(raw.info['sfreq'])

        with open(experiment_path, 'r') as f:
            experiment_seq = json.load(f)

        clean_blocks = [
            (block_data['timestamp'], block_data['content']['duration'])
            for block_data in experiment_seq.values()
            if not (block_data['type'] == 'rest' and block_data['content']['duration'] in [30, 300])
        ]

        epochs = []
        for start_time, duration in clean_blocks:
            start_sample = int(start_time)
            stop_sample = min(int(start_time + duration), len(raw) / sr - 0.001)
            segment = raw.copy().crop(tmin=start_sample, tmax=stop_sample)
            epochs.append(segment)

        if epochs:
            return mne.concatenate_raws(epochs)
    except Exception as e:
        print(f"[WARNING] Ошибка при обработке {core_path}: {e}")
    
    return None

if __name__ == "__main__":
    all_paths = collect_valid_paths()
    print(f"Найдено {len(all_paths)} подходящих директорий. Начинаем сбор данных...")

    all_segments = []

    for path in tqdm(all_paths):
        segment = load_clean_segments(path)
        if segment:
            all_segments.append(segment)

    if not all_segments:
        print("Нет данных для построения спектров.")
        exit(1)

    # Объединяем все сегменты
    combined_raw = mne.concatenate_raws(all_segments)

    # Строим спектры
    out_dir = "./Generated/Figures/Spectral_Analysis/"
    os.makedirs(out_dir, exist_ok=True)

    title = "Combined PSD Spectrum for All Subjects"

    fig1 = combined_raw.plot_psd(fmax=40, average=True, show=False)
    fig1.suptitle(f"{title} (Averaged)")
    fig1.savefig(os.path.join(out_dir, "AllSubjects_PSD_Avg.png"))

    fig2 = combined_raw.plot_psd(fmax=40, show=False)
    fig2.suptitle(f"{title} (Per Channel)")
    fig2.savefig(os.path.join(out_dir, "All_Subjects_PSD_PerChannel.png"))

    plt.close(fig1)
    plt.close(fig2)

    print("Готово. Спектры сохранены.")

