# This file is a simple automatation of `Spectral_Analysis.ipynb` "All experiment spectrum" block for all the subjects localization of 4 basic rythms
import numpy as np
import matplotlib.pyplot as plt
import mne

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
    # Предполагается, что `combined_raw` уже загружен и содержит объединённые сегменты
    
    # Параметры PSD
    psd = combined_raw.compute_psd(fmin=1, fmax=40, n_fft=2048, verbose=False)
    psds_data = 10 * np.log10(psd.get_data())  # [n_channels, n_freqs]
    freqs = psd.freqs
    
    # Усреднение по частотным диапазонам
    band_limits = {
        'Delta': (1, 4),
        'Tetta': (4, 7),
        'Alpha': (7, 13),
        'Beta': (13, 30)
    }
    
    # Отрисовка топомап
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    for ax, (band_name, (fmin_band, fmax_band)) in zip(axes, band_limits.items()):
        freq_mask = (freqs >= fmin_band) & (freqs <= fmax_band)
        psds_band = psds_data[:, freq_mask].mean(axis=1)
    
        vmin = np.min(psds_band)
        vmax = np.max(psds_band)
    
        mne.viz.plot_topomap(
            psds_band, combined_raw.info, axes=ax, show=False,
            cmap='plasma', contours=0, vlim=(vmin, vmax)
        )
        ax.set_title(f'{band_name}')
    
    plt.suptitle("Topomap of EEG Power in Frequency Bands")
    plt.tight_layout()
    
    out_dir = "./Generated/Figures/Spectral_Analysis/"
    
    # Сохраняем график
    fig.savefig(f"{out_dir}All_Subjects_Freq_Bands_Localization.png", dpi=300)