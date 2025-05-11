# This file simply automates Preprocessing.ipynb via multiprocessing

import pandas as pd 
import os
import json
import mne
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs
import numpy as np
from scipy.stats import pearsonr
import re
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

os.chdir('..')

root_dir = "Datasets/Data"
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



def preprocess(core_path):

    s_name = os.path.basename(os.path.dirname(core_path))
    trial_name = os.path.basename(core_path)

    save_path = f"./Generated/Data/{s_name}/{trial_name}/"
    
    properties_path = f'{core_path}/EEG_Properties.json'
    datablocks_path = f'{core_path}/EEG.csv'
    experiment_path = f'{core_path}/Experiment.json'
    eyetracker_path = f'{core_path}/Eyetracker.asc'
    markers_path    = f'{core_path}/EEG_Markers.csv'
    
    # Load and parse properties
    with open(properties_path,'r') as f:
        props = json.load(f)
    sr = props['sampling_interval']
    chan_names  = props['channel_names']
    resolutions = props['resolutions']
    
    # Load and parse markers (if given)
    if os.path.exists(markers_path) and os.path.getsize(markers_path) > 0:
        markers_df = pd.read_csv(markers_path, names=['block_id', 'position', 'points', 'channel', 'type_name', 'description'])
    else:
        markers_df = None
        
    # Load experiment sequence
    with open(experiment_path,'r') as f:
        experiment_seq = json.load(f)
        experiment_duration = sum(block['content'].get('duration') for block in experiment_seq.values())
    
    # Load eyetracker
    eye_raw = mne.io.read_raw_eyelink(eyetracker_path, verbose=False)
    
    # Load EEG itself
    eeg_df = pd.read_csv(datablocks_path, header=None, names=['block_id']+chan_names)
    eeg_df = eeg_df.replace(r';', '', regex=True).astype(float)
    eeg_df["block_id"] = eeg_df["block_id"].astype(int)
    
    print(f"EEG len:\t{len(eeg_df)} samples\nEyetracker len:\t{len(eye_raw)} samples")
    
    
    
    
    
    if markers_df is not None:
        # Иногда, когда block_id указан не правильно, оно не отрабатывает
        try:
            # Get the lost samples per block
            markers_df['lost_samples'] = markers_df['description'].str.extract(r'(\d+)').astype(int)
            markers_df = markers_df[['block_id', 'position', 'lost_samples']]
        
            # Итерирование по markers_df
            for index, row in tqdm(markers_df.iterrows(), total=len(markers_df)):
                block_id = row['block_id']
                position = row['position']
                lost_samples = row['lost_samples']
            
                # Фильтрация строк для текущего block_id
                block_df = eeg_df[eeg_df['block_id'] == block_id]
            
                # Получение индекса начала блока
                start_idx = block_df.index[0]
            
                # Создание датафрейма с NaN значениями для вставки
                nan_df = pd.DataFrame({col: np.nan for col in eeg_df.columns}, index=[start_idx + position]*2)
                nan_df['block_id'] = block_id
            
                # Вставка nan_df в eeg_df
                eeg_df = pd.concat([eeg_df.iloc[:start_idx + position], nan_df, eeg_df.iloc[start_idx + position:]]).reset_index(drop=True)
        
            # Interpolation
            eeg_df_interpolated.loc[:, eeg_df.columns != 'block_id'] = eeg_df_interpolated.loc[:, eeg_df.columns != 'block_id'].interpolate(method='linear', limit_direction='both')
            eeg_df = eeg_df_interpolated
            eeg_df
            
        except:
            print(f"Failed to interpolate lost samples: {markers_df['lost_samples'].astype(int).sum()}")
    
    

    max_samples = sr * experiment_duration
    
    # Trim EEG value to the experiment length
    eeg_len = int(min(max_samples, len(eeg_df)-1))
    eeg_df = eeg_df.iloc[:eeg_len]
    
    # Trim EyeTracker to the experiment length
    eye_len = int(min(eeg_len, eye_raw.n_times))
    eye_raw.crop(tmax=(eye_len - 1) * eye_raw.info['sfreq']**-1)
    
    print(f"EEG len:\t{len(eeg_df)} samples\nEyetracker len:\t{len(eye_raw)} samples\nExpected len:\t{int(max_samples)} samples")
    
    

    
    # Get only data
    eeg_data = eeg_df.drop(columns=['block_id']).to_numpy().T  # Transpose: (n_channels, n_times)
    
    # Scaling to µV (или хз что за единицы)
    resolutions = np.array(resolutions).reshape(-1, 1)  # (n_channels, 1)
    eeg_data = eeg_data * (resolutions * 0.01)  # Применяем масштабирование к каждому каналу (хз почему 0.01)
    
    # Convert to mne
    info = mne.create_info(ch_names=chan_names, sfreq=sr, ch_types='eeg')
    eeg_raw = mne.io.RawArray(eeg_data, info)
    montage = mne.channels.make_standard_montage('standard_1020')
    eeg_raw.set_montage(montage)
    


    
    # ----------------------------------------------------------  Refferencing ------------------------------------------------------------------------------------------------------------------
    eeg_raw.set_eeg_reference('average', projection=True)  # усредненная ссылка
    eeg_raw.apply_proj()  # применим референтную проекцию
    
    
    
    # ----------------------------------------------------------  Frequency Filtering -----------------------------------------------------------------------------------------------------------
    eeg_raw.filter(l_freq=1.0, h_freq=40.0, fir_design='firwin')  # высоко- и низкочастотная фильтрация
    eeg_raw.notch_filter(freqs=[50, 100])  # убрать сеть (50 Гц и гармоники)
    
    
    
    # ---------------------------------------------------------- Remove eye artefacts ----------------------------------------------------------------------------------------------------------
    # Rename bad_blink to blink (хз почему конфликтует)
    new_descriptions = ['blink' if desc == 'BAD_blink' else desc for desc in eye_raw.annotations.description]
    eye_raw.set_annotations(mne.Annotations(onset=eye_raw.annotations.onset, duration=eye_raw.annotations.duration, description=new_descriptions, ch_names=eye_raw.annotations.ch_names))
    annotations = eye_raw.annotations
    
    # eye_x = eye_raw['xpos_left']
    # eye_y = eye_raw['ypos_left']
    
    blink_times = annotations.onset[annotations.description == 'blink']
    saccade_times = annotations.onset[annotations.description == 'saccade']
    
    # Получим времена из eeg_raw
    times = eeg_raw.times
    
    # Создаем сигналы для морганий и саккад
    blink_signal = np.zeros_like(times)
    saccade_signal = np.zeros_like(times)
    
    # Заполняем сигнал морганий
    for annot in annotations[annotations.description == 'blink']:
        start = np.searchsorted(times, annot['onset'])
        end = np.searchsorted(times, annot['onset'] + annot['duration'])
        blink_signal[start:end] = 1
    
    # Заполняем сигнал саккад
    for annot in annotations[annotations.description == 'saccade']:
        start = np.searchsorted(times, annot['onset'])
        end = np.searchsorted(times, annot['onset'] + annot['duration'])
        saccade_signal[start:end] = 1
    
    # Применяем ICA к EEG данным
    ica = ICA(n_components=20, random_state=42, verbose=False)
    ica.fit(eeg_raw)
    
    # Получаем ICA источники
    sources = ica.get_sources(eeg_raw)
    sources_data = sources.get_data()  # shape: (n_components, n_times)
    
    # Вычисляем корреляцию с сигналом морганий
    blink_corrs = []
    for i in range(sources_data.shape[0]):
        corr, _ = pearsonr(sources_data[i], blink_signal)
        blink_corrs.append(abs(corr))
    
    # Вычисляем корреляцию с сигналом саккад
    saccade_corrs = []
    for i in range(sources_data.shape[0]):
        corr, _ = pearsonr(sources_data[i], saccade_signal)
        saccade_corrs.append(abs(corr))
    
    # Определяем артефактные компоненты с помощью статистического порога (mean + 2*std)
    blink_corrs = np.array(blink_corrs)
    saccade_corrs = np.array(saccade_corrs)
    
    mean_blink = np.mean(blink_corrs)
    std_blink = np.std(blink_corrs)
    blink_bads = [i for i, corr in enumerate(blink_corrs) if corr > mean_blink + 2 * std_blink]
    
    mean_saccade = np.mean(saccade_corrs)
    std_saccade = np.std(saccade_corrs)
    saccade_bads = [i for i, corr in enumerate(saccade_corrs) if corr > mean_saccade + 2 * std_saccade]
    
    # Объединяем список артефактных компонент
    bads = list(set(blink_bads + saccade_bads))
    
    # Исключаем артефактные компоненты
    ica.exclude = bads
    
    # Применяем ICA для очистки данных
    eeg_raw = ica.apply(eeg_raw.copy(), verbose=False)
    
    
    # TODO --------------------------------------------------------------------------------------------------------
    # 1e-6 — соответствует ~1 μV. Если твои данные масштабированы иначе — нужно адаптировать.
    # 500e-6 — это 500 μV, это очень большая амплитуда, в клинической EEG такие всплески чаще — мусор.
    # === 4. Удаление выбросов с помощью автоматической разметки ===
    # eeg_raw = eeg_raw.copy().annotate_flat(threshold=1e-6)  # плоские сигналы
    # eeg_raw = eeg_raw.copy().annotate_amplitude(threshold=500e-6)  # слишком большие амплитуды
    
    # TODO
    # Bad channels
    # eeg_raw.info['bads'] = ['F7', 'FT9']  # можно добавить вручную
    # eeg_raw.interpolate_bads(reset_bads=True)
    
    # Save
    os.makedirs(save_path, exist_ok=True)
    eeg_raw.save(f'{save_path}EEG_clean.fif', overwrite=True)
    
    
    
    
    
    # Инициализация переменных для отслеживания текущего паттерна и команды
    current_pattern_type = None
    current_pattern_id = None
    command_state = None
    
    # Итерация по блокам JSON
    pattern_trio = []
    for block_key, block_value in experiment_seq.items():
        if block_value['type'] == 'pattern':
            if len(pattern_trio) == 3:
                pattern_trio = []
                
            # Запоминаем тип и ID текущего паттерна
            current_pattern_type = block_value['content']['type']
            if current_pattern_type == 'geometric':
                current_pattern_id = block_value['content']['pattern_id']
            elif current_pattern_type == 'random':
                current_pattern_id = block_value['content']['seed']
    
            pattern_trio.append([current_pattern_type, current_pattern_id])
            
        elif block_value['type'] == 'command':
            # Запоминаем состояние команды
            command_state = int(block_value['content']['state']) - 1
            # Здесь мы могли бы проверить, что current_pattern_type и current_pattern_id не None,
            # но в данном случае это не обязательно, поскольку структура JSON описана жестко.
        elif block_value['type'] == 'execution':
            # Добавляем информацию о паттерне в блок execution
            block_value['content']['pattern_type'] = pattern_trio[command_state][0]
            block_value['content']['pattern_id'] = pattern_trio[command_state][1]
    
    
    
    timestamp = 0
    for block in sorted(experiment_seq.keys(), key=lambda x: int(x.split('_')[1])):
        block_data = experiment_seq[block]
        block_data['timestamp'] = timestamp
        timestamp += block_data['content']['duration']
    
    # Save updated experiment sequence .json file
    experiment_path = f'{save_path}Experiment.json'
    with open(experiment_path, "w", encoding="utf-8") as f:
        json.dump(experiment_seq, f, indent=2, ensure_ascii=False)

    print("Done!")














if __name__ == "__main__":
    paths = collect_valid_paths()
    print(f"Найдено {len(paths)} подходящих директорий. Запускаю обработку...")

    # Выбираем количество процессов (по умолчанию — по числу ядер)
    with Pool(processes=cpu_count()) as pool:
        pool.map(preprocess, paths)
