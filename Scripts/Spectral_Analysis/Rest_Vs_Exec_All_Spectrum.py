import os
import re
import numpy as np
import mne
import matplotlib.pyplot as plt
from tqdm import tqdm

# Корневая папка с данными по субъектам и трейлам
root_dir = "Generated/Data"

# Регулярка для S_директорий: S_ и только цифры после
s_dir_pattern = re.compile(r"^S_\d+$")
# Регулярка для Trial_директорий: Trial_ и только цифры
trial_dir_pattern = re.compile(r"^Trial_\d+$")

def collect_valid_paths():
    """
    Сканируем root_dir и собираем полные пути до тех папок Trial_<число>,
    где внутри есть хотя бы один файл.
    """
    valid_dirs = []
    for s_dir in os.listdir(root_dir):
        s_path = os.path.join(root_dir, s_dir)
        if os.path.isdir(s_path) and s_dir_pattern.match(s_dir):
            for trial_dir in os.listdir(s_path):
                trial_path = os.path.join(s_path, trial_dir)
                if os.path.isdir(trial_path) and trial_dir_pattern.match(trial_dir):
                    # Проверяем, есть ли хотя бы один файл внутри
                    if any(os.path.isfile(os.path.join(trial_path, f)) for f in os.listdir(trial_path)):
                        valid_dirs.append(trial_path)
    return valid_dirs

# Собираем все валидные пути к папкам Trial_<id>
all_trial_paths = collect_valid_paths()

# Списки, в которых будем копить спектральные массивы по всем трейлам
rest_trials = []
exec_trials = []

for i, trial_path in tqdm(enumerate(all_trial_paths), desc="Processing trials"):
    eeg_clean_path = os.path.join(trial_path, "EEG_clean.fif")
    # Если файла нет — пропускаем
    if not os.path.isfile(eeg_clean_path):
        continue

    # Загрузка .fif
    raw = mne.io.read_raw_fif(eeg_clean_path, preload=True, verbose=False)
    raw.pick_types(eeg=True, verbose=False)  # только EEG-каналы
    sr = int(raw.info['sfreq'])  # sampling rate
    

    # Извлекаем времена «execution»-блоков
    execution_times = [
        (block_data['timestamp'], block_data['content']['duration'])
        for block_data in experiment_seq.values()
        if block_data['type'] == 'execution'
    ]
    # Извлекаем времена «rest»-блоков длительностью ровно 10 секунд
    rest_times = [
        (block_data['timestamp'], block_data['content']['duration'])
        for block_data in experiment_seq.values()
        if block_data['type'] == 'rest' and block_data['content']['duration'] == 10
    ]

    # Собираем сегменты для execution
    execution_epochs = []
    for start_time, duration in execution_times:
        # переводим время (в секундах) в отрезок во временных единицах Raw
        tmin = start_time
        tmax = start_time + duration
        # чтобы не выйти за границы, tmax ограничиваем длительностью сигнала
        tmax = min(tmax, raw.n_times / sr - 0.001)
        segment = raw.copy().crop(tmin=tmin, tmax=tmax)
        execution_epochs.append(segment)

    # Собираем сегменты для rest
    rest_epochs = []
    for start_time, duration in rest_times:
        tmin = start_time
        tmax = start_time + duration
        tmax = min(tmax, raw.n_times / sr - 0.001)
        segment = raw.copy().crop(tmin=tmin, tmax=tmax)
        rest_epochs.append(segment)

    print(f"[{trial_path}] Detected {len(execution_epochs)} execution blocks, {len(rest_epochs)} rest blocks.")

    # Параметры анализа: частоты и число циклов для Morlet
    freqs = np.linspace(2, 40, 50)
    n_cycles = freqs / 2.

    def compute_spectra_array(raw_epochs):
        """
        Даны raw_epochs — список Raw-объектов (эпох), возвращает:
          - spec_array: shape (n_epochs, n_freqs, n_times)
          - times: ось времени (для усреднённых сигналов внутри эпох)
        """
        all_power = []
        # Находим минимальную длину (в сэмплах) среди всех эпох
        min_len = min(epoch.get_data(picks='eeg').shape[1] for epoch in raw_epochs)

        for epoch in raw_epochs:
            data = epoch.get_data(picks='eeg')[:, :min_len]  # (n_channels, n_times)
            sfreq = epoch.info['sfreq']
            times = np.arange(min_len) / sfreq

            # приводим к форме (1, n_channels, n_times), чтобы tfr_array_morlet принял
            data = data[np.newaxis, ...]

            power = mne.time_frequency.tfr_array_morlet(
                data, sfreq=sfreq,
                freqs=freqs, n_cycles=n_cycles,
                output='power', n_jobs=72, verbose=False
            )[0]  # → (n_channels, n_freqs, n_times)

            # усредняем по каналам, чтобы получить (n_freqs, n_times)
            avg_power = power.mean(axis=0)
            all_power.append(avg_power)

        # → spec_array: (n_epochs, n_freqs, n_times)
        spec_array = np.stack(all_power, axis=0)
        return spec_array, times

    # Если нет эпох — пропускаем добавление
    if len(rest_epochs) > 0:
        rest_spec_arr, times_rest = compute_spectra_array(rest_epochs)
        rest_trials.append(rest_spec_arr)
    if len(execution_epochs) > 0:
        exec_spec_arr, times_exec = compute_spectra_array(execution_epochs)
        exec_trials.append(exec_spec_arr)

# После цикла по всем trial-папкам: объединяем всё

# 1) Объединяем все массивы «rest» по первой оси (эпохи)
if len(rest_trials) > 0:
    # rest_trials — список элементов shape (n_epochs_i, n_freqs, n_times)
    rest_all = np.concatenate(rest_trials, axis=0)  # (total_rest_epochs, n_freqs, n_times)
    # 2) Усредняем по эпохам → (n_freqs, n_times)
    rest_avg = rest_all.mean(axis=0)
else:
    raise RuntimeError("Не найдено ни одной rest-эпохи во всех трейлах!")

# Аналогично для «execution»
if len(exec_trials) > 0:
    exec_all = np.concatenate(exec_trials, axis=0)  # (total_exec_epochs, n_freqs, n_times)
    exec_avg = exec_all.mean(axis=0)
else:
    raise RuntimeError("Не найдено ни одной execution-эпохи во всех трейлах!")

# Усреднение по времени: получаем одномерный спектр (n_freqs,)
rest_spectrum = rest_avg.mean(axis=1)   # усредняем rest_avg по оси времени
exec_spectrum = exec_avg.mean(axis=1)

# Построение итогового графика
plt.figure(figsize=(10, 5))
plt.plot(freqs, rest_spectrum, label='Rest', color='blue')
plt.plot(freqs, exec_spectrum, label='Execution', color='red')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Average Power')
plt.title('Average Power Spectrum: Rest vs Execution')
plt.legend()
plt.grid(True)
plt.savefig('Rest_Vs_Exec_All_Spectrum.png', dpi=300, bbox_inches='tight')
plt.show()