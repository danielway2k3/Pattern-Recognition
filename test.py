import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import os
from scipy import signal
from scipy.fft import fft, fftshift

import warnings
warnings.filterwarnings("ignore")

"""## 1. Data processing"""

folder_path = r"C:\Users\Daniel\OneDrive - VNU-HCMUS\Desktop\Pattern-Recognition\data\EEG Data"
file_paths = []

for dirname, _, filenames in os.walk(folder_path):
    for filename in filenames:
        file_paths.append(os.path.join(dirname, filename))

file_paths

mat = scipy.io.loadmat(file_paths[0])
data = mat['o']['data'][0,0]

column_names = [
    'COUNTER', 'INTERPOLATED', 'RAW_CQ', 'AF3', 'F7', 'F3', 'FC5',
    'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8',
    'AF4', 'GYROX', 'GYROY', 'TIMESTAMP', 'ES_TIMESTAMP', 'FUNC_ID',
    'FUNC_VALUE', 'MARKER', 'SYNC_SIGNAL'
]
data_1 = pd.DataFrame(data, columns=column_names)

data_1.head()

data_1.shape

fig, ax = plt.subplots(14,1)
fig.set_figwidth(20)
fig.set_figheight(50)
fig.set_size_inches(30, 20)
for i in range(14):
    data_ave = data_1.iloc[200000:230000, i+3] - np.mean(data_1.iloc[200000:230000, i+3])
    ax[i].plot(data_ave)
    ax[i].set_title(data_1.columns[i+3], fontsize=20)
    ax[i].set_ylim(-100, 100)

"""Trong bài báo liên quan, 4 dây dẫn được xác định là T3, T4, T5 và T6 được sử dụng để cung cấp dòng điện và thiết lập tham chiếu EEG và không dùng cho việc thu thập dữ liệu

Những dữ liệu được thu thập từ 7 điểm dẫn được cho là F7, F3, P7, O1, O2, P8 và AF4

Từ hình vẽ ta cũng dễ dàng nhận thấy được điều đó khi chỉ có 7 channels cung cấp được nhiều thông tin hơn với phần còn lại. Chính vì thế nên ta cũng sẽ chỉ lấy 7 tín hiệu từ 7 channels này để mang đi train mô hình dự đoán.
"""

useful_channels = np.array([1,2,5,6,7,8,13]) + 3
useful_channels_names = data_1.columns[useful_channels]

channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

# chọn những channnl EEG quan trọng
fig, ax = plt.subplots(7)
fig.set_figwidth(20)
fig.set_figheight(50)
fig.set_size_inches(30, 20)
j=0
for i in useful_channels:
    data_ave = data_1.iloc[200000:230000, i] - np.mean(data_1.iloc[200000:230000, i])
    ax[j].plot(data_ave)
    ax[j].set_ylim(-200,200)
    ax[j].set_title(channels[i-3], fontsize=20)
    j = j+1

"""Môi tập dữ liệu là một bản ghi lại tín hiệu sóng não thông qua quá trình thử nghiệm lái tàu mô phỏng trên máy tính có tên là "Microsoft Train Simulator". Với 5 người tham gia, mỗi file sẽ kéo dài trong khoảng 35-55 phút, mỗi người chơi thực hiện 7 lần thử nghiệm.

Trong 10 phút đầu người chơi duy trì trạng thái tập trung, ở 10 phút tiếp theo thì người chơi sẽ vào trạng thái không tập trung, và thời gian còn lại sẽ rơi vào trạng thái buồn ngủ.

Trong bản ghi tác giả có nói 2 bản ghi đầu của mỗi đối tượng sẽ để làm quen nên ta sẽ chỉ lấy 5 bản ghi cuối cùng của mỗi người tham gia, và người thứ 5 sẽ không được ghi lại ở bản ghi thứ 7
"""

marker = 128*60*10 # lấy mẫu dữ liệu trong 10 phút
# loại bỏ file 28 trong các bản ghi vì file này không đủ dữ liệu
useful_file_index = [3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 24, 25, 26, 27, 31, 32, 33, 34]
FS = mat['o']['sampFreq'][0][0][0][0]

chan_num = 7
trail_names = []
data_focus = {}
data_unfocus = {}
data_drowsy = {}
focus = {}
unfocus = {}
drowsy = {}

# trích xuất dữ liệu theo từng trạng thái dựa vào các khoảng thời gian, mỗi khoảng là 10 phút
for dirname, _, filenames in os.walk(folder_path):
    for index, filename in enumerate(filenames):
        if int(filename.split('d')[1].split('.')[0]) in useful_file_index:
            mat = scipy.io.loadmat(os.path.join(dirname, filename))
            trail_names.append(filename.split('.')[0])
            data_focus[trail_names[-1]] = mat['o']['data'][0,0][0:marker,useful_channels].copy()
            data_unfocus[trail_names[-1]] = mat['o']['data'][0,0][marker:2*marker,useful_channels].copy()
            data_drowsy[trail_names[-1]] = mat['o']['data'][0,0][2*marker:3*marker,useful_channels].copy()
            focus[trail_names[-1]] = mat['o']['data'][0,0][0:marker,useful_channels].copy()
            unfocus[trail_names[-1]] = mat['o']['data'][0,0][marker:2*marker,useful_channels].copy()
            drowsy[trail_names[-1]] = mat['o']['data'][0,0][2*marker:3*marker,useful_channels].copy()

trail_names

data_focus.keys()

data_focus[trail_names[0]].shape

from scipy.signal import butter, iirnotch, filtfilt

def combined_filter(data, fs=128):
    """
    Kết hợp bộ lọc thông dải và bộ lọc Notch để xử lý tín hiệu EEG

    Parameters:
    - data: tín hiệu đầu vào
    - fs: tần số lấy mẫu (mặc định 128 Hz)

    Returns:
    - filtered_signal: tín hiệu đã được lọc
    """
    # Thiết kế bộ lọc thông dải (0.5-40 Hz)
    nyquist = fs/2
    low = 0.5/nyquist
    high = 40/nyquist
    b_band, a_band = butter(4, [low, high], btype='band')

    # Thiết kế bộ lọc Notch tại 50Hz
    notch_freq = 50  # tần số nguồn điện
    quality_factor = 30  # độ rộng của bộ lọc
    b_notch, a_notch = iirnotch(notch_freq, quality_factor, fs)

    # Áp dụng bộ lọc band-pass
    bandpass_filtered = filtfilt(b_band, a_band, data)

    # Áp dụng bộ lọc Notch
    filtered_signal = filtfilt(b_notch, a_notch, bandpass_filtered)

    return filtered_signal

# Thêm hàm apply_ica sau hàm combined_filter
def apply_ica(data, n_components=None):
    """
    Áp dụng ICA để loại bỏ nhiễu từ tín hiệu EEG.
    
    Parameters:
    - data: ndarray, shape (n_samples, n_channels)
        Dữ liệu EEG đầu vào
    - n_components: int, optional
        Số components muốn giữ lại. Mặc định bằng số kênh
        
    Returns:
    - cleaned_data: ndarray, shape (n_samples, n_channels)
        Dữ liệu EEG đã được làm sạch
    """
    # Khởi tạo ICA
    ica = FastICA(n_components=n_components, random_state=42)
    
    # Chuẩn hóa dữ liệu
    data_normalized = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    
    # Fit và transform dữ liệu
    components = ica.fit_transform(data_normalized)
    
    # Tính mixing matrix
    mixing_matrix = ica.mixing_
    
    # Loại bỏ components chứa nhiễu (ví dụ: components có biên độ cao bất thường)
    # Có thể điều chỉnh ngưỡng tùy theo dữ liệu
    var_threshold = np.percentile(np.var(components, axis=0), 90)
    components[:, np.var(components, axis=0) > var_threshold] = 0
    
    # Khôi phục tín hiệu đã làm sạch
    cleaned_data = np.dot(components, mixing_matrix.T)
    
    # Khôi phục lại scale ban đầu
    cleaned_data = cleaned_data * np.std(data, axis=0) + np.mean(data, axis=0)
    
    return cleaned_data

row, col = data_focus['eeg_record4'].shape

# lọc tín hiệu có tần số cao hơn 0.16Hz
for name in trail_names:
    for i in range(col):
        data_focus[name][:, i] = combined_filter(data_focus[name][:, i], fs=128)
        data_unfocus[name][:, i] = combined_filter(data_unfocus[name][:, i], fs=128)
        data_drowsy[name][:, i] = combined_filter(data_drowsy[name][:, i], fs=128)

# Sau phần lọc tín hiệu combined_filter, thêm đoạn code sau:
for name in trail_names:
    # Áp dụng ICA cho từng trạng thái
    data_focus[name] = apply_ica(data_focus[name])
    data_unfocus[name] = apply_ica(data_unfocus[name])
    data_drowsy[name] = apply_ica(data_drowsy[name])

"""## 2. Features Engineering"""

"""Sử dụng biến đổi Fourier bằng STFT

Trong quá trình xử lý tín hiệu EEG trong tập lệnh trên, Biến đổi Fourier được sử dụng để chuyển đổi tín hiệu từ miền thời gian sang miền tần số. Việc sử dụng biến đổi Fourier, cụ thể là Biến đổi Fourier ngắn hạn (STFT), mang lại nhiều lợi ích trong việc phân tích và trích xuất đặc trưng từ tín hiệu EEG.

1. Lý do sử dụng Biến đổi Fourier trong xử lý tín hiệu EEG:

Phân tích thành phần tần số: Tín hiệu EEG là một dạng tín hiệu phức tạp, chứa đựng thông tin ở nhiều dải tần số khác nhau. Mỗi dải tần số tương ứng với các hoạt động não bộ cụ thể:
Delta (0.5 - 4 Hz): Liên quan đến giấc ngủ sâu.
Theta (4 - 8 Hz): Liên quan đến buồn ngủ, mơ mộng.
Alpha (8 - 12 Hz): Liên quan đến thư giãn, tỉnh táo.
Beta (12 - 30 Hz): Liên quan đến tập trung, hoạt động nhận thức.
Trích xuất đặc trưng hữu ích: Việc chuyển tín hiệu sang miền tần số cho phép trích xuất công suất tại các dải tần số quan trọng, giúp phân biệt các trạng thái não bộ khác nhau.
Hiểu rõ hơn về hoạt động não bộ: Phân tích tần số giúp xác định các mẫu sóng đặc trưng, hỗ trợ trong việc chẩn đoán và nghiên cứu khoa học.
"""

# STFT was then calculated at a time step of 1 s producing a set of time-varying DFT
# amplitudes X STFT (t,ω) at 1s intervals within each input EEG channel.
t_win = np.arange(0,128)
M = FS
window_blackman = 0.42 - 0.5*np.cos((2*np.pi*t_win)/(M-1)) + 0.08*np.cos((4*np.pi*t_win)/(M-1))

# tạo các matrix lưu công suất phổ( power spectrogram) cho 3 trạng thái
power_focus = {}
power_unfocus = {}
power_drowsy = {}

for name in trail_names:
    power_focus[name] = np.zeros([col, 513, 601])
    power_unfocus[name] = np.zeros([col, 513, 601])
    power_drowsy[name] = np.zeros([col, 513, 601])

# biến đổi tín hiệu từ miền thời gian sang miền tần số
for name in trail_names:
    for i in range(col):
        f, t,y1 = scipy.signal.stft(data_focus[name][:,i],fs=128, window=window_blackman, nperseg=128,
                      noverlap=0, nfft=1024, detrend=False,return_onesided=True, boundary='zeros',
                      padded=True)
        f, t,y2 = scipy.signal.stft(data_unfocus[name][:,i],fs=128, window=window_blackman, nperseg=128,
                      noverlap=0, nfft=1024, detrend=False,return_onesided=True, boundary='zeros',
                      padded=True)
        f, t,y3 = scipy.signal.stft(data_drowsy[name][:,i],fs=128, window=window_blackman, nperseg=128,
                      noverlap=0, nfft=1024, detrend=False,return_onesided=True, boundary='zeros',
                      padded=True)
        # tính toán spectrogram cho từng trạng thái
        power_focus[name][i,:,:] = (np.abs(y1))**2
        power_unfocus[name][i,:,:] = (np.abs(y2))**2
        power_drowsy[name][i,:,:] = (np.abs(y3))**2


plot_comparison_spectrograms(
    channel_idx1=channel1_idx,
    channel_idx2=channel2_idx,
    trail_name=trail_name,
    useful_channels_names=['F7', 'F3', 'P7', 'O1', 'O2', 'P8', 'AF4'],
    power_focus=power_focus,
    power_unfocus=power_unfocus,
    power_drowsy=power_drowsy
)

"""nfft xác định số điểm DFT(Discrete Fourier Transform)

$[ \text{Độ phân giải tần số} = \frac{\text{Tần số lấy mẫu (fs)}}{\text{nfft}} = \frac{128}{0.125} = 0.125]$

Nghĩa là sẽ có các mẫu tần số tại 0 Hz, 0.125 Hz, 0.25 Hz,...
"""

power_focus[trail_names[0]].shape

useful_channels_names

power_focus[trail_names[0]][0,:,:]

"""Chữ cái đầu tiên chỉ vùng chính trên da đầu:

F: Trán (Frontal)

P: Đỉnh đầu (Parietal)

O: Hậu đầu (Occipital)

C: Trung tâm (Central)

T: Thái dương (Temporal)

AF: Khu vực trước trán (Anterior Frontal)

Số sau chữ cái chỉ hướng trái (1-9 từ trái sang phải với 0 là trung tâm).
"""

def plot_comparison_spectrograms(channel_idx1, channel_idx2, trail_name, useful_channels_names,
                                 power_focus, power_unfocus, power_drowsy):
    """
    Vẽ và so sánh các spectrograms của hai kênh EEG trên ba trạng thái (focus, unfocus, drowsy).

    Parameters:
    - channel_idx1, channel_idx2: int, chỉ số của hai kênh EEG cần so sánh.
    - trail_name: str, tên thí nghiệm (trail).
    - useful_channels_names: list, tên các kênh EEG tương ứng với chỉ số.
    - power_focus, power_unfocus, power_drowsy: dict, dữ liệu spectrogram cho các trạng thái.

    """
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))  # Tạo lưới 3x2: 3 trạng thái x 2 kênh

    # Tên kênh
    channel1_name = useful_channels_names[channel_idx1]
    channel2_name = useful_channels_names[channel_idx2]

    # Trạng thái 1: Focus
    power1_focus = power_focus[trail_name][channel_idx1, :, :]
    power2_focus = power_focus[trail_name][channel_idx2, :, :]

    axs[0, 0].imshow(power1_focus, aspect='auto', origin='lower', cmap='jet',
                     extent=[0, power1_focus.shape[1], 0, 64])
    axs[0, 0].set_title(f'Focus - {channel1_name}')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('Frequency (Hz)')

    axs[0, 1].imshow(power2_focus, aspect='auto', origin='lower', cmap='jet',
                     extent=[0, power2_focus.shape[1], 0, 64])
    axs[0, 1].set_title(f'Focus - {channel2_name}')
    axs[0, 1].set_xlabel('Time (s)')

    # Trạng thái 2: Unfocus
    power1_unfocus = power_unfocus[trail_name][channel_idx1, :, :]
    power2_unfocus = power_unfocus[trail_name][channel_idx2, :, :]

    axs[1, 0].imshow(power1_unfocus, aspect='auto', origin='lower', cmap='jet',
                     extent=[0, power1_unfocus.shape[1], 0, 64])
    axs[1, 0].set_title(f'Unfocus - {channel1_name}')
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].set_ylabel('Frequency (Hz)')

    axs[1, 1].imshow(power2_unfocus, aspect='auto', origin='lower', cmap='jet',
                     extent=[0, power2_unfocus.shape[1], 0, 64])
    axs[1, 1].set_title(f'Unfocus - {channel2_name}')
    axs[1, 1].set_xlabel('Time (s)')

    # Trạng thái 3: Drowsy
    power1_drowsy = power_drowsy[trail_name][channel_idx1, :, :]
    power2_drowsy = power_drowsy[trail_name][channel_idx2, :, :]

    axs[2, 0].imshow(power1_drowsy, aspect='auto', origin='lower', cmap='jet',
                     extent=[0, power1_drowsy.shape[1], 0, 64])
    axs[2, 0].set_title(f'Drowsy - {channel1_name}')
    axs[2, 0].set_xlabel('Time (s)')
    axs[2, 0].set_ylabel('Frequency (Hz)')

    axs[2, 1].imshow(power2_drowsy, aspect='auto', origin='lower', cmap='jet',
                     extent=[0, power2_drowsy.shape[1], 0, 64])
    axs[2, 1].set_title(f'Drowsy - {channel2_name}')
    axs[2, 1].set_xlabel('Time (s)')

    # Bố trí đồ thị
    plt.tight_layout()
    plt.show()

# Sử dụng hàm
channel1_idx = 4  # 'AF4'
channel2_idx = 2  # 'P7'
trail_name = trail_names[0]  # Ví dụ trail đầu tiên

plot_comparison_spectrograms(
    channel_idx1=channel1_idx,
    channel_idx2=channel2_idx,
    trail_name=trail_name,
    useful_channels_names=['F7', 'F3', 'P7', 'O1', 'O2', 'P8', 'AF4'],  # Tên các kênh
    power_focus=power_focus,
    power_unfocus=power_unfocus,
    power_drowsy=power_drowsy
)

# Hàm tính công suất theo dải tần số
def compute_band_powers(power_spectrogram, freqs, freq_bands):
    band_powers = {band: [] for band in freq_bands}
    for band, (low, high) in freq_bands.items():
        idx = np.where((freqs >= low) & (freqs <= high))[0]
        band_powers[band] = power_spectrogram[idx, :].mean(axis=0)
    return band_powers

# Hàm vẽ đồ thị so sánh giữa hai kênh
def plot_band_power_comparison(channel_idx1, channel_idx2, trail_name, useful_channels_names,
                               power_focus, power_unfocus, power_drowsy, freq_bands):
    """
    So sánh công suất của các dải tần số giữa hai kênh EEG trên ba trạng thái (focus, unfocus, drowsing).

    Parameters:
    - channel_idx1, channel_idx2: int, chỉ số của hai kênh EEG cần so sánh.
    - trail_name: str, tên thí nghiệm (trail).
    - useful_channels_names: list, tên các kênh EEG.
    - power_focus, power_unfocus, power_drowsy: dict, dữ liệu spectrogram của các trạng thái.
    - freq_bands: dict, các dải tần số (band) và giới hạn tương ứng.
    """
    freqs = np.linspace(0, 64, power_focus[trail_name].shape[1])  # Tần số mẫu

    # Tính công suất theo dải tần số cho từng trạng thái và kênh
    band_powers1 = {
        "Focus": compute_band_powers(power_focus[trail_name][channel_idx1, :, :], freqs, freq_bands),
        "Unfocus": compute_band_powers(power_unfocus[trail_name][channel_idx1, :, :], freqs, freq_bands),
        "Drowsing": compute_band_powers(power_drowsy[trail_name][channel_idx1, :, :], freqs, freq_bands),
    }

    band_powers2 = {
        "Focus": compute_band_powers(power_focus[trail_name][channel_idx2, :, :], freqs, freq_bands),
        "Unfocus": compute_band_powers(power_unfocus[trail_name][channel_idx2, :, :], freqs, freq_bands),
        "Drowsing": compute_band_powers(power_drowsy[trail_name][channel_idx2, :, :], freqs, freq_bands),
    }

    # Tạo lưới 3x2: 3 trạng thái x 2 kênh
    fig, axs = plt.subplots(3, 2, figsize=(15, 12), sharex=True)

    channel1_name = useful_channels_names[channel_idx1]
    channel2_name = useful_channels_names[channel_idx2]

    for i, (state, band_powers1_state) in enumerate(band_powers1.items()):
        band_powers2_state = band_powers2[state]

        # channel 1
        for band, power in band_powers1_state.items():
            axs[i, 0].plot(power, label=f'{band} Band')
        axs[i, 0].legend()
        axs[i, 0].set_title(f'{state} State - {channel1_name}')
        axs[i, 0].set_ylabel('Power')
        axs[i, 0].grid()

        # channel 2
        for band, power in band_powers2_state.items():
            axs[i, 1].plot(power, label=f'{band} Band')
        axs[i, 1].legend()
        axs[i, 1].set_title(f'{state} State - {channel2_name}')
        axs[i, 1].grid()

    axs[-1, 0].set_xlabel('Time (s)')
    axs[-1, 1].set_xlabel('Time (s)')

    plt.tight_layout()
    plt.show()

# Sử dụng hàm
freq_bands = {'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 12), 'Beta': (12, 30)}
channel1_idx = 4  # 'AF4'
channel2_idx = 2  # 'P7'
trail_name = trail_names[0]

plot_band_power_comparison(
    channel_idx1=channel1_idx,
    channel_idx2=channel2_idx,
    trail_name=trail_name,
    useful_channels_names=['F7', 'F3', 'P7', 'O1', 'O2', 'P8', 'AF4'],
    power_focus=power_focus,
    power_unfocus=power_unfocus,
    power_drowsy=power_drowsy,
    freq_bands=freq_bands
)

print(power_focus['eeg_record4'][0, :, :])

# gộp các bin dữ liệu trong khoảng 0.5Hz và lấy trong khoảng 0-18Hz

num = []

power_focus_bin = {}
for name in trail_names:
    power_focus_bin[name] = np.zeros([7,36,601])

power_unfocus_bin = {}
for name in trail_names:
    power_unfocus_bin[name] = np.zeros([7,36,601])

power_drowsy_bin = {}
for name in trail_names:
    power_drowsy_bin[name] = np.zeros([7,36,601])

for name in trail_names:
    for chn in range(col):
        j=0
        for i in range(1,144,4):
            power_focus_bin[name][chn,j,:] = np.average(power_focus[name][chn,i:i+4,:],axis=0)
            power_unfocus_bin[name][chn,j,:] = np.average(power_unfocus[name][chn,i:i+4,:],axis=0)
            power_drowsy_bin[name][chn,j,:] = np.average(power_drowsy[name][chn,i:i+4,:],axis=0)
            j=j+1

power_focus_bin['eeg_record11'].shape

# avarage over 15 seconds running window.

power_focus_ave = {}
for name in trail_names:
    power_focus_ave[name] = np.zeros([7,36,585])

power_unfocus_ave = {}
for name in trail_names:
    power_unfocus_ave[name] = np.zeros([7,36,585])

power_drowsy_ave = {}
for name in trail_names:
    power_drowsy_ave[name] = np.zeros([7,36,585])

for name in trail_names:
    for chn in range(col):
        j=0
        for k in range(0,585):
            power_focus_ave[name][chn,:,j] = np.average(power_focus_bin[name][chn,:,k:k+15],axis=1)
            power_unfocus_ave[name][chn,:,j] = np.average(power_unfocus_bin[name][chn,:,k:k+15],axis=1)
            power_drowsy_ave[name][chn,:,j] = np.average(power_drowsy_bin[name][chn,:,k:k+15],axis=1)
            j=j+1

power_focus_bin['eeg_record11'].shape #(7, 36, 601)


# Turn the data into a vector 

focus_features = {}
for name in trail_names:
    focus_features[name] = np.zeros([252,300])
    
unfocus_features = {}
for name in trail_names:
    unfocus_features[name] = np.zeros([252,300])
    
drowsy_features = {}
for name in trail_names:
    drowsy_features[name] = np.zeros([252,300])

for name in trail_names:
    for j in range(300):      
        focus_features[name][:,j] = power_focus_ave[name][:,:,j].reshape(1,-1)
        unfocus_features[name][:,j] = power_unfocus_ave[name][:,:,j].reshape(1,-1)
        drowsy_features[name][:,j] = power_drowsy_ave[name][:,:,j].reshape(1,-1)
    focus_features[name] = 10*np.log(focus_features[name])
    unfocus_features[name] = 10*np.log(unfocus_features[name])
    drowsy_features[name] = 10*np.log(drowsy_features[name])

"""#### Thử nghiệm với CNN"""

arr = np.arange(new_subj.shape[0])
np.random.shuffle(arr)
data_train, data_test = new_subj[arr[:int(0.8 * new_subj.shape[0])]], new_subj[arr[int(0.8 * new_subj.shape[0]):]]
data_train_target, data_test_target = new_target[arr[:int(0.8 * new_subj.shape[0])]], new_target[arr[int(0.8 * new_subj.shape[0]):]]

import torch

device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
device=torch.device(device_name)
print(f"Device: {device}")


