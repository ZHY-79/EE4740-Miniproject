import wave
import numpy as np

def analyze_wav(file_path):

    try:
        # wav
        with wave.open(file_path, 'rb') as wav_file:
            n_channels = wav_file.getnchannels()    # 声道数
            sample_width = wav_file.getsampwidth()  # 样本宽度（字节）
            frame_rate = wav_file.getframerate()    # 采样率（Hz）
            n_frames = wav_file.getnframes()        # 帧数
            duration = n_frames / frame_rate        # 持续时间（秒）
            signal = wav_file.readframes(n_frames)
        
        if sample_width == 1:  # 8位音频
            dtype = np.uint8
        elif sample_width == 2:  # 16位音频
            dtype = np.int16
        else:
            pass
        signal_array = np.frombuffer(signal, dtype=dtype)
        if n_channels == 2:
            signal_array = signal_array[::2]
        
        time = np.linspace(0, duration, num=len(signal_array))
        results = {
            'channels': n_channels,
            'sample_width': sample_width,
            'frame_rate': frame_rate,
            'n_frames': n_frames,
            'duration': duration
        }
        
        return results, signal_array, time
    
    except Exception as e:
        raise Exception(f"Error: {str(e)}")


