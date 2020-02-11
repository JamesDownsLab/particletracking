import numpy as np


def duty(video_filename, num_frames):
    duty_cycle = read_audio_file(video_filename, num_frames)
    duty_cycle = np.uint16(duty_cycle)
    return duty_cycle


def read_audio_file(file, frames):
    wav = audio.extract_wav(file)
    wav_l = wav[:, 0]
    freqs = frame_frequency(wav_l, frames, 48000)
    d = (freqs - 1000) / 15
    return d


def frame_frequency(wave, frames, audio_rate):
    waves = np.array_split(wave, frames)
    freqs = np.array(
        [fourier_transform_peak(wav, 1 / audio_rate) for wav in waves])
    return freqs


def fourier_transform_peak(sig, time_step):
    ft = np.abs(np.fft.fft(sig, 48000))
    freq = np.fft.fftfreq(48000, time_step)
    peak = np.argmax(ft)
    return abs(freq[peak])