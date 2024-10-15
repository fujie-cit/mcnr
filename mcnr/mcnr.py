import librosa
import numpy as np


def proc_stft(x, fft_size=512, hop_size=128):
    """Compute the short-time Fourier transform of an audio signal.

    Args:
        x (np.ndarray): Input audio signal. (n_channels, n_samples)

    Returns:
        np.ndarray: Short-time Fourier transform of the input audio signal. (n_channels, n_fft//2 + 1, n_frames)
    """
    # Compute the short-time Fourier transform of an audio signal
    d = librosa.stft(x, n_fft=fft_size, hop_length=hop_size)

    return d


def proc_filtering(d, inplace=True):
    # calculate power spectrogram
    p = np.abs(d)**2

    # take the maimux channel index for each frame, each freq. bin
    p_max = np.argmax(p, axis=0)

    # make a copy of the input data
    d_out = d if inplace else d.copy()

    # leave only the maximum channel for each frame, each freq. bin
    for i in range(d.shape[0]):
        d[i, p_max != i] = 0.0 + 0.0j

    return d


def proc_istft(d, hop_size=128):
    # Compute the inverse short-time Fourier transform of a complex-valued spectrogram
    x = librosa.istft(d, hop_length=hop_size)

    return x


def do_multi_channel_noise_reduction(x, fft_size=512, hop_size=128):
    """Apply multi-channel noise reduction to an audio signal.

    Example:
        x = do_multi_channel_noise_reduction(x, fft_size=512, hop_size=128)

    Note:
        The input audio signal must have shape (n_channels, n_samples).
        Data type of the input audio signal must be float.

    Args:
        x (np.ndarray): Input audio signal. (n_channels, n_samples)
        fft_size (int): FFT size (default: 512)
        hop_size (int): Hop size (default: 128)

    Returns:
        np.ndarray: Output audio signal. (n_channels, n_samples)
    """
    # Compute the short-time Fourier transform of the audio signal
    d = proc_stft(x, fft_size=fft_size, hop_size=hop_size)

    # Filtering
    d = proc_filtering(d)

    # Compute the inverse short-time Fourier transform of the complex-valued spectrogram
    x = proc_istft(d, hop_size=hop_size)

    return x
