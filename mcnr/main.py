import librosa
import soundfile
import numpy as np

def load_audio_file(filename):
    # Load an audio file and return it as a numpy array
    y, sr = librosa.load(filename, sr=None, mono=False)

    # check if sr is 16000
    assert sr == 16000, f"Sample rate of {sr} is not supported. Please resample to 16000."

    # check if y is multi-channel
    assert y.ndim > 1, f"Audio file must be multi-channel."

    return y

def save_audio_file(filename, x, sr=16000):
    # Save an audio file
    soundfile.write(filename, x.T, sr, subtype="PCM_16")

    return None


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


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Filtering the multi-channel audio signal")
    parser.add_argument("--input", "-i", type=str, help="Input audio file", required=True)
    parser.add_argument("--fft_size", type=int, default=512, help="FFT size (default: 512)")
    parser.add_argument("--hop_size", type=int, default=128, help="Hop size (default: 128)")
    parser.add_argument("output", type=str, help="Output audio file")

    args = parser.parse_args()

    # Load an audio file
    x = load_audio_file(args.input)

    # Compute the short-time Fourier transform of the audio signal
    d = proc_stft(x, fft_size=args.fft_size, hop_size=args.hop_size)

    # Filtering
    d = proc_filtering(d)

    # Compute the inverse short-time Fourier transform of the complex-valued spectrogram
    x = proc_istft(d, hop_size=args.hop_size)

    # Save the output audio file
    save_audio_file(args.output, x)

if __name__ == "__main__":
    main()
