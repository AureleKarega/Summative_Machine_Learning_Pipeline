"""Audio feature extraction shared by training, the API and the notebook."""
import io
import numpy as np
import librosa

SR = 22050            # sample rate everything is resampled to
DURATION = 5.0        # ESC-10 clips are 5 s long
N_MFCC = 13

FEATURE_NAMES = (
    [f"mfcc{i}_mean" for i in range(1, N_MFCC + 1)]
    + [f"mfcc{i}_std" for i in range(1, N_MFCC + 1)]
    + ["spectral_centroid_mean", "spectral_centroid_std", "spectral_bandwidth_mean",
       "spectral_rolloff_mean", "spectral_flatness_mean", "zcr_mean", "rms_mean", "rms_std"]
)


def load_audio(source):
    """Load a path / bytes / file-like object as mono float audio at SR (max DURATION s)."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    y, _ = librosa.load(source, sr=SR, mono=True, duration=DURATION)
    return y


def extract_features(y, sr=SR):
    """Return a dict {feature_name: value} for one waveform."""
    if len(y) < int(0.5 * sr):
        raise ValueError("Audio is shorter than 0.5 seconds")
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    cen = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    flat = librosa.feature.spectral_flatness(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]
    vals = (list(mfcc.mean(axis=1)) + list(mfcc.std(axis=1))
            + [cen.mean(), cen.std(), bw.mean(), roll.mean(), flat.mean(), zcr.mean(), rms.mean(), rms.std()])
    return dict(zip(FEATURE_NAMES, map(float, vals)))
