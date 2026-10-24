"""Write the demo test stems: two talkers and music, 12 s, 16 kHz mono WAV, equal speech
levels, music 5 dB lower. Sources are librosa's example recordings (LibriSpeech readers and a
Brahms recording; openly licensed, downloaded on first use).

  uv run --with librosa python -m mockclient.demo_stems sessions/stems
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.io import wavfile

SR, DUR = 16000, 12.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    import librosa

    a.out.mkdir(parents=True, exist_ok=True)
    n = int(SR * DUR)
    for name, key, offset, gain_db in [("voice1", "libri2", 0.0, 0.0), ("voice2", "libri3", 0.0, 0.0),
                                       ("music", "brahms", 5.0, -5.0)]:
        y, _ = librosa.load(librosa.ex(key), sr=SR, offset=offset, duration=DUR)
        y = np.pad(y, (0, n - len(y)))
        y = y / (np.sqrt(np.mean(y**2)) + 1e-9) * 0.05 * 10 ** (gain_db / 20)
        wavfile.write(a.out / f"{name}.wav", SR, y.astype(np.float32))
        print(a.out / f"{name}.wav")


if __name__ == "__main__":
    main()
