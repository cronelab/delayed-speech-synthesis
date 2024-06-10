import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io.wavfile import read as wavread
from scipy.signal import spectrogram

# Plot configuration
reco_indices = [1, 2, 3, 4, 5, 6]
reco_xticks = [15000, 42000, 72000, 91000, 121000, 155000]  # Positions of the x-ticks in the waveform plot
reco_words = ["Enter", "Right", "Back", "Up", "Left", "Down"]  # Labels of the x-ticks

# The run folder contains the data written out by the online decoder.
# The original speech file corresponds to the microphone recording of the specific session of the closed-loop run.
# The paper uses the data from 2023_04_18.
run_folder = "... path ... to ... output ... folder ... from ... decode_online.py"
orig_speech_file = "... path ... to ... microphone ... recording"

# Concatenate examples from patient speech
vad = pd.read_csv(os.path.join(run_folder, "log.vad.lab"), sep="\t", names=["start", "stop", "label"])
orig_audio = wavread(orig_speech_file)[1]
orig_samples = [orig_audio[int(vad.iloc[i].start * 16000):int(vad.iloc[i].stop * 16000)]for i in reco_indices]
orig_samples = np.hstack(orig_samples)

# Concatenate some examples from the synthesizer output
reco_samples = [wavread(os.path.join(run_folder, "reco", f"reco_{i+1:05d}.wav"))[1] for i in reco_indices]
reco_samples = np.hstack(reco_samples)

# Create spectrogram from the synthesized speech
f_orig, t_orig, Sxx_orig = spectrogram(orig_samples, 16000, mode='magnitude', window='hann', nperseg=800, noverlap=640)
f_reco, t_reco, Sxx_reco = spectrogram(reco_samples, 16000, mode='magnitude', window='hann', nperseg=800, noverlap=640)

fig, ((ax_orig_wav, ax_orig_spec), (ax_reco_wav, ax_reco_spec)) = plt.subplots(2, 2, figsize=(11, 4.5))

ax_orig_wav.plot(orig_samples)
ax_orig_wav.set_title("Patient's Original Speech", loc="left", fontsize=10)
ax_orig_wav.set_xlim(0, len(reco_samples))
ax_orig_wav.set_ylabel("Amplitude")
ax_orig_wav.set_yticks([])
ax_orig_wav.spines["top"].set_visible(False)
ax_orig_wav.spines["bottom"].set_visible(False)
ax_orig_wav.spines["left"].set_visible(False)
ax_orig_wav.spines["right"].set_visible(False)
ax_orig_wav.set_xticks(reco_xticks)
ax_orig_wav.set_xticklabels([])

# Bracket for 1s time period
time_base = 140_000
half_second = 8000
props = {'connectionstyle': 'bar', 'arrowstyle': '-', 'shrinkA': 8, 'shrinkB': 8, 'linewidth': 1.5}
ax_orig_wav.annotate("1 s", xy=(time_base - 4000, 12_500), zorder=10)
ax_orig_wav.annotate('', xy=(time_base - half_second, 10_000), xytext=(time_base + half_second, 10_000),
                     arrowprops=props)

ax_orig_spec.imshow(10 * np.log10(Sxx_orig), aspect="auto", origin="lower", cmap='Blues',
                    extent=[0, Sxx_orig.shape[1], 0, 8000])

ax_orig_spec.set_yscale('log')
ax_orig_spec.set_ylim(100, 8000)
ax_orig_spec.set_xticks((np.array(reco_xticks) / 16000) / 0.01)
ax_orig_spec.set_xticklabels([])
ax_orig_spec.set_ylabel("Frequency [log Hz]")
ax_orig_spec.yaxis.tick_right()

time_base = ((time_base / 16000) / 0.01)
half_second = ((half_second / 16000) / 0.01)
props = {'connectionstyle': 'bar', 'arrowstyle': '-', 'shrinkA': 8, 'shrinkB': 8, 'linewidth': 1.5, "edgecolor": "black"}

ax_orig_spec.annotate("1 s", xy=(time_base - 25, 4500), zorder=10, color="black")
ax_orig_spec.annotate('', xy=(time_base - half_second, 2800), xytext=(time_base + half_second, 2800), arrowprops=props)

ax_reco_wav.plot(reco_samples)
ax_reco_wav.set_xlim(0, len(reco_samples))
ax_reco_wav.set_ylabel("Amplitude")
ax_reco_wav.set_yticks([])
ax_reco_wav.spines["top"].set_visible(False)
ax_reco_wav.spines["bottom"].set_visible(False)
ax_reco_wav.spines["left"].set_visible(False)
ax_reco_wav.spines["right"].set_visible(False)
ax_reco_wav.set_xticks(reco_xticks)
ax_reco_wav.set_xticklabels(reco_words)
ax_reco_wav.set_title("Closed-Loop Synthesis Output", loc="left", fontsize=10)

ax_reco_spec.imshow(10 * np.log10(Sxx_reco), aspect="auto", origin="lower", cmap='Blues',
                    extent=[0, Sxx_reco.shape[1], 0, 8000])

ax_reco_spec.set_yscale('log')
ax_reco_spec.set_ylim(100, 8000)
ax_reco_spec.set_xticks((np.array(reco_xticks) / 16000) / 0.01)
ax_reco_spec.set_xticklabels(reco_words)
ax_reco_spec.set_ylabel("Frequency [log Hz]")
ax_reco_spec.yaxis.tick_right()

plt.tight_layout()
plt.subplots_adjust(hspace=0.35, wspace=0.1)
plt.show()
# plt.savefig("plots/figure_2ab.png", dpi=300)
