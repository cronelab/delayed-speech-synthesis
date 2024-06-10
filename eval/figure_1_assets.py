import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ArrowStyle
from pathlib import Path
from typing import Optional
from scipy.io.wavfile import read as wavread


def render_hga_feature_computation_plot(data_snippet: np.ndarray, out_filename: Optional[Path] = None, dpi=300):
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    ax.imshow(data_snippet.T, origin="lower", aspect="auto", cmap="PiYG", vmin=-4, vmax=4)
    ax.grid(False)

    ax.set_xticks([])
    ax.set_yticks([0, 63])
    ax.set_yticklabels([1, 82])
    ax.set_ylabel("Selected Channels", labelpad=-10)
    ax.set_facecolor("white")

    style_a = ArrowStyle("|-|", widthA=0.5, angleA=0, widthB=0.5, angleB=0)
    style_b = ArrowStyle("<->", widthA=0.5, angleA=0, widthB=0.5, angleB=0)
    props_a = {'arrowstyle': style_a, 'shrinkA': 0, 'shrinkB': 0, 'linewidth': 1., "color": "black"}
    props_b = {'arrowstyle': style_b, 'shrinkA': 0, 'shrinkB': 0, 'linewidth': 1., "color": "black"}
    ax.annotate("1 s", xy=(60, -8), zorder=10, color="black", annotation_clip=False)
    ax.annotate('', xy=(20, -4), xytext=(120, -4), arrowprops=props_a, annotation_clip=False)
    ax.annotate('', xy=(20, -4), xytext=(120, -4), arrowprops=props_b, annotation_clip=False)

    ax.set_title("High-γ Feature Computation")
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color("black")
    ax.spines['right'].set_visible(True)
    ax.spines['right'].set_color("black")
    ax.spines['top'].set_visible(True)
    ax.spines['top'].set_color("black")
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color("black")

    plt.tight_layout()
    if out_filename:
        plt.savefig(out_filename.as_posix(), dpi=dpi, transparent=True)
    else:
        plt.show()


def render_masked_hga_features_plot(data_snippet: np.ndarray, mask: np.ndarray, out_filename: Optional[Path] = None,
                                    dpi=300):
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    im_bar = ax.imshow(data_snippet.T, origin="lower", aspect="auto", cmap="PiYG", vmin=-4, vmax=4)
    ax.imshow(mask.T, origin="lower", aspect="auto", cmap="gray", alpha=0.3)
    ax.grid(False)

    ax.set_title("Speech Segment Extraction")
    ax.set_xticks([])
    ax.set_yticks([0, 63])
    ax.set_yticklabels([1, 82])
    ax.set_ylabel("Selected Channels", labelpad=-10)
    ax.set_facecolor("white")

    style_a = ArrowStyle("|-|", widthA=0.5, angleA=0, widthB=0.5, angleB=0)
    style_b = ArrowStyle("<->", widthA=0.5, angleA=0, widthB=0.5, angleB=0)
    props_a = {'arrowstyle': style_a, 'shrinkA': 0, 'shrinkB': 0, 'linewidth': 1., "color": "black"}
    props_b = {'arrowstyle': style_b, 'shrinkA': 0, 'shrinkB': 0, 'linewidth': 1., "color": "black"}
    ax.annotate("1 s", xy=(60, -8), zorder=10, color="black", annotation_clip=False)
    ax.annotate('', xy=(20, -4), xytext=(120, -4), arrowprops=props_a, annotation_clip=False)
    ax.annotate('', xy=(20, -4), xytext=(120, -4), arrowprops=props_b, annotation_clip=False)

    # ax.set_title("Speech Segment Extraction")
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color("black")
    ax.spines['right'].set_visible(True)
    ax.spines['right'].set_color("black")
    ax.spines['top'].set_visible(True)
    ax.spines['top'].set_color("black")
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color("black")

    # ax_ins = ax.inset_axes([0.2, 1.05, 0.8, 0.1])
    # fig.colorbar(im_bar, cax=ax_ins, orientation="horizontal", pad=20)
    # ax_ins.xaxis.set_ticks_position("top")
    # ax_ins.set_title("High-γ Activity [distance to baseline in STD]")

    plt.tight_layout()
    if out_filename:
        plt.savefig(out_filename.as_posix(), dpi=dpi, transparent=True)
    else:
        plt.show()


def render_lpc_features_plot(data_snippet: np.ndarray, out_filename: Optional[Path] = None, dpi=300):
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111)
    ax.imshow(data_snippet.T, origin="lower", aspect="auto", cmap="inferno")
    ax.grid(False)

    ax.set_xticks([])
    ax.set_yticks([0, 19])
    ax.set_yticklabels([1, 20])
    ax.set_ylabel("LPC Coefficients", labelpad=-10)
    ax.set_facecolor("white")

    style_a = ArrowStyle("|-|", widthA=0.4, angleA=0, widthB=0.3, angleB=0)
    style_b = ArrowStyle("<->", widthA=0.4, angleA=0, widthB=0.3, angleB=0)
    props_a = {'arrowstyle': style_a, 'shrinkA': 0, 'shrinkB': 0, 'linewidth': 1., "color": "black"}
    props_b = {'arrowstyle': style_b, 'shrinkA': 0, 'shrinkB': 0, 'linewidth': 1., "color": "black"}
    ax.annotate("1 s", xy=(65, -4), zorder=10, color="black", annotation_clip=False)
    ax.annotate('', xy=(20, -2), xytext=(120, -2), arrowprops=props_a, annotation_clip=False)
    ax.annotate('', xy=(20, -2), xytext=(120, -2), arrowprops=props_b, annotation_clip=False)

    ax.set_title("Estimated Vocoder Features")
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color("black")
    ax.spines['right'].set_visible(True)
    ax.spines['right'].set_color("black")
    ax.spines['top'].set_visible(True)
    ax.spines['top'].set_color("black")
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color("black")

    plt.tight_layout()
    if out_filename:
        plt.savefig(out_filename.as_posix(), dpi=dpi, transparent=True)
    else:
        plt.show()


def render_patient_and_synthesized_speech(orig_snippet: np.ndarray, reco_snippet: np.ndarray,
                                          out_filename: Optional[Path] = None, dpi=300):
    fig, (ax_orig, ax_reco) = plt.subplots(2, 1, figsize=(8, 2.5))

    xs = np.linspace(0, int(len(orig_snippet) / 16000), len(orig_snippet))

    ax_orig.plot(xs, orig_snippet, color="black")
    ax_orig.set_xlim(0, int(len(orig_snippet) // 16000))
    ax_orig.set_xticks([])
    ax_orig.spines["top"].set_visible(False)
    ax_orig.spines["bottom"].set_visible(False)
    ax_orig.spines["left"].set_visible(False)
    ax_orig.spines["right"].set_visible(False)
    ax_orig.set_ylabel("Participant")
    ax_orig.set_yticks([])

    ax_reco.plot(xs, reco_snippet, color="#9C0000")
    ax_reco.spines["top"].set_visible(False)
    ax_reco.spines["bottom"].set_visible(False)
    ax_reco.spines["left"].set_visible(False)
    ax_reco.spines["right"].set_visible(False)
    ax_reco.set_ylabel("Synthesizer")
    ax_reco.set_xlim(0, int(len(reco_snippet) // 16000))
    ax_reco.set_xticks([])
    ax_reco.set_yticks([])
    ax_reco.set_xlabel("Time [s]")

    props = {'connectionstyle': 'bar', 'arrowstyle': '-', 'shrinkA': 4, 'shrinkB': 4, 'linewidth': 1.5,
             "edgecolor": "black"}

    ax_reco.annotate("1 s", xy=(3.15, -20_500), zorder=100, color="black", annotation_clip=False)
    ax_reco.annotate('', xy=(4, -12_000), xytext=(3, -12_000), arrowprops=props)

    plt.tight_layout()
    if out_filename:
        plt.savefig(out_filename.as_posix(), dpi=dpi, transparent=True)
    else:
        plt.show()


def render_colorbar(out_filename: Optional[Path] = None, dpi=300):
    fig = plt.figure(figsize=(4, 0.9))
    ax = fig.add_subplot(111)

    ax.set_xlabel("High-γ Activity [distance to baseline in STDs]")
    ax.set_yticks([])

    ax.imshow(np.linspace(-4, 4, 400, endpoint=True).reshape((1, -1)), aspect="auto", cmap="PiYG", vmin=-4, vmax=4)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xticks(np.linspace(0, 400, 9))
    ax.set_xticklabels(np.arange(-4, 5))
    plt.tight_layout()
    if out_filename:
        plt.savefig(out_filename.as_posix(), dpi=dpi, transparent=True)
    else:
        plt.show()


if __name__ == "__main__":
    # Plot configuration
    to_file = False

    start = 46
    stop = 50

    lpc_start = 821
    lpc_stop = 983

    audio_snippet_start = 0
    audio_snippet_stop = 25

    hga_file = "/path/to/log.hga.f64"
    lpc_file = "/path/to/log.lpc.f32"
    orig_speech_file = "/path/to/video/segment/patient_speech_segment.wav"
    reco_speech_file = "/path/to/video/segment/synthesized_speech_segment.wav"

    hga = np.fromfile(hga_file, dtype=np.float64).reshape((-1, 64))
    lpc = np.fromfile(lpc_file, dtype=np.float32).reshape((-1, 20))
    orig_speech = wavread(orig_speech_file)[1]
    reco_speech = wavread(reco_speech_file)[1]

    # Render assets
    render_colorbar(out_filename=Path("plots/figure_1_cb.png") if to_file else None)

    data_snippet = hga[int(start * 100):int(stop * 100), :64]
    render_hga_feature_computation_plot(data_snippet=data_snippet,
                                        out_filename=Path("plots/figure_1_b.png") if to_file else None)

    mask = np.zeros_like(data_snippet)
    mask[int((47.63 - start) * 100):int((49.25 - start) * 100), :] = np.NaN
    render_masked_hga_features_plot(data_snippet=data_snippet, mask=mask,
                                    out_filename=Path("plots/figure_1_d.png") if to_file else None)

    data_snippet = lpc[lpc_start:lpc_stop, :]
    render_lpc_features_plot(data_snippet=data_snippet,
                             out_filename=Path("plots/figure_1_f.png") if to_file else None)

    orig_speech_snippet = orig_speech[int(audio_snippet_start * 16000): int(audio_snippet_stop * 16000)]
    reco_speech_snippet = reco_speech[int(audio_snippet_start * 16000): int(audio_snippet_stop * 16000)]
    render_patient_and_synthesized_speech(orig_snippet=orig_speech_snippet, reco_snippet=reco_speech_snippet,
                                          out_filename=Path("plots/figure_1_g.png") if to_file else None)
