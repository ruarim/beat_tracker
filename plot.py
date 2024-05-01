from matplotlib import pyplot as plt
import librosa
import numpy as np

def plot_onset_envelope_strength(y, sr, onset_strength, hop_sec):
    hop_length = int(hop_sec * sr)
    
    # normalise range 0 - 1
    onset_strength = onset_strength / np.max(onset_strength)
    
    # Time vector for plotting
    times = librosa.frames_to_time(np.arange(len(onset_strength)), sr=sr, hop_length=hop_length)
    
    # Plotting
    plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(0, (len(y) / sr), num=len(y)), y, alpha=0.6)  # Plot waveform
    plt.plot(times, onset_strength)
    plt.title('Onset Strength Envelope')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Normalized Onset Strength')
    plt.xlim([0, 10])
    plt.ylim([-0.1, 1.1])
    plt.show()

def plot_mel_spectrogram(mel_db, sr):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_db, y_axis='mel', sr=sr, fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram (dB)')
    plt.tight_layout()
    plt.show()
    
def plot_auto_c(auto_c, tps, faster_tempo, slower_tempo, selected_tempo, sr, hop_length):
    # calculate bpm for plot
    faster_tempo_bpm = round(60 * sr / (faster_tempo * hop_length), 2)
    slower_tempo_bpm = round(60 * sr / (slower_tempo * hop_length), 2)
    selected_tempo_bpm = round(60 * sr / (selected_tempo * hop_length), 2)
    
    auto_c = auto_c / np.max(auto_c)
    tps = tps / np.max(tps)
    
    times = np.arange(len(auto_c)) * (hop_length / sr)

    plt.plot(times, auto_c)
    plt.title('Auto-Correlation')
    plt.xlabel('Lag (seconds)')
    plt.show()

    plt.plot(times, tps)
    plt.vlines(faster_tempo * (hop_length / sr), ymin=min(tps), ymax=max(tps), color='g', linestyle='--', label='faster tempo: {} BPM'.format(faster_tempo_bpm))
    plt.vlines(slower_tempo * (hop_length / sr), ymin=min(tps), ymax=max(tps), color='r', linestyle='--', label='slower tempo: {} BPM'.format(slower_tempo_bpm))
    plt.scatter(selected_tempo * (hop_length / sr), max(tps), color='red', zorder=5, label='Selected BPM')
    plt.legend()
    plt.title('Perception Weighted Auto-Correlation Tempo BPM: {}'.format(selected_tempo_bpm))
    plt.xlabel('Lag (seconds)')
    plt.show()

def plot_dynamic_programming(c_score, backlink, ose, beats):
    # Normalize values for better visualisation
    ose = (ose - np.min(ose)) / (np.max(ose) - np.min(ose))
    c_score = (c_score - np.min(c_score)) / (np.max(c_score) - np.min(c_score))
    
    # Plot the cumulative score and onset strength envelope
    plt.figure(figsize=(14, 6))
    plt.plot(ose, label='Onset Strength Envelope', color='blue', alpha=0.5)
    plt.plot(c_score, label='Cumulative Score', color='orange', alpha=0.8)

    # Add  backlink arrows
    for i in range(1, len(c_score)):
        if backlink[i] != -1:
            plt.annotate('', xy=(i, c_score[i]), xytext=(backlink[i], c_score[backlink[i]]),
                        arrowprops=dict(arrowstyle="<-", color='gray', alpha=0.5))

    # highlight the beats
    plt.scatter(beats, c_score[beats], color='red', zorder=5, label='Beats')

    plt.title('Dynamic Programming Beat Tracking Visualization')
    plt.xlabel('Time (frames)')
    plt.ylabel('Normalized Score')
    plt.legend()
    plt.grid(True)
    plt.xlim([100, 750])
    plt.ylim([0.0, 0.2])
    plt.show()
    
def plot_estimated_vs_annotation(y, sr, beats_estimates, annotations, name, genre, xlim=[10, 20],):
    plt.figure(figsize=(10, 4))
    plt.title('Beats {} {}'.format(name, genre)) # add bpm
    plt.plot(np.linspace(0, (len(y) / sr), num=len(y)), y, alpha=0.6)  # plot waveform
    plt.vlines(beats_estimates, ymin=0, ymax=max(y), color='r', linestyle='--', label='Estimates')  # plot beats
    plt.vlines(annotations, ymin=min(y), ymax=0, color='g', linestyle='--', label='Annotations')  # plot annotations
    plt.legend()
    plt.xlim(xlim)