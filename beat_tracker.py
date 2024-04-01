import numpy as np
import librosa
from matplotlib import pyplot as plt
from scipy.signal import lfilter

def mel_db(sr, stft, n_fft, hop_length, n_mels=40):
    abs_stft = abs(stft)
    
    # create a Mel spectrogram with 40 Mel bands
    mel = librosa.feature.melspectrogram(S=abs_stft**2, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    
    # convert power spectrogram (amplitude squared) to dB
    mel_db = librosa.power_to_db(mel)
    
    return mel_db
    
# onset strength envelope
def onset_strength_envelope(y, sr, window_sec=0.032, hop_sec=0.004, show=False):    
    # calculate the STFT with a 32ms window and 4ms hop size
    n_fft = int(window_sec * sr)  # window seconds to samples
    hop_length = int(hop_sec * sr)  # hop seconds to samples
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

    # apply mel scale with 40 bands
    mel = mel_db(sr, stft, n_fft, hop_length)
    
    # apply first order difference
    mel_dif = np.diff(mel, axis=1)
    
    # half wave rectify
    mel_half_rec = np.maximum(0, mel_dif)
    
    # sum across bands
    onset_strength = np.sum(mel_half_rec , axis=0)
    
    # remove DC (0 Hz) component 
    onset_strength = lfilter([1.0, -1.0], [1.0, -0.99], onset_strength, axis=-1)
    
    # normalise to standard deviation (add epsilon to avoid div by 0) 
    # helps find values with significant difference from mean   
    onset_strength_norm  = onset_strength / np.std(onset_strength) + 1e-10 
    
    if(show): 
        plot_onset_envelope_strength(y, sr, onset_strength_norm, hop_sec)
        # plot_mel_spectrogram(mel, sr)
        
    return onset_strength_norm

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
    
# Apply perceptual weighting based on tempo tapping data
def perceptual_weighting(t, t0=0.5, sigma_t=0.9): # t0 (bpm) and sigma_t (octaves) values from ellis 2007  
    # avoid div by zero error 
    epsilon = 1e-10
    t = np.maximum(t, epsilon)
    
    # create gaussian window based on ellis 2007
    log_2_t_over_t0 = np.log2(t / t0)
    W_t = np.exp(-0.5 * (log_2_t_over_t0 / sigma_t) ** 2)
    
    return W_t

def tempo_multiples(tps):
    # only search 1/3 of the tps this should be plenty with 5 second tps
    tps_range = round(len(tps) / 3)
    
    tps_2 = np.zeros(tps_range)
    tps_3 = np.zeros(tps_range)

    for t in range(tps_range):
        # equation (7) and (8) - ellis 2007
        tps_2[t] = tps[t] + (0.5 * tps[2 * t]) + (0.25 * tps[(2 * t) - 1]) + (0.25 * tps[(2 * t) + 1])
        tps_3[t] = tps[t] + (0.33 * tps[3 * t]) + (0.33 * tps[(3 * t) - 1]) +(0.33 * tps[(3 * t) + 1])            
        
    duple = np.max(tps_2)
    triple = np.max(tps_3)
     
    # Whichever sequence contains the larger value determines whether the tempo is considered 
    # duple or triple, respectively, and the location of the largest value is treated as the 
    # faster target tempo, with one-half or one-third of that tempo, respectively, as the adjacent 
    # metrical level. - ellis 2007
    if duple > triple:
        faster_tempo_frame = np.argmax(tps_2)
        slower_tempo_frame = np.argmax(tps_2) * 2
        return faster_tempo_frame, slower_tempo_frame
    else: 
        faster_tempo_frame = np.argmax(tps_3)
        slower_tempo_frame = np.argmax(tps_3) * 3
        return faster_tempo_frame, slower_tempo_frame
  
def estimate_tempo(odf, sr, hop_sec=0.004, max_lag_s=5, show=False):
    hop_length = int(hop_sec * sr)

    # only correlate reasonable lag range 
    max_size = max_lag_s * sr / hop_length

    # auto correlate onset strength
    auto_c = librosa.autocorrelate(odf, max_size=max_size)

    # weighting function needs lag in seconds
    lags_seconds = np.arange(len(auto_c)) * (hop_length / sr)

    # apply perceptual weightinig get tempo period strengths
    tps = perceptual_weighting(lags_seconds) * auto_c
    
    # no tps resampling to tempo multiples
    # selected_tempo_frame = np.argmax(tps) 
    
    # # calculate the tempo multiples
    # # two further functions are calculated by resampling T P S to one-half and one-third, 
    # # respectively, of its original length, adding this to the original T P S, 
    # # then choosing the largest peak across both these sequences - ellis 2007
    faster_tempo_frame, slower_tempo_frame = tempo_multiples(tps)
    
    faster_tempo_peak = tps[faster_tempo_frame]
    slower_tempo_peak = tps[slower_tempo_frame]
    
    # Relative weights of the two levels are again taken from the relative peak heights 
    # at the two period estimates in the original T P S. This approach finds the tempo that maximizes
    # the sum of the T P S values at both metrical levels - ellis 2007
    faster_tempo_weight = faster_tempo_peak / (faster_tempo_peak + slower_tempo_peak)
    slower_tempo_weight = slower_tempo_peak / (faster_tempo_peak + slower_tempo_peak)
        
    # weight difference
    if (slower_tempo_weight > faster_tempo_weight): selected_tempo_frame = slower_tempo_frame
    else: selected_tempo_frame = faster_tempo_frame
        
    # plot results
    if(show): 
        print("faster_tempo_level: {}, slower_tempo_level {}".format(faster_tempo_peak, slower_tempo_peak))
        print("faster_tempo weigth: {}, slower_tempo weight {}".format(faster_tempo_weight, slower_tempo_weight))
        print('weight ratio: {}'.format(abs(faster_tempo_weight - slower_tempo_weight)))
        plot_auto_c(auto_c, tps, faster_tempo_frame, slower_tempo_frame, selected_tempo_frame, sr, hop_length)

    return selected_tempo_frame

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

# ported from ellis 2007 matlab code
def estimate_beats(ose, tempo_estimate, sr, alpha=680, hop_sec=0.004, show=False):
    # initialize backlink and cumulative score arrays
    backlink = -np.ones(len(ose), dtype=int)
    c_score = ose.copy()
    
    # define search range for previous beat based on the period
    prev_range = np.arange(-2*tempo_estimate, -round(tempo_estimate/2), dtype=int)
    
    # calculate transition cost using a log-gaussian window over the search range
    cost = -alpha * np.abs(np.log(prev_range / -tempo_estimate) ** 2)

    # set up the dynamic programming loop bounds
    loop_start = max(-prev_range[0], 0)
    loop_end = len(ose)
    
    # use the loop to fill in backlink and cumlative score
    for i in range(loop_start, loop_end):
        timerange = i + prev_range
        # ensure timerange indices are within bounds
        valid_timerange = timerange[(timerange >= 0) & (timerange < len(ose))]
        
        # calculate score candidates and find the best predecessor beat
        score_candiadates = cost[:len(valid_timerange)] + c_score[valid_timerange]
        max_score_index = np.argmax(score_candiadates)
        max_score = score_candiadates[max_score_index]
        
        # update cumulative score and backlink
        c_score[i] = max_score + ose[i]
        backlink[i] = valid_timerange[max_score_index]
    
    # start backtrace from the highest cumulative score
    beats = [np.argmax(c_score)]
        
    # backtrace to find all predecessors
    while backlink[beats[0]] > 0:
        beats.insert(0, backlink[beats[0]])
    
    hop_length = int(hop_sec * sr)
    times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
    
    if(show): plot_dynamic_programming(c_score, backlink, ose, beats)
    
    return times

def plot_estimated_vs_annotation(y, sr, beats_estimates, annotations, name, genre, xlim=[10, 20],):
    plt.figure(figsize=(10, 4))
    plt.title('Beats {} {}'.format(name, genre)) # add bpm
    plt.plot(np.linspace(0, (len(y) / sr), num=len(y)), y, alpha=0.6)  # plot waveform
    plt.vlines(beats_estimates, ymin=0, ymax=max(y), color='r', linestyle='--', label='Estimates')  # plot beats
    plt.vlines(annotations, ymin=min(y), ymax=0, color='g', linestyle='--', label='Annotations')  # plot annotations
    plt.legend()
    plt.xlim(xlim)
    
# The whole system should be called via this function
def beat_track(audio_file, name="", genre="", annotations=[], show=False):
    hop_sec=0.004
    resample_rate = 8000

    # load audio
    y, sr = librosa.load(audio_file, sr=resample_rate)
    ose = onset_strength_envelope(y, sr, hop_sec=hop_sec, show=show)
    tempo_estimate = estimate_tempo(ose, sr, hop_sec=hop_sec, show=show)
    beats_estimates = estimate_beats(ose, tempo_estimate, sr, hop_sec=hop_sec, show=show) # plot here can take over a miniute
    
    if(show and len(annotations) > 0): 
        # plot estimated vs reference 
        plot_estimated_vs_annotation(y, sr, beats_estimates, annotations, name, genre)
    
    return beats_estimates

audio_file = "sample_audio.mp3"
beats = beat_track(audio_file)
print(beats)