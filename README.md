# Beat Tracking by Dynamic Programming.

Given an audio file, the beat tracker uses a dynamic programming algorithm to compute an estimated sequence of beat times[1].

The system is made up of three distinct phases. 
- First, perceptually weighted spectral flux analysis extracts an onset strength envelope (OSE) from an audio file.

![full_ose_Albums-Commitments-10](https://github.com/ruarim/beat_tracker/assets/48099261/1175b867-fddd-496a-98ca-8bc21c24fed4)

- This OSE is then used to estimate a tempo period for the audio via a perceptually weighted auto-correlation.

![5secs_weighted_ac_Albums-Commitments-10](https://github.com/ruarim/beat_tracker/assets/48099261/19ee7057-ced0-4e12-b1ea-a32f45d8c5db)

- Finally, a dynamic programming algorithm is used to identify the most likely sequence of beats based on the OSE and estimated tempo period.

![10sec_to_20sec_estimated_beats_Albums-Commitments-10](https://github.com/ruarim/beat_tracker/assets/48099261/8e981ff8-f815-412e-96af-08de17477870)

![dynamic_programming_visualisation](https://github.com/ruarim/beat_tracker/assets/48099261/f91ba64e-e6b1-4a13-9117-f9ab27885aeb)

## The system was evaluated using a mean f-measure score across ~700 Ballroom dance music recordings.
![f_measure_with_trim_resample_tps](https://github.com/ruarim/beat_tracker/assets/48099261/e26aa6b9-cd10-4f74-ad37-4a85686557f5)

- Find the audio dataset [here](http://mtg.upf.edu/ismir2004/contest/tempoContest/data1.tar.gz).
- Find the associated beat annotations [here](https://github.com/CPJKU/BallroomAnnotations).

Due to Dynamic Programming, this algorithm is particularly efficient and can process and evaluate performance on the full Ballroom dance dataset in ~2-3 minutes.

[1] D. P. W. Ellis, ‘Beat Tracking by Dynamic Programming’, Journal of New Music Research, vol. 36, no. 1, pp. 51–60, Mar. 2007.
