The codes named with _FFT and _FFT3 are modified from the original codes of the team.
The codes excluding the above are copyed & pasted.

There are two types: (1) FFT and (2) FFT3

(1) FFT conducts only 2D spatial Fourier transform using FFT algorithm.

(2) FFT3 conducts 3 cases of 2D Fourier transforms on xy(space-space), xt(space-time), and yt(space-time).

After that, the input is a tuple of two(FFT_xy_amplitude, FFT_xy_phase) for (1) and is a tuple of six(FFT_xy_amplitude, FFT_xy_phase, FFT_xt_amplitude, FFT_xt_phase, FFT_yt_amplitude, FFT_yt_phase) for (2).

The procedure of running the codes is explained below.

1. run kth_download.py

2. run KTH_FFT.py for (1) or KTH_FFT3.py for (2)

3. run run_FFT.py for (1) or run_FFT3.py for (2)


※ If you want to change the proportion of cutting FFT, you can change the 'cut_param' in the code named 'model_FFT.py' or 'model_FFT3.py' (cut_param = 1 means you do not discard any part of the Fourier transformed data)
