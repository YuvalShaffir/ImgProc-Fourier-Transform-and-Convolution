import numpy as np
import matplotlib.pyplot as plt
import imageio
import scipy
from scipy import signal
from scipy.ndimage.interpolation import map_coordinates
from skimage.color import rgb2gray
GRAYSCALE = 1
RGB = 2


def read_image(filename, representation):
    """
    Reads an image and converts it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    :return: Returns the image as an np.float64 matrix normalized to [0,1]
    """
    # read the image - check if it is greyscale image:
    img = imageio.imread(filename)

    # greyscale representation
    if representation == GRAYSCALE:
        img_g = rgb2gray(img)
        img_g = img_g.astype('float64')
        return img_g

    # RGB representation
    if representation == RGB:
        img_rgb = img.astype('float64')
        img_rgb_norm = img_rgb / 255
        return img_rgb_norm


def DFT(signal):
    """
    calculates the discrete fourier transform of the signal of shape (N,1)
    :param signal: shape (N,1)
    :return: dft of signal.
    """
    N = signal.shape[0]
    x = np.arange(N)
    u = x.reshape((N, 1))
    pi = np.pi
    exponent = np.exp((-2 * 1j * pi * u * x)/N)
    return np.around(np.dot(exponent, signal), decimals=9)


def IDFT(inv_signal):
    """
    Calculates the inverted discrete fourier transform.
    :param inv_signal: a dft signal.
    :return: the original signal.
    """
    N = inv_signal.shape[0]
    x = np.arange(N)
    u = x.reshape((N, 1))
    pi = np.pi
    exponent = np.exp((2 * 1j * pi * u * x)/ N)
    return (1/N) * np.dot(exponent, inv_signal)


def DFT2(image):
    """
    Calculates the DFT of a 2D image.
    :param image: grayscale image of type float64 of shape (M,N) or (M,N,1)
    :return: fourier transform of the image.
    """
    (M, N, *K) = image.shape
    dft_result = np.zeros((M, N), dtype=np.complex128)
    for y in range(M):
        dft_result[y] = DFT(image[y].reshape((N,)))
    for x in range(N):
        dft_result.T[x] = DFT(dft_result.T[x].reshape((M,)))
    return dft_result.reshape((M, N, *K))


def IDFT2(fourier_image):
    """
    Calculates the IDFT of a 2D fourier transformed array of an image.
    :param fourier_image: type complex128 of shape (M,N) or (M,N,1)
    :return: original image.
    """

    (M, N, *K) = fourier_image.shape
    result = np.zeros((M, N), dtype=np.complex128)
    for y in range(M):
        result[y] = IDFT(fourier_image[y].reshape((N,)))
    for x in range(N):
        result.T[x] = IDFT(result.T[x].reshape((M,)))
    return result.reshape((M, N, *K))


def change_rate(filename, ratio):
    """
    The function changes the duration of an audio file by keeping the same samples, but changing the
    sample rate written in the file header. this function saves the audio in a new file called change_rate.wav .
    :param filename: a string representing the path to a WAV file.
    :param ratio: a positive float64 representing the duration change (0.25 < ratio < 4).
    """
    rate, data = scipy.io.wavfile.read(filename)
    new_rate = int(rate * ratio)
    scipy.io.wavfile.write("change_rate.wav", new_rate, data)


def add_zero_padding(signal, sample_dif):
    if sample_dif % 2 == 0:
        zeros_start = int(np.floor(sample_dif/2))
        zeros_end = int(np.floor(sample_dif/2))
    else:
        zeros_start = int(np.floor(sample_dif//2))
        zeros_end = int(np.floor(sample_dif - zeros_start))
    return np.pad(signal, (zeros_start, zeros_end), 'constant', constant_values=(0))


def resize(data, ratio):
    """
    The function changes the number of samples by a given ratio.
    :param data: a 1D ndarray of dtype float64 or complex128(*).
    :param ratio:
    :return:
    """
    dft_data = DFT(data)
    samples = int(np.floor(dft_data.size))
    new_samples = int(np.floor(samples / ratio))
    if ratio < 1:
        sample_dif = new_samples - samples
        dft_data = add_zero_padding(dft_data, sample_dif)
    else:
        data_fft_shift = np.fft.fftshift(dft_data)
        dft_data = data_fft_shift[0: new_samples]
    new_data = np.real(IDFT(dft_data))
    return new_data.astype(np.int16)


def change_samples(filename, ratio):
    """
    A fast-forward function that changes the duration of an audio file by reducing the number of samples
    using Fourier. This function does not change the sample rate of the given file.
    The result are saved in a file called change_samples.wav.
    :param filename:  a string representing the path to a WAV file
    :param ratio: a positive float64 representing the duration change.
    :return:
    """
    rate, data = scipy.io.wavfile.read(filename)
    if ratio == 1:
        scipy.io.wavfile.write("change_samples.wav", rate, data)
    else:
        new_data = resize(data, ratio)
        scipy.io.wavfile.write("change_samples.wav", rate, new_data)


"Question 1: I can hear that the result of change_samples has lower frequencies than the result of change_rate."
# change_rate("aria_4kHz.wav",2)
# change_samples("aria_4kHz.wav", 2)


def resize_spectrogram(data, ratio):
    """
    The function speeds up a WAV file, without changing the pitch, using spectrogram scaling.
    :param data: a 1D ndarray of dtype float64 representing the original sample points.
    :param ratio: a positive float64 representing the rate change of the WAV file.
    :return:
    """
    if ratio == 1:
        return data
    spectogram_mat = stft(data)
    specto_rows = spectogram_mat.shape[0]
    lst = []
    for row in range(specto_rows):
        lst.append(resize(spectogram_mat[row], ratio))
    new_spectogram = np.asarray(lst, dtype=np.int16)
    new_data = istft(new_spectogram).astype(np.int16)
    return new_data


def resize_vocoder(data, ratio):
    """
    The function speedups a WAV file by phase vocoding its spectrogram.
    :param data: a 1D ndarray of dtype float64 representing the original sample points.
    :param ratio: a positive float64 representing the rate change of the WAV file.
    :return:
    """
    if ratio == 1:
        return data
    spectogram_mat = stft(data)
    warped_spec = phase_vocoder(spectogram_mat, ratio)
    new_data = istft(warped_spec).astype(np.int16)
    return new_data


def conv_der(im):
    """
    The function computes the magnitude of image derivatives.
    :param im: grayscale images of type float64
    :return: grayscale magnitude image of type float64
    """
    dx_vec = np.array([[0.5, 0, -0.5]])
    dy_vec = dx_vec.T
    dx_im = scipy.signal.convolve2d(im, dx_vec, mode='same')
    dy_im = scipy.signal.convolve2d(im, dy_vec, mode='same')
    magnitude = np.sqrt(np.abs(dx_im) ** 2 + np.abs(dy_im) ** 2)
    return magnitude


def fourier_der(im):
    """
    The function computes the magnitude of the image derivatives using Fourier transform.
    :param im: grayscale images of type float64
    :return: grayscale magnitude image of type float64
    """
    fourier_im = np.fft.fftshift(DFT2(im))
    X = fourier_im.shape[0]
    Y = fourier_im.shape[1]
    U_arr = np.arange(X)
    V_arr = np.arange(Y)
    U_diag = np.diag(U_arr)
    V_diag = np.diag(V_arr)
    du_fourier_im = np.dot(U_diag, fourier_im)
    dv_fourier_im = np.dot(fourier_im, V_diag)
    dx_im = IDFT2(du_fourier_im)
    dy_im = IDFT2(dv_fourier_im)
    magnitude = np.sqrt(np.abs(dx_im) ** 2 + np.abs(dy_im) ** 2)
    return magnitude


def stft(y, win_length=640, hop_length=160):
    """
    Preforms the short time fourier transform
    """
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    """
    Preforms the inverse short time fourier transform
    """
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    """
    Scales the spectorgram's "spec" by "ratio"
    """
    num_timesteps = int(spec.shape[1] / ratio)
    time_steps = np.arange(num_timesteps) * ratio

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec
