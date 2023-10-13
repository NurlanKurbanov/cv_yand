import numpy as np
import scipy


def gaussian_kernel(size, sigma):
    """
    Построение ядра фильтра Гаусса.

    @param  size  int    размер фильтра
    @param  sigma float  параметр размытия
    @return numpy array  фильтр Гаусса размером size x size
    """
    kernel1d = scipy.signal.windows.gaussian(size, std=sigma)
    k = kernel1d[None, :] * kernel1d[:, None]
    k /= np.sum(k)
    return k


def fourier_transform(h, shape):
    """
    Получение Фурье-образа искажающей функции

    @param  h            numpy array  искажающая функция h (ядро свертки)
    @param  shape        list         требуемый размер образа
    @return numpy array  H            Фурье-образ искажающей функции h
    """
    ph, pw = shape[0] - h.shape[0], shape[1] - h.shape[1]
    padding = [((ph + 1) // 2, ph // 2), ((pw + 1) // 2, pw // 2)]
    h = np.pad(h, padding)
    h_shifted = np.fft.ifftshift(h)
    f = np.fft.fft2(h_shifted)
    return f


def inverse_kernel(H, threshold=1e-10):
    """
    Получение H_inv

    @param  H            numpy array    Фурье-образ искажающей функции h
    @param  threshold    float          порог отсечения для избежания деления на 0
    @return numpy array  H_inv
    """
    idx = np.abs(H) <= threshold
    H[idx] = 0
    H[~idx] = 1 / H[~idx]
    return H


def inverse_filtering(blurred_img, h, threshold=1e-10):
    """
    Метод инверсной фильтрации

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  threshold      float        параметр получения H_inv
    @return numpy array                 восстановленное изображение
    """
    blurred_img_fft = fourier_transform(blurred_img, blurred_img.shape)
    h_fft = fourier_transform(h, blurred_img.shape)
    h_fft_inv = inverse_kernel(h_fft, threshold)
    x = np.fft.ifft2(blurred_img_fft * h_fft_inv)
    return np.abs(np.fft.fftshift(x))


def wiener_filtering(blurred_img, h, K=3e-05):
    """
    Винеровская фильтрация

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  K              float        константа из выражения (8)
    @return numpy array                 восстановленное изображение
    """
    h_fft = fourier_transform(h, blurred_img.shape)
    blurred_img_fft = fourier_transform(blurred_img, blurred_img.shape)
    x = (np.conj(h_fft) * blurred_img_fft) / (np.conj(h_fft) * h_fft + K)
    x = np.fft.ifft2(x)
    return np.abs(np.fft.fftshift(x))


def compute_psnr(img1, img2):
    """
    PSNR metric

    @param  img1    numpy array   оригинальное изображение
    @param  img2    numpy array   искаженное изображение
    @return float   PSNR(img1, img2)
    """
    mse = np.mean((img1 - img2)**2)
    return 20 * np.log10(255 / np.sqrt(mse))

