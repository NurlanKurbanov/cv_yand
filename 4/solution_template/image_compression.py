import os
import numpy as npcd
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.ndimage import gaussian_filter
from skimage.metrics import peak_signal_noise_ratio


def pca_compression(matrix, p):
    """ Сжатие изображения с помощью PCA
    Вход: двумерная матрица (одна цветовая компонента картинки), количество компонент
    Выход: собственные векторы, проекция матрицы на новое пр-во и средние значения до центрирования
    """
    # Your code here
    m = np.copy(matrix).astype('float64')
    # Отцентруем каждую строчку матрицы
    row_mean = np.mean(matrix, axis=1)
    m -= row_mean[:, None]
    # Найдем матрицу ковариации
    cov = np.cov(m)
    # Ищем собственные значения и собственные векторы матрицы ковариации, используйте linalg.eigh из numpy
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Посчитаем количество найденных собственных векторов
    #...
    # Сортируем собственные значения в порядке убывания
    max_idx = np.argsort(-eigenvalues)
    # Сортируем собственные векторы согласно отсортированным собственным значениям
    # !Это все для того, чтобы мы производили проекцию в направлении максимальной дисперсии!

    max_disp_vect = eigenvectors.T[max_idx]
    # Оставляем только p собственных векторов
    max_disp_vect = max_disp_vect[:p]
    max_disp_vect = max_disp_vect.T

    # Проекция данных на новое пространство
    proj = np.dot(max_disp_vect.T, m)

    return max_disp_vect, proj, row_mean


def pca_decompression(compressed):
    """ Разжатие изображения
    Вход: список кортежей из собственных векторов и проекций для каждой цветовой компоненты
    Выход: разжатое изображение
    """
    result_img = []
    for i, comp in enumerate(compressed):
        # Матрично умножаем собственные векторы на проекции и прибавляем среднее значение по строкам исходной матрицы
        # !Это следует из описанного в самом начале примера!

        # Your code here
        result_img.append(np.dot(comp[0], comp[1]) + comp[2][:, None])
    result_img = np.dstack(result_img)
    return np.clip(result_img, 0, 255).astype('uint8')


def pca_visualize():
    plt.clf()
    img = imread('cat.jpg')
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(3, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 5, 10, 20, 50, 100, 150, 200, 256]):
        compressed = []
        for j in range(0, 3):
            comp = pca_compression(img[:, :, j], p)
            compressed.append(comp)
        decomp = pca_decompression(compressed)

        axes[i // 3, i % 3].imshow(decomp)
        axes[i // 3, i % 3].set_title('Компонент: {}'.format(p))

    fig.savefig("pca_visualization.png")


def rgb2ycbcr(img):
    """ Переход из пр-ва RGB в пр-во YCbCr
    Вход: RGB изображение
    Выход: YCbCr изображение
    """
    # Your code here
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    Y = 0.299 * r + 0.587 * g + 0.114 * b
    C_b = -0.1687 * r - 0.3313 * g + 0.5 * b + 128
    C_r = 0.5 * r - 0.4187 * b - 0.0813 * b + 128
    return np.clip(np.dstack([Y, C_b, C_r]), 0, 255).astype('uint8')


def ycbcr2rgb(img):
    img = img.astype('float')
    img[:, :, 1:] -= 128.0

    m = np.array([[1.0, 0.0, 1.402],
                  [1.0, -0.344136, -0.714136],
                  [1.0, 1.772, 0.0]])

    rgb = np.dot(img, m.T)
    return np.clip(rgb, 0, 255).astype('uint8')


def get_gauss_1():
    plt.clf()
    rgb_img = imread('Lenna.png')
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]

    # Your code here
    lena_ycbcr = rgb2ycbcr(rgb_img)
    lena_ycbcr[:, :, 1] = gaussian_filter(lena_ycbcr[:, :, 1], 3)
    lena_ycbcr[:, :, 2] = gaussian_filter(lena_ycbcr[:, :, 2], 3)

    lena_rgb = ycbcr2rgb(lena_ycbcr)
    plt.imshow(lena_rgb)
    plt.savefig("gauss_1.png")


def get_gauss_2():
    plt.clf()
    rgb_img = imread('Lenna.png')
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]

    # Your code here
    lena_ycbcr = rgb2ycbcr(rgb_img)
    lena_ycbcr[:, :, 0] = gaussian_filter(lena_ycbcr[:, :, 0], 3)

    lena_rgb = ycbcr2rgb(lena_ycbcr)
    plt.imshow(lena_rgb)
    plt.savefig("gauss_2.png")


def downsampling(component):
    """Уменьшаем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B, 1]
    Выход: цветовая компонента размера [A // 2, B // 2, 1]
    """

    # Your code here
    return gaussian_filter(component, 10)[::2, ::2]


def dct(block):
    """Дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после ДКП
    """
    # Your code here
    res = np.zeros_like(block, dtype='float64')
    idx = np.array([2*x + 1 for x in range(block.shape[0])])

    sum_ = lambda  u,v : np.sum(block * np.outer(np.cos((idx * u * np.pi)/16), np.cos((idx * v * np.pi)/16)))

    res[0, 0] = 0.125 * np.sum(block)
    for i in range(1, block.shape[0]):
        res[i, 0] = sum_(i, 0) / (4 * np.sqrt(2))
    for j in range(1, block.shape[1]):
        res[0, j] = sum_(0, j) / (4 * np.sqrt(2))
    for i in range(1, block.shape[0]):
        for j in range(1, block.shape[1]):
            res[i, j] = 0.25 * sum_(i, j)

    return res


# Матрица квантования яркости
y_quantization_matrix = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

# Матрица квантования цвета
color_quantization_matrix = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])


def quantization(block, quantization_matrix):
    """Квантование
    Вход: блок размера 8x8 после применения ДКП; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление осуществляем с помощью np.round
    """
    # Your code here
    return np.round(block / quantization_matrix)


def own_quantization_matrix(default_quantization_matrix, q):
    """Генерация матрицы квантования по Quality Factor
    Вход: "стандартная" матрица квантования; Quality Factor
    Выход: новая матрица квантования
    Hint: если после проделанных операций какие-то элементы обнулились, то замените их единицами
    """
    assert 1 <= q <= 100
    # Your code here

    s = 1 if q == 100 else (5000/q if q < 50 else 200 - 2*q)

    new_q_matr = np.floor((50 + s * default_quantization_matrix) / 100)
    new_q_matr[new_q_matr == 0] = 1
    return new_q_matr


def zigzag(block):
    """Зигзаг-сканирование
    Вход: блок размера 8x8
    Выход: список из элементов входного блока, получаемый после его обхода зигзаг-сканированием
    """
    # Your code here
    idx = np.array([0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12,
                    19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35,
                    42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58,
                    59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63])
    return block.flatten()[idx]


def compression(zigzag_list):
    """Сжатие последовательности после зигзаг-сканирования
    Вход: список после зигзаг-сканирования
    Выход: сжатый список в формате, который был приведен в качестве примера в самом начале данного пункта
    """
    # Your code here
    res = []

    k = 0
    len_ = len(zigzag_list)
    for i in range(len_):
        if (i == len_ - 1) and (zigzag_list[i] == 0):
            if k == 0:
                res.append(zigzag_list[i])
                res.append(1)
            else:
                res.append(k + 1)
        elif zigzag_list[i] == 0:
            if k == 0:
                res.append(zigzag_list[i])
            k += 1
        else:
            if k > 0:
                res.append(k)
            k = 0
            res.append(zigzag_list[i])

    return res


def jpeg_compression(img, quantization_matrixes):
    """JPEG-сжатие
    Вход: цветная картинка, список из 2-ух матриц квантования
    Выход: список списков со сжатыми векторами: [[compressed_y1,...], [compressed_Cb1,...], [compressed_Cr1,...]]
    """
    # Your code here
    res = [[], [], []]
    # Переходим из RGB в YCbCr
    img_ycbcr = rgb2ycbcr(img)
    # Уменьшаем цветовые компоненты
    y, cb, cr = img_ycbcr[:, :, 0], downsampling(img_ycbcr[:, :, 1]), downsampling(img_ycbcr[:, :, 2])
    # Делим все компоненты на блоки 8x8 и все элементы блоков переводим из [0, 255] в [-128, 127]
    for ch_num, ch in enumerate((y, cb, cr)):
        for i in range(0, ch.shape[0] - 7, 8):
            for j in range(0, ch.shape[1] - 7, 8):
                block = ch[i:i+8, j:j+8].astype('float64')
                block -= 128
    # Применяем ДКП, квантование, зизгаз-сканирование и сжатие
                block_ = compression(zigzag(quantization(dct(block), quantization_matrixes[0 if ch_num == 0 else 1])))
                res[ch_num].append(block_)
    return res


def inverse_compression(compressed_list):
    """Разжатие последовательности
    Вход: сжатый список
    Выход: разжатый список
    """
    # Your code here
    res = []
    len_ = len(compressed_list)

    i = 0
    while i < len_:
        if compressed_list[i] == 0:
            res.extend([0] * compressed_list[i+1])
            i += 2
        else:
            res.append(compressed_list[i])
            i += 1
    return res


def inverse_zigzag(input):
    """Обратное зигзаг-сканирование
    Вход: список элементов
    Выход: блок размера 8x8 из элементов входного списка, расставленных в матрице в порядке их следования в зигзаг-сканировании
    """
    # Your code here
    idx = np.array([0, 1, 5, 6, 14, 15, 27, 28,
                    2, 4, 7, 13, 16, 26, 29, 42,
                    3, 8, 12, 17, 25, 30, 41, 43,
                    9, 11, 18, 24, 31, 40, 44, 53,
                    10, 19, 23, 32, 39, 45, 52, 54,
                    20, 22, 33, 38, 46, 51, 55, 60,
                    21, 34, 37, 47, 50, 56, 59, 61,
                    35, 36, 48, 49, 57, 58, 62, 63
                    ])
    return np.array(input)[idx].reshape(8, 8)


def inverse_quantization(block, quantization_matrix):
    """Обратное квантование
    Вход: блок размера 8x8 после применения обратного зигзаг-сканирования; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление не производится
    """
    # Your code here
    return block * quantization_matrix


def inverse_dct(block):
    """Обратное дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после обратного ДКП. Округление осуществляем с помощью np.round
    """
    # Your code here
    res = np.zeros_like(block, dtype='float64')
    idx = np.arange(0,8)

    alphas = [1] * 8
    alphas[0] = 1/np.sqrt(2)
    alphas = np.array(alphas)

    sum_ = lambda x,y: np.sum(block * np.outer(alphas,alphas)
                              * np.outer(np.cos((2*x+1) * idx * np.pi / 16),
                                         np.cos((2*y+1) * idx * np.pi / 16)))
    for i in range(8):
        for j in range(8):
            res[i, j] = 0.25 * sum_(i, j)
    return np.round(res)


def upsampling(component):
    """Увеличиваем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B, 1]
    Выход: цветовая компонента размера [2 * A, 2 * B, 1]
    """
    # Your code here
    #arr = component[:, :, 0]
    component = np.repeat(component, 2, axis=1)
    arr = np.repeat(component, 2, axis=0)
    return arr


def jpeg_decompression(result, result_shape, quantization_matrixes):
    """Разжатие изображения
    Вход: result список сжатых данных, размер ответа, список из 2-ух матриц квантования
    Выход: разжатое изображение
    """
    # Your code here
    blocks = [[], [], []]
    for ch_num, compressed_vectors_list in enumerate(result):
        for compressed_vector in compressed_vectors_list:
            q_matr_num = 0 if ch_num == 0 else 1
            block = inverse_zigzag(inverse_compression(compressed_vector))
            block = inverse_dct(inverse_quantization(block, quantization_matrixes[q_matr_num]))
            blocks[ch_num].append(block)

    down_comp_size = [(result_shape[0], result_shape[1]), (result_shape[0]//2, result_shape[1]//2), (result_shape[0]//2, result_shape[1]//2)]
    down_components = [np.zeros(down_comp_size[i]) for i in range(3)]
    for ch_num in range(3):
        block_num = 0
        for i in range(0, down_comp_size[ch_num][0] - 7, 8):
            for j in range(0, down_comp_size[ch_num][1] - 7, 8):
                down_components[ch_num][i:i+8, j:j+8] = blocks[ch_num][block_num] + 128
                block_num += 1

    components = [down_components[0], upsampling(down_components[1]), upsampling(down_components[2])]
    rgb_img = ycbcr2rgb(np.dstack(components))
    return np.clip(rgb_img, 0, 255).astype('uint8')


def jpeg_visualize():
    plt.clf()
    img = imread('Lenna.png')
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(2, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 10, 20, 50, 80, 100]):
        # Your code here
        q_mat_br = own_quantization_matrix(y_quantization_matrix, p)
        q_matr_color = own_quantization_matrix(color_quantization_matrix, p)

        comp_img = jpeg_compression(img, (q_mat_br, q_matr_color))
        decomp_img = jpeg_decompression(comp_img, img.shape, (q_mat_br, q_matr_color))

        axes[i // 3, i % 3].imshow(decomp_img)
        axes[i // 3, i % 3].set_title('Quality Factor: {}'.format(p))

    fig.savefig("jpeg_visualization.png")


def compression_pipeline(img, c_type, param=1):
    """Pipeline для PCA и JPEG
    Вход: исходное изображение; название метода - 'pca', 'jpeg';
    param - кол-во компонент в случае PCA, и Quality Factor для JPEG
    Выход: изображение; количество бит на пиксель
    """

    assert c_type.lower() == 'jpeg' or c_type.lower() == 'pca'

    if c_type.lower() == 'jpeg':
        y_quantization = own_quantization_matrix(y_quantization_matrix, param)
        color_quantization = own_quantization_matrix(color_quantization_matrix, param)
        matrixes = [y_quantization, color_quantization]

        compressed = jpeg_compression(img, matrixes)
        img = jpeg_decompression(compressed, img.shape, matrixes)
    elif c_type.lower() == 'pca':
        compressed = []
        for j in range(0, 3):
            compressed.append((pca_compression(img[:, :, j].astype(np.float64).copy(), param)))

        img = pca_decompression(compressed)
        compressed.extend([np.mean(img[:, :, 0], axis=1), np.mean(img[:, :, 1], axis=1), np.mean(img[:, :, 2], axis=1)])

    if 'tmp' not in os.listdir() or not os.path.isdir('tmp'):
        os.mkdir('tmp')

    compressed = np.array(compressed, dtype=np.object_)
    np.savez_compressed(os.path.join('tmp', 'tmp.npz'), compressed)
    size = os.stat(os.path.join('tmp', 'tmp.npz')).st_size * 8
    os.remove(os.path.join('tmp', 'tmp.npz'))

    return img, size / (img.shape[0] * img.shape[1])


def calc_metrics(img_path, c_type, param_list):
    """Подсчет PSNR и Rate-Distortion для PCA и JPEG. Построение графиков
    Вход: пусть до изображения; тип сжатия; список параметров: кол-во компонент в случае PCA, и Quality Factor для JPEG
    """

    assert c_type.lower() == 'jpeg' or c_type.lower() == 'pca'

    img = imread(img_path)
    if len(img.shape) == 3:
        img = img[..., :3]

    outputs = []
    for param in param_list:
        outputs.append(compression_pipeline(img.copy(), c_type, param))

    psnr = [peak_signal_noise_ratio(img, output[0]) for output in outputs]
    rate = [output[1] for output in outputs]

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    fig.set_figwidth(20)
    fig.set_figheight(5)

    ax1.set_title('PSNR for {}'.format(c_type.upper()))
    ax1.plot(param_list, psnr, 'tab:orange')
    ax1.set_xlabel('Quality Factor')
    ax1.set_ylabel('PSNR')

    ax2.set_title('Rate-Distortion for {}'.format(c_type.upper()))
    ax2.plot(psnr, rate, 'tab:red')
    ax2.set_xlabel('Distortion')
    ax2.set_ylabel('Rate')
    return fig


def get_pca_metrics_graph():
    plt.clf()
    fig = calc_metrics('Lenna.png', 'pca', [1, 5, 10, 20, 50, 100, 150, 200, 256])
    fig.savefig("pca_metrics_graph.png")


def get_jpeg_metrics_graph():
    plt.clf()
    fig = calc_metrics('Lenna.png', 'jpeg', [1, 10, 20, 50, 80, 100])
    fig.savefig("jpeg_metrics_graph.png")


if __name__ == "__main__":
    pca_visualize()
    get_gauss_1()
    get_gauss_2()
    jpeg_visualize()
    get_pca_metrics_graph()
    get_jpeg_metrics_graph()
