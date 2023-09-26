import numpy as np

def repeat(row1, row2, repeats):
    row_1_2 = np.stack([row1, row2], axis=0)
    matrix = np.tile(row_1_2, (repeats // 2, 1))
    if repeats % 2:
        matrix = np.vstack([matrix, row1])

    return matrix

def get_bayer_masks(n_rows, n_cols):
    red_1 = np.zeros(n_cols, dtype='bool')
    red_1[1::2] = 1
    red = repeat(red_1, np.zeros(n_cols, dtype='bool'), n_rows)

    green_1 = np.zeros(n_cols, dtype='bool')
    green_1[::2] = 1
    green = repeat(green_1, red_1, n_rows)

    blue = repeat(np.zeros(n_cols, dtype='bool'), green_1, n_rows)

    return np.dstack([red, green, blue])


def get_colored_img(raw_img):
    d = get_bayer_masks(*raw_img.shape)
    res = raw_img[:, :, None] * d
    return res


def check(colored_img):
    res = np.copy(colored_img)
    masks = get_bayer_masks(*colored_img.shape[:2])
    masks = masks.astype('int')
    for channel in range(colored_img.shape[2]):
        for i in range(1, colored_img.shape[0] - 1):
            for j in range(1, colored_img.shape[1] - 1):
                if masks[i, j, channel] == 0:
                    res[i, j, channel] = (np.sum(colored_img[i - 1: i + 2, j - 1: j + 2, channel])
                                          / np.sum(masks[i - 1: i + 2, j - 1: j + 2, channel]))
    return res

def bilinear_interpolation(colored_img):
    colored_img = colored_img.astype('int')
    res = np.copy(colored_img).astype('int')
    #g
    for i in range(1, colored_img.shape[0] - 1, 2):
        for j in range(2, colored_img.shape[1] - 1, 2):
            res[i, j, 1] = 0.25*(colored_img[i, j-1, 1]
                                 + colored_img[i, j+1, 1]
                                 + colored_img[i-1, j, 1]
                                 + colored_img[i+1, j, 1])
    for i in range(2, colored_img.shape[0] - 1, 2):
        for j in range(1, colored_img.shape[1] - 1, 2):
            res[i, j, 1] = 0.25*(colored_img[i, j-1, 1]
                                 + colored_img[i, j+1, 1]
                                 + colored_img[i-1, j, 1]
                                 + colored_img[i+1, j, 1])
    #r
    for i in range(2, colored_img.shape[0] - 1, 2):
        for j in range(2, colored_img.shape[1] - 1, 2):
            res[i, j, 0] = 0.5*(colored_img[i, j-1, 0] + colored_img[i, j+1, 0])
    for i in range(1, colored_img.shape[0] - 1, 2):
        for j in range(1, colored_img.shape[1] - 1, 2):
            res[i, j, 0] = 0.5 * (colored_img[i-1, j, 0] + colored_img[i+1, j, 0])
    for i in range(1, colored_img.shape[0] - 1, 2):
        for j in range(2, colored_img.shape[1] - 1, 2):
            res[i, j, 0] = 0.25*(colored_img[i-1, j-1, 0]
                                 + colored_img[i+1, j+1, 0]
                                 + colored_img[i-1, j+1, 0]
                                 + colored_img[i+1, j-1, 0])
    #b
    for i in range(1, colored_img.shape[0] - 1, 2):
        for j in range(1, colored_img.shape[1] - 1, 2):
            res[i, j, 2] = 0.5*(colored_img[i, j-1, 2] + colored_img[i, j+1, 2])
    for i in range(2, colored_img.shape[0] - 1, 2):
        for j in range(2, colored_img.shape[1] - 1, 2):
            res[i, j, 2] = 0.5 * (colored_img[i-1, j, 2] + colored_img[i+1, j, 2])
    for i in range(2, colored_img.shape[0] - 1, 2):
        for j in range(1, colored_img.shape[1] - 1, 2):
            res[i, j, 2] = 0.25*(colored_img[i-1, j-1, 2]
                                 + colored_img[i+1, j+1, 2]
                                 + colored_img[i-1, j+1, 2]
                                 + colored_img[i+1, j-1, 2])
    return res


def improved_interpolation(raw_img):
    colored_img = get_colored_img(raw_img)
    colored_img = colored_img.astype('float32')
    res = np.copy(colored_img).astype('float32')

    #g at r
    for i in range(2, colored_img.shape[0] - 2, 2):
        for j in range(3, colored_img.shape[1] - 2, 2):
            x = (2 * colored_img[i - 1, j, 1] +
                            2 * colored_img[i + 1, j, 1] +
                            2 * colored_img[i, j - 1, 1] +
                            2 * colored_img[i, j + 1, 1] -
                            colored_img[i - 2, j, 0] -
                            colored_img[i + 2, j, 0] -
                            colored_img[i, j - 2, 0] -
                            colored_img[i, j + 2, 0] +
                            4 * colored_img[i, j, 0]) / 8
            res[i, j, 1] = (2 * colored_img[i - 1, j, 1] +
                            2 * colored_img[i + 1, j, 1] +
                            2 * colored_img[i, j - 1, 1] +
                            2 * colored_img[i, j + 1, 1] -
                            colored_img[i - 2, j, 0] -
                            colored_img[i + 2, j, 0] -
                            colored_img[i, j - 2, 0] -
                            colored_img[i, j + 2, 0] +
                            4 * colored_img[i, j, 0]) / 8

    # g at b
    for i in range(3, colored_img.shape[0] - 2, 2):
        for j in range(2, colored_img.shape[1] - 2, 2):
            res[i, j, 1] = (2 * colored_img[i - 1, j, 1] +
                            2 * colored_img[i + 1, j, 1] +
                            2 * colored_img[i, j - 1, 1] +
                            2 * colored_img[i, j + 1, 1] -
                            colored_img[i - 2, j, 2] -
                            colored_img[i + 2, j, 2] -
                            colored_img[i, j - 2, 2] -
                            colored_img[i, j + 2, 2] +
                            4 * colored_img[i, j, 2]) / 8

    #r at g(r row)
    for i in range(2, colored_img.shape[0] - 2, 2):
        for j in range(2, colored_img.shape[1] - 2, 2):
            res[i, j, 0] = (4 * colored_img[i, j - 1, 0] +
                            4 * colored_img[i, j + 1, 0] +
                            5 * colored_img[i, j, 1] -
                            colored_img[i - 1, j - 1, 1] -
                            colored_img[i - 1, j + 1, 1] -
                            colored_img[i + 1, j - 1, 1] -
                            colored_img[i + 1, j + 1, 1] -
                            colored_img[i, j - 2, 1] -
                            colored_img[i, j + 2, 1] +
                            0.5 * colored_img[i - 2, j, 1] +
                            0.5 * colored_img[i + 2, j, 1]) / 8
    #r ar g(r column)
    for i in range(3, colored_img.shape[0] - 2, 2):
        for j in range(3, colored_img.shape[1] - 2, 2):
            res[i, j, 0] = (4 * colored_img[i - 1, j, 0] +
                            4 * colored_img[i + 1, j, 0] +
                            5 * colored_img[i, j, 1] -
                            colored_img[i - 1, j - 1, 1] -
                            colored_img[i - 1, j + 1, 1] -
                            colored_img[i + 1, j - 1, 1] -
                            colored_img[i + 1, j + 1, 1] -
                            colored_img[i - 2, j, 1] -
                            colored_img[i + 2, j, 1] +
                            0.5 * colored_img[i, j - 2, 1] +
                            0.5 * colored_img[i, j + 2, 1]) / 8
    #r at b
    for i in range(3, colored_img.shape[0] - 2, 2):
        for j in range(2, colored_img.shape[1] - 2, 2):
            res[i, j, 0] = (2 * colored_img[i - 1, j - 1, 0] +
                            2 * colored_img[i - 1, j + 1, 0] +
                            2 * colored_img[i + 1, j - 1, 0] +
                            2 * colored_img[i + 1, j + 1, 0] +
                            6 * colored_img[i, j, 2] -
                            1.5 * colored_img[i - 2, j, 2] -
                            1.5 * colored_img[i + 2, j, 2] -
                            1.5 * colored_img[i, j - 2, 2] -
                            1.5 * colored_img[i, j + 2, 2]) / 8
    #b at g(b row)
    for i in range(3, colored_img.shape[0] - 2, 2):
        for j in range(3, colored_img.shape[1] - 2, 2):
            res[i, j, 2] = (4 * colored_img[i, j - 1, 2] +
                            4 * colored_img[i, j + 1, 2] +
                            5 * colored_img[i, j, 1] -
                            colored_img[i - 1, j - 1, 1] -
                            colored_img[i - 1, j + 1, 1] -
                            colored_img[i + 1, j - 1, 1] -
                            colored_img[i + 1, j + 1, 1] -
                            colored_img[i, j - 2, 1] -
                            colored_img[i, j + 2, 1] +
                            0.5 * colored_img[i - 2, j, 1] +
                            0.5 * colored_img[i + 2, j, 1]) / 8
    #b at g(b col)
    for i in range(2, colored_img.shape[0] - 2, 2):
        for j in range(2, colored_img.shape[1] - 2, 2):
            res[i, j, 2] = (4 * colored_img[i - 1, j, 2] +
                            4 * colored_img[i + 1, j, 2] +
                            5 * colored_img[i, j, 1] -
                            colored_img[i - 1, j - 1, 1] -
                            colored_img[i - 1, j + 1, 1] -
                            colored_img[i + 1, j - 1, 1] -
                            colored_img[i + 1, j + 1, 1] -
                            colored_img[i - 2, j, 1] -
                            colored_img[i + 2, j, 1] +
                            0.5 * colored_img[i, j - 2, 1] +
                            0.5 * colored_img[i, j + 2, 1]) / 8
    #b at red
    for i in range(2, colored_img.shape[0] - 2, 2):
        for j in range(3, colored_img.shape[1] - 2, 2):
            res[i, j, 2] = (2 * colored_img[i - 1, j - 1, 2] +
                            2 * colored_img[i - 1, j + 1, 2] +
                            2 * colored_img[i + 1, j - 1, 2] +
                            2 * colored_img[i + 1, j + 1, 2] +
                            6 * colored_img[i, j, 0] -
                            1.5 * colored_img[i - 2, j, 0] -
                            1.5 * colored_img[i + 2, j, 0] -
                            1.5 * colored_img[i, j - 2, 0] -
                            1.5 * colored_img[i, j + 2, 0]) / 8

    res = np.clip(res, 0, 255)
    return res.astype('int')


def compute_psnr(img_pred, img_gt):
    img_pred = img_pred.astype('float64')
    img_gt = img_gt.astype('float64')

    mse = np.sum((img_pred - img_gt)**2) / img_pred.size

    if mse == 0:
        raise ValueError

    psnr = 10 * np.log10((img_gt.max()**2) / mse)
    return psnr
