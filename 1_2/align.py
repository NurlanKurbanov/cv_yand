import numpy as np


def calc_shift(x, y):
    c = np.fft.ifft2(np.fft.fft2(x) * np.conj(np.fft.fft2(y)))
    flat_argmax = np.argmax(c)
    # Convert the flattened index to a tuple of (row, column) indices
    max_index_2d = np.unravel_index(flat_argmax, c.shape)
    return max_index_2d


def find_match_point(row, col, row_shift, col_shift, img_shape):
    row_res = -((row + row_shift) % img_shape[0] - row)
    col_res = -((col + col_shift) % img_shape[1] - col)
    return row_res, col_res


def make_shift(img, row_shift, col_shift):
    img = np.roll(img, row_shift, axis=0)
    img = np.roll(img, col_shift, axis=1)
    return img


def align(img, g_coord):
    g_row, g_col = g_coord[0], g_coord[1]
    h = img.shape[0] // 3
    b, g, r = img[:h, :], img[h:2*h, :], img[2*h:3*h, :]

    row_crop, col_crop = int(b.shape[0] * 0.1), int(b.shape[1] * 0.1)
    b, g, r = map(lambda x: x[row_crop:x.shape[0]-row_crop, col_crop: x.shape[1]-col_crop], (b, g, r))

    #направления
    b_to_g = calc_shift(g, b)
    r_to_g = calc_shift(g, r)

    #координfты в своей системе кооординат
    g_row_new, g_col_new = g_row - h - row_crop, g_col - col_crop
    r_row_new, r_col_new = find_match_point(g_row_new, g_col_new, r_to_g[0], r_to_g[1], r.shape)
    b_row_new, b_col_new = find_match_point(g_row_new, g_col_new, b_to_g[0], b_to_g[1], b.shape)

    #координаты r и b в изначальной системе
    b_row, b_col = g_row - h + b_row_new, g_col + b_col_new
    r_row, r_col = g_row + h + r_row_new, g_col + r_col_new

    b_shifted = make_shift(b, -b_row_new, -b_col_new)
    r_shifted = make_shift(r, -r_row_new, -r_col_new)
    aligned_img = np.dstack([r_shifted, g, b_shifted])

    return aligned_img, (b_row, b_col), (r_row, r_col)
