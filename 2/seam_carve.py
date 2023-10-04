import numpy as np


def energy(brightness):
    x_grad = np.zeros_like(brightness, dtype='float64')
    y_grad = np.zeros_like(brightness, dtype='float64')

    x_grad[:, 0] = brightness[:, 1] - brightness[:, 0]
    x_grad[:, -1] = brightness[:, -1] - brightness[:, -2]
    x_grad[:, 1:-1] = 0.5 * (brightness[:, 2:] - brightness[:, : - 2])

    y_grad[0] = brightness[1] - brightness[0]
    y_grad[-1] = brightness[-1] - brightness[-2]
    y_grad[1:-1] = 0.5 * (brightness[2:] - brightness[:-2])

    nrg = np.sqrt(x_grad ** 2 + y_grad ** 2)
    return np.clip(nrg, 0, 255)


def compute_energy(image):
    img = image.astype('float64')

    Y_matrix = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

    energy_matrix = energy(Y_matrix)
    return energy_matrix


def compute_seam_matrix(energy, mode, mask=None):
    # mode = vertical -> horizontal seam
    if mode == 'vertical':
        energy = energy.T

    if mask is not None:
        mask = mask.astype('float64')
        if mode == 'vertical':
            mask = mask.T
        energy += mask * (256 * energy.shape[0] * energy.shape[1])

    seam_matrix = np.zeros_like(energy, dtype='float64')
    seam_matrix[0] = energy[0]

    for i in range(1, seam_matrix.shape[0]):
        seam_matrix[i, 0] = np.min(seam_matrix[i - 1, 0:2]) + energy[i, 0]
        seam_matrix[i, -1] = np.min(seam_matrix[i - 1, -2:-1]) + energy[i, -1]

        for j in range(1, seam_matrix.shape[1]):
            seam_matrix[i, j] = np.min(seam_matrix[i - 1, j-1:j+2]) + energy[i, j]

    return seam_matrix if mode == 'horizontal' else seam_matrix.T


def remove_minimal_seam(image, seam_matrix, mode, mask=None):
    if mode == 'vertical shrink':
        image = image.transpose(1, 0, 2)
        seam_matrix = seam_matrix.T
        if mask is not None:
            mask = mask.T

    min_seam_elms_idx = np.zeros(seam_matrix.shape[0], dtype='int')
    min_seam_elms_idx[-1] = np.argmin(seam_matrix[-1])

    for i in range(seam_matrix.shape[0] - 1, 0, -1):
        j = min_seam_elms_idx[i]
        if j == 0:
            min_seam_elms_idx[i - 1] = np.argmin(seam_matrix[i-1, j:j+2])
            continue
        if j == seam_matrix.shape[1] - 1:
            min_seam_elms_idx[i - 1] = j - 1 + np.argmin(seam_matrix[i - 1, j-1:j + 1])
            continue
        min_seam_elms_idx[i - 1] = j - 1 + np.argmin(seam_matrix[i - 1, j-1:j + 2])

    seam_mask = np.zeros(seam_matrix.shape, dtype='uint8')
    seam_mask[np.arange(seam_mask.shape[0]), min_seam_elms_idx] = 1

    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    new_r = r[seam_mask == 0].reshape(r.shape[0], -1)
    new_g = g[seam_mask == 0].reshape(g.shape[0], -1)
    new_b = b[seam_mask == 0].reshape(b.shape[0], -1)

    new_mask = None
    if mask is not None:
        new_mask = mask[seam_mask == 0].reshape(mask.shape[0], -1)
        if mode == 'vertical shrink':
            new_mask = new_mask.T

    if mode == 'vertical shrink':
        new_img = np.dstack([new_r.T, new_g.T, new_b.T]).astype('uint8')
        return new_img, new_mask, seam_mask.T
    else:
        new_img = np.dstack([new_r, new_g, new_b]).astype('uint8')
        return new_img, new_mask, seam_mask


def seam_carve(image, mode, mask=None):
    energy = compute_energy(image)
    seam_matrix = compute_seam_matrix(energy, mode.split()[0], mask)

    new_img, new_mask, seam_mask = remove_minimal_seam(image, seam_matrix, mode, mask)

    return new_img, new_mask, seam_mask
