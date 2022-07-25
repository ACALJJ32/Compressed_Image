import numpy as np
import mmcv
import cv2
import os


def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.

    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.

    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.

    Returns:
        ndarray: reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    if len(img.shape) == 2:
        img = img[..., None]
        return img
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img


def psnr(img1, img2, crop_border=0, input_order='HWC', convert_to=None):
    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')

    img1, img2 = img1.astype(np.float32), img2.astype(np.float32)
    if isinstance(convert_to, str) and convert_to.lower() == 'y':
        img1 = mmcv.bgr2ycbcr(img1 / 255., y_only=True) * 255.
        img2 = mmcv.bgr2ycbcr(img2 / 255., y_only=True) * 255.
    elif convert_to is not None:
        raise ValueError('Wrong color model. Supported values are '
                         '"Y" and None.')

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    mse_value = np.mean((img1 - img2)**2)
    if mse_value == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse_value))


def _ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1, img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def ssim(img1, img2, crop_border=0, input_order='HWC', convert_to=None):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edges of an image. These
            pixels are not involved in the SSIM calculation. Default: 0.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        convert_to (str): Whether to convert the images to other color models.
            If None, the images are not altered. When computing for 'Y',
            the images are assumed to be in BGR order. Options are 'Y' and
            None. Default: None.

    Returns:
        float: ssim result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if isinstance(convert_to, str) and convert_to.lower() == 'y':
        img1, img2 = img1.astype(np.float32), img2.astype(np.float32)
        img1 = mmcv.bgr2ycbcr(img1 / 255., y_only=True) * 255.
        img2 = mmcv.bgr2ycbcr(img2 / 255., y_only=True) * 255.
        img1 = np.expand_dims(img1, axis=2)
        img2 = np.expand_dims(img2, axis=2)
    elif convert_to is not None:
        raise ValueError('Wrong color model. Supported values are '
                         '"Y" and None')

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ssim(img1[..., i], img2[..., i]))
    return np.array(ssims).mean()


def calculate_psnr_frame(path1, path2, frame_name):
    frame_name = frame_name + '.jpg'
    path1 = os.path.join(path1, frame_name)
    path2 = os.path.join(path2, frame_name)

    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    return psnr(img1, img2)


def calculate_psnr_frames(path1, path2):
    frames_list1 = os.listdir(path1)

    psnr_tol = 0.
    for frame in frames_list1:
        frame_name1 = os.path.join(path1, frame)
        frame_name2 = os.path.join(path2, frame)
        print(frame_name1)
        print(frame_name2)
        frame_name1 = cv2.imread(frame_name1)
        frame_name2 = cv2.imread(frame_name2)
        psnr_tol += psnr(frame_name1, frame_name2)
    return psnr_tol / len(frames_list1)


def calculate_ssim_frames(path1, path2):
    frames_list1 = os.listdir(path1)

    psnr_tol = 0.
    for frame in frames_list1:
        frame_name1 = os.path.join(path1, frame)
        frame_name2 = os.path.join(path2, frame)
        print(frame_name1)

        frame_name1 = cv2.imread(frame_name1)
        frame_name2 = cv2.imread(frame_name2)
        psnr_tol += ssim(frame_name1, frame_name2)
    return psnr_tol / len(frames_list1)


def tmp(input_path, output_path):
    img_list = os.listdir(input_path)
    img_list = [os.path.join(input_path, w) for w in img_list if w.endswith('.jpg')]

    for img_name in img_list:
        # tmp_name = os.path.basename(img_name)[:17] + '.jpg'
        img = cv2.imread(img_name)

        resize = cv2.resize(img, (512,512))
        cv2.imwrite(os.path.join(output_path, os.path.basename(img_name)), resize)

if __name__ == "__main__":
    inputs = '/data1/AIM2022/LIVE1/jpeg_qf_10/live1/gt'
    gts = '/data1/ljj/QGCN_v2/results/live_hinet_patch512_flikr2k_epoch57'
    psnr_num = calculate_psnr_frames(inputs, gts)
    print(psnr_num)
