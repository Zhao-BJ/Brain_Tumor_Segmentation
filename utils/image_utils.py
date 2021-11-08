import numpy as np


def object_crop(org_arr, roi_size, d_coo, h_coo, w_coo):
    obj_region = np.zeros(roi_size, dtype=org_arr.dtype)
    crop_coord = np.array([d_coo - (roi_size[0] // 2), d_coo + (roi_size[0] // 2), h_coo - (roi_size[1] // 2), h_coo + (roi_size[1] // 2), w_coo - (roi_size[2] // 2), w_coo + (roi_size[2] // 2)], dtype=int)
    err_coord = [0, roi_size[0], 0, roi_size[1], 0, roi_size[2]]

    if crop_coord[0] < 0:
        err_coord[0] = abs(crop_coord[0])
        crop_coord[0] = 0

    if crop_coord[2] < 0:
        err_coord[2] = abs(crop_coord[2])
        crop_coord[2] = 0
    if crop_coord[4] < 0:
        err_coord[4] = abs(crop_coord[4])
        crop_coord[4] = 0

    if crop_coord[1] > org_arr.shape[0]:
        err_coord[1] = err_coord[1] - (crop_coord[1] - org_arr.shape[0])
        crop_coord[1] = org_arr.shape[0]

    if crop_coord[3] > org_arr.shape[1]:
        err_coord[3] = err_coord[3] - (crop_coord[3] - org_arr.shape[1])
        crop_coord[3] = org_arr.shape[1]
    if crop_coord[5] > org_arr.shape[2]:
        err_coord[5] = err_coord[5] - (crop_coord[5] - org_arr.shape[2])
        crop_coord[5] = org_arr.shape[2]

    obj_region[err_coord[0]:err_coord[1], err_coord[2]: err_coord[3], err_coord[4]: err_coord[5]] = org_arr[crop_coord[0]: crop_coord[1], crop_coord[2]: crop_coord[3], crop_coord[4]: crop_coord[5]]

    return obj_region, err_coord, crop_coord