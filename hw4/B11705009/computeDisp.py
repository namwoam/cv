import numpy as np
import cv2.ximgproc as xip
import cv2


def computeDisp(Il, Ir, max_disp):
    max_disp += 1
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)
    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel
    # [Tips] Compute cost both Il to Ir and Ir to Il for later left-right consistency\
    max_disp += 1
    cost_l2r = np.zeros((max_disp, w, h), dtype=np.float32)
    cost_r2l = np.zeros((max_disp, w, h), dtype=np.float32)
    for d in range(max_disp):
        for x in range(w):
            d_l = max(x-d, 0)
            d_r = min(x+d, w-1)
            for y in range(h):
                cost_l2r[d, x, y] = np.sum(np.abs((Il[y, x]-Ir[y, d_l])))
                cost_r2l[d, x, y] = np.sum(np.abs(Ir[y, x]-Il[y, d_r]))
    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparty)
    for d in range(max_disp):
        cost_l2r[d,] = xip.jointBilateralFilter(
            np.swapaxes(Il, 0, 1), cost_l2r[d,], 20, 5, 5)
        cost_r2l[d,] = xip.jointBilateralFilter(
            np.swapaxes(Ir, 0, 1), cost_r2l[d,], 20, 5, 5)
    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all
    disparty_r2l = np.argmin(cost_r2l, axis=0)
    disparty_l2r = np.argmin(cost_l2r, axis=0)
    x, y = np.meshgrid(np.arange(0, w, 1), np.arange(
        0, h, 1), sparse=True,  indexing='ij')
    valid_disparty = np.where(
        disparty_l2r[x, y] == disparty_r2l[x - disparty_l2r[x, y], y], disparty_l2r[x, y], -1)
    # print("hole:", np.sum(np.where(valid_disparty == -1, 1, 0)))
    # print("total pixel:", w*h)
    padded_disparity = cv2.copyMakeBorder(valid_disparty, 1, 1, 1, 1,
                                          cv2.BORDER_CONSTANT, value=max_disp)
    for x in range(w):
        for y in range(h):
            dl, dr = 0, 0
            px, py = 1, 1
            if valid_disparty[x, y] != -1:
                continue
            while padded_disparity[x+px+dl, y+py] == -1 or padded_disparity[x+px+dr, y+py] == -1:
                if padded_disparity[x+px+dl, y+py] == -1:
                    dl -= 1
                if padded_disparity[x+px+dr, y+py] == -1:
                    dr += 1
            valid_disparty[x, y] = min(
                padded_disparity[x+px+dl, y+py], padded_disparity[x+px+dr, y+py])
    labels = xip.weightedMedianFilter(
        Il.astype(np.uint8), valid_disparty.astype(np.uint8).T, 20, 1)
    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering
    return labels.astype(np.uint8)
