import numpy as np


def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]
    H = None

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    # TODO: 1.forming A
    A = np.zeros((2*N, 9))
    for i in range(N):
        ux, uy = u[i]
        vx, vy = v[i]
        A[2*i] = np.array([ux, uy, 1, 0, 0, 0, -ux*vx, -uy*vx, -vx])
        A[2*i+1] = np.array([0, 0, 0, ux, uy, 1, -ux*vy, -uy*vy, -vy])
    # TODO: 2.solve H with A
    U, S, VT = np.linalg.svd(A, full_matrices=True)
    V = VT.T
    h = VT[-1, :]/VT[-1, -1]
    H = np.reshape(h, (3, 3))
    return H


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """
    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)

    # TODO: 1.meshgrid the (x,y) coordinate pairs
    x, y = np.meshgrid(np.arange(xmin, xmax, 1),
                       np.arange(ymin, ymax, 1), sparse=False)
    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate
    x_row = x.reshape(((xmax - xmin)*(ymax - ymin), 1))
    y_row = y.reshape(((xmax - xmin)*(ymax - ymin), 1))
    homogeneous_coords = np.concatenate((x_row, y_row, np.ones(
        ((xmax-xmin)*(ymax-ymin), 1))), axis=1)
    if direction == 'b':
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        u = np.matmul(H_inv, homogeneous_coords.T).T
        u = np.divide(u, u[:, 2, None])
        ux = np.round(u[:, 0]).reshape(
            ((ymax - ymin), (xmax - xmin))).astype(np.int32)
        uy = np.round(u[:, 1]).reshape(
            ((ymax - ymin), (xmax - xmin))).astype(np.int32)
        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)
        mask = ((0 < uy)*(uy < h_src))*((0 < ux)*(ux < w_src))
        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates
        dst[y[mask], x[mask]] = src[uy[mask], ux[mask]]
        # TODO: 6. assign to destination image with proper masking
        pass

    elif direction == 'f':
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        v = np.matmul(H, homogeneous_coords.T).T
        v = np.divide(v, v[:, 2, None])
        vx = np.round(v[:, 0].reshape(
            (ymax - ymin), (xmax - xmin))).astype(np.int32)
        vy = np.round(v[:, 1].reshape(
            (ymax-ymin), (xmax - xmin))).astype(np.int32)
        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)

        # TODO: 5.filter the valid coordinates using previous obtained mask
        masked_vx = np.clip(vx, 0, w_dst-1)
        masked_vy = np.clip(vy, 0, h_dst-1)
        # TODO: 6. assign to destination image using advanced array indicing
        dst[masked_vy, masked_vx] = src

    return dst
