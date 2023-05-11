import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)
np.random.seed(999)


def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    w = 0

    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    out = None

    # for all images to be stitched:
    for idx in tqdm(range(len(imgs)-1)):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]
        w += im2.shape[1]

        # TODO: 1.feature detection & matching

        kp1, des1 = orb.detectAndCompute(im1, None)
        kp2, des2 = orb.detectAndCompute(im2, None)

        kp1 = np.array(list(map(lambda x: x.pt, kp1)))
        kp2 = np.array(list(map(lambda x: x.pt, kp2)))

        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        matches = matches[:len(matches)//2]
        matches = np.array(list(map(lambda match: [
            match.distance, match.trainIdx, match.queryIdx, match.imgIdx], matches)))
        u = kp2[matches[:, 1].astype(np.int32)]
        v = kp1[matches[:, 2].astype(np.int32)]

        # TODO: 2. apply RANSAC to choose best H
        iterations = 1000
        threshold = 3
        max_cnt = 0
        for _ in range(iterations):
            random_choice = np.array(
                random.sample(list(range(len(matches))), 4))
            U = u[random_choice]
            V = v[random_choice]
            H = solve_homography(U, V)
            ones = np.ones((1, matches.shape[0]))
            M_train = np.concatenate((np.transpose(u), ones), axis=0)
            M_val = np.concatenate((np.transpose(v), ones), axis=0)
            M = np.dot(H, M_train)
            M = np.divide(M, M[-1])
            error = np.linalg.norm((M_val-M), ord=1, axis=0)
            cnt = sum(error < threshold)
            if cnt > max_cnt:
                # choose_best = choose
                H_best = H
                max_cnt = cnt
        # TODO: 3. chain the homographies
        last_best_H = last_best_H.dot(H_best)
        # TODO: 4. apply warping
        out = warping(im2, dst, last_best_H, 0,
                      h_max, 0, w_max)

    return out


if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x))
            for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)
