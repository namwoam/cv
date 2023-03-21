import numpy as np
import cv2


class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1
        self.debug_mode = True

    def get_octave_blur(self, source):
        result = [source.copy()]
        for i in range(1, self.num_guassian_images_per_octave):
            result.append(cv2.GaussianBlur(source, ksize=(0, 0),
                          sigmaX=self.sigma**i, sigmaY=self.sigma**i))
        return result

    def get_keypoints(self, image):
        if self.debug_mode:
            cv2.imwrite("./tmp/source.png", image)
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        blurred_images = []
        blurred_images.append(self.get_octave_blur(image))
        resized_last_image = cv2.resize(
            blurred_images[-1][-1], (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        blurred_images.append(self.get_octave_blur(resized_last_image))
        if self.debug_mode:
            for i,  octave in enumerate(blurred_images):
                for k, img in enumerate(octave):
                    cv2.imwrite(f"./tmp/Blurred{i+1}-{k+1}.png", img)

        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = []
        for octave in blurred_images:
            dog_images.append([])
            for i in range(len(octave) - 1):
                dog_images[-1].append(cv2.subtract(octave[i+1], octave[i]))

        if self.debug_mode:
            for i, octave in enumerate(dog_images):
                for k, img in enumerate(octave):
                    cv2.imwrite(f"./tmp/DoG{i+1}-{k+1}.png", img)

        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint
        keypoints = []
        for i, octave in enumerate(dog_images):
            stacked_DoG = np.stack(octave)
            size_z, size_y, size_x = stacked_DoG.shape
            for z in range(1, size_z-1):
                for y in range(1, size_y-1):
                    for x in range(1, size_x-1):
                        subspace = stacked_DoG[z-1:z+2, y-1:y+2, x-1:x+2]
                        if stacked_DoG[z, y, x] == subspace.max() or stacked_DoG[z, y, x] == subspace.min():
                            if abs(stacked_DoG[z, y, x]) >= self.threshold:
                                keypoints.append(
                                    (y, x) if i == 0 else (y*2, x*2))
                                if self.debug_mode:
                                    print((y, x))
        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.unique(keypoints, axis=0)
        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:, 1], keypoints[:, 0]))]
        return keypoints
