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
            result.append(cv2.GaussianBlur(source, (0, 0), self.sigma**i))
        return result

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        blurred_images = []
        blurred_images.append(self.get_octave_blur(image))
        resized_last_image = cv2.resize(
            blurred_images[-1][-1], (0, 0), fx=0.5, fy=0.5)
        blurred_images.append(self.get_octave_blur(resized_last_image))
        if self.debug_mode:
            for i,  octave in enumerate(blurred_images):
                for k, img in enumerate(octave):
                    cv2.imwrite(f"./tmp/Blurred{i+1}-{k+1}.png" , img)

        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = []

        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint

        # Step 4: Delete duplicate keypoints
        # - Function: np.unique

        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:, 1], keypoints[:, 0]))]
        return keypoints
