import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
        self.debug_mode = False
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(
            img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(
            guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = padded_guidance.astype(np.float64)
        padded_guidance /= 255.0
        assert padded_img.shape[:2] == padded_guidance.shape[:2]
        output = img.copy()
        gaussain_kernal = cv2.getGaussianKernel(
            ksize=self.wndw_size, sigma=self.sigma_s) * cv2.getGaussianKernel(ksize=self.wndw_size, sigma=self.sigma_s).T
        if padded_img.shape != padded_guidance.shape:
            padded_guidance = padded_guidance.reshape(
                (padded_guidance.shape[0], padded_guidance.shape[1], 1))
        windows = np.lib.stride_tricks.sliding_window_view(
            padded_guidance, (self.wndw_size, self.wndw_size, 1))
        if self.debug_mode:
            print(windows.shape)
        for x, y in np.ndindex(windows.shape[:2]):
            window = windows[x][y]
            color_difference_square = np.zeros(
                (self.wndw_size, self.wndw_size), dtype=np.float64)
            #color_difference_square = np.sum([np.multiply(el - el[self.wndw_size//2][self.wndw_size//2], el - el[self.wndw_size//2][self.wndw_size//2]).reshape((self.wndw_size, self.wndw_size)) for el in window],  axis=0)
            for channel in window:
                middle_element = channel[self.wndw_size//2][self.wndw_size//2]
                single_color_difference_square = np.multiply(
                    (channel - middle_element), (channel - middle_element))
                if self.debug_mode:
                    print("Value of center:")
                    print(middle_element)
                    print("Color of one channel:")
                    print(channel.reshape((19,19)))
                    print("Color difference squared")
                    print(single_color_difference_square.reshape((19,19)))
                    pass
                single_color_difference_square = single_color_difference_square.reshape(
                    (self.wndw_size, self.wndw_size))
                color_difference_square += single_color_difference_square
            range_kernal = np.exp(-(color_difference_square /
                                  (2*(self.sigma_r**2))))
            kernal = np.multiply(range_kernal, gaussain_kernal)
            # print(kernal.reshape(19,19))
            kernal /= np.sum(kernal)
            kernal = np.dstack([kernal, kernal, kernal])
            px = x+self.pad_w
            py = y+self.pad_w
            img_window = padded_img[px-self.wndw_size//2:px +
                                    self.wndw_size//2+1, py-self.wndw_size//2:py+self.wndw_size//2+1, :]
            new_intensity = np.sum(np.multiply(img_window,
                                   kernal), axis=(0, 1))
            output[x][y] = new_intensity

        """
        for channel in np.moveaxis(padded_guidance, 2, 0):
            range_kernal = np.zeros((self.wndw_size, self.wndw_size) , dtype=np.int32)
            
            for (x, y), window in np.ndenumerate(windows):
                middle_index = self.wndw_size//2
                middle_el = window[middle_index][middle_index]
                single_color = (window - middle_el)*(window - middle_el)
                range_kernal += single_color
            range_kernal = np.exp(-range_kernal/2*self.sigma_r)
            kernal = range_kernal * gaussain_kernal
            kernal = kernal / np.sum(kernal)
            print(kernal)

        """
        ### TODO ###

        return np.clip(output, 0, 255).astype(np.uint8)
