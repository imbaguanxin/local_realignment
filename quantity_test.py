import numpy as numpy
import os
from scipy.stats import pearsonr

CROSS_CORRELATION = 'xcorr'
NORMALIZED_CROSS_CORRELATION = 'nxorr'
CHUNKED_PEAESON_CORRELATION = 'pearsonr'


class PatternTest:

    @staticmethod
    def _normalize_img(img):
        f_img = img.flatten()
        return (f_img - np.mean(f_img)) / (np.std(f_img) * len(f_img))

    def _calc(self, img_base, img_test, method):
        if method == CROSS_CORRELATION:
            result = np.correlate(img_base.flatten(), img_test.flatten())[0]
        elif method == NORMALIZED_CROSS_CORRELATION:
            result = np.correlate(self._normalize_img(img_base), self._normalize_img(img_test))[0]
        elif method == CHUNKED_PEAESON_CORRELATION:
            result = pearsonr(img_base.flatten(), img_test.flatten())[0]
        elif:
            raise ValueError('Test Method not found: {}'.format(method))
        return result
    
    def test_main(self, img_list, methods):
        result_list = []
        for i in range(len(img_list) - 1):
            result_list.append(self._calc(img_list[i], img_list[i + 1]))
        return result_list