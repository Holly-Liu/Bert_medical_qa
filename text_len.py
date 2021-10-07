from typing import List
import bisect
import matplotlib.pyplot as plt
import pandas as pd


class TextLenAnalyser():
    def __init__(self, texts: List[str] = None, lens = None):
        """

        :param texts: 文本数组
        :param sep_arr:
        :param sep_count: 从0 到 max_len 等分sep_count 份
        """
        self.texts = texts
        self.max_len = 0
        self.lens = [len(text) for text in texts] if lens == None else lens
        self.max_len = max(self.lens)

    def analyse(self, sep_arr: List[int]):
        if not sep_arr:
            raise ValueError("needs sep array")
        if sep_arr[0] != 0:
            sep_arr = [0] + sep_arr
        labels = []
        counts = []
        for i, sep in enumerate(sep_arr):
            if i != len(sep_arr) - 1:
                labels.append(f"{sep} - {sep_arr[i+1]}")
                counts.append(0)
            else:
                labels.append(f"{sep} - inf")
                counts.append(0)
        for l in self.lens:
            bucket_index = bisect.bisect_right(sep_arr, l)
            counts[bucket_index-1] += 1

        plt.title("similar questions count bar")
        self.autolabel(plt.bar(labels, counts))
        plt.xlabel("similar question count")
        plt.ylabel("test count")
        plt.show()


    def auto_sep_analyse(self, sep_begin: int, sep_end, sep_delta):
        sep_arr = list(range(sep_begin, sep_end+1, sep_delta))
        self.analyse(sep_arr)



    def autolabel(self, rects):
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2. - 0.25, 1.01 * height, '%s' % int(height))
