import cv2
from features import Features
import numpy as np
import matplotlib.cm
import matplotlib.pyplot as plt

class FeatureGraph:
    def __init__(self, feat_obj):
        self.features = feat_obj
        self.ax = ax = plt.subplot(111)

    def draw_graph_with_curvature(self,frames=None):
        """
        draw graph with curvature
        :param frames: use all frame if None else use only frames [frames[0]:frames[1]
        :return: graph
        """

        curv = np.array(self.features.get_curvature_spline())
        if frames is not None:
            curv = curv[frames[0]:frames[1]]
            x_range = range(frames[0],frames[1])
        else:
            x_range = range(len(curv))

        a = np.nanmean(curv, axis=1)
        self.ax.plot(a, 'b', linewidth=1)
        plt.show()



if __name__== "__main__":
    import os
    print(os.getcwd())
    source_folder = os.getcwd()+os.path.join(r'\\..\\data\\')
    hdf5_features_1 = source_folder + '20180823_124209_1_15m0s_None_None_None_skeletons.hdf5'
    feat = Features(hdf5_features_1)
    p = FeatureGraph(feat)
    p.draw_graph_with_curvature(frames=[100,200])