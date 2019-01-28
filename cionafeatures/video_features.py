import cv2
from cionafeatures import Features
import numpy as np
import matplotlib.cm
cmap = matplotlib.cm.get_cmap('RdBu')
cmapNeg = matplotlib.cm.get_cmap('Reds')
cmapPos = matplotlib.cm.get_cmap('Blues')
class FeatureVideo:
    def __init__(self, vid_file, feat_obj):
        self.features = feat_obj
        self.cap = cv2.VideoCapture(vid_file)
        self.playing = False

    def play_video_with_curvature(self,frames=None):
        """
        play video where curvature is color coded
        :param frames: use all frame if None else use only frames [frames[0]:frames[1]
        :return:
        """

        curv = self.features.get_curvature_spline()
        skel = self.features.get_skeleton()
        init_frame = 0
        if frames is not None:
            init_frame = frames[0]

        current_frame = -1
        self.playing = True
        pause = False
        while self.playing:
            ret, frame = self.cap.read()
            if current_frame > 0:
                for curv_mag, ptx, pty in zip(curv[current_frame, ::5],
                                              skel[current_frame, ::5, 0],
                                              skel[current_frame, ::5, 1]):
                    cmap = cmapPos
                    curv_mag = curv_mag * 5
                    if curv_mag < 0:
                        cmap = cmapNeg
                        curv_mag = -curv_mag

                    clr_base = np.array(cmap(curv_mag))[:3] * 255
                    cv2.circle(frame, tuple(np.round([ptx, pty]).astype(np.int32)), 5,
                               (int(clr_base[0]), int(clr_base[1]), int(clr_base[2])), -1)

            cv2.imshow('frame', frame)
            k = cv2.waitKey(500)
            if k & 0xFF == ord('q'):
                self.playing = False
            if k == 32:
                pause = not pause

            while pause:
                k = cv2.waitKey(500)
                if k == 32:
                    pause = not pause
            current_frame += 1
        return current_frame

    def play_video_points(self,frames=None):
        """
        play video where curvature is color coded
        :param frames: use all frame if None else use only frames [frames[0]:frames[1]
        :return:
        """

        param_skel = self.features.get_parametrize()
        skel = self.features.get_skeleton()
        init_frame = 0
        if frames is not None:
            init_frame = frames[0]

        current_frame = -1
        self.playing = True
        pause = False
        while self.playing:
            ret, frame = self.cap.read()
            if current_frame > 0:

                pt1 = self.features._get_point_along_skel([0.2],current_frame).flatten()
                pt2 = self.features._get_point_along_skel([0.6],current_frame).flatten()
                cv2.line(frame,
                         tuple(np.round(pt1).astype(np.int32)),
                         tuple(np.round(pt2).astype(np.int32)),
                         (255,255,255))

            cv2.imshow('frame', frame)
            k = cv2.waitKey(500)
            if k & 0xFF == ord('q'):
                self.playing = False
            if k == 32:
                pause = not pause

            while pause:
                k = cv2.waitKey(500)
                if k == 32:
                    pause = not pause
            current_frame += 1
        return current_frame

    def play_video_contour(self,frames=None):
        """
        play video with contour displayed
        :param frames: use all frame if None else use only frames [frames[0]:frames[1]
        :return:
        """

        shape = self.features.get_contour_shape()
        init_frame = 0
        if frames is not None:
            init_frame = frames[0]

        current_frame = -1
        self.playing = True
        pause = False
        while self.playing:
            ret, frame = self.cap.read()
            if current_frame > 0:
                cnt_pts = shape[current_frame]
                print(cnt_pts.shape)
                for i in range(len(cnt_pts) - 1):
                    pt1 = cnt_pts[i, :]
                    pt2 = cnt_pts[i+1 , :]
                    cv2.line(frame,
                             tuple(np.round(pt1).astype(np.int32)),
                             tuple(np.round(pt2).astype(np.int32)),
                             (50, 150, 10),
                             2)
            cv2.imshow('frame', frame)
            k = cv2.waitKey(500)
            if k & 0xFF == ord('q'):
                self.playing = False
            if k == 32:
                pause = not pause

            while pause:
                k = cv2.waitKey(500)
                if k == 32:
                    pause = not pause
            current_frame += 1
        return current_frame


if __name__== "__main__":
    import os
    print(os.getcwd())
    source_folder = os.getcwd()+os.path.join(r'\\..\\data\\')
    hdf5_features_1 = source_folder + '20180823_124209_1_15m0s_None_None_None_skeletons.hdf5'
    vid_folder = 'd:/JernejaS13toJulius/moreExamplesForJulius/'
    video_file = vid_folder + '20180823_124209_1_15m0s_None_None_None.avi'
    feat = Features(hdf5_features_1)
    p = FeatureVideo(video_file, feat)
    p.play_video_contour()