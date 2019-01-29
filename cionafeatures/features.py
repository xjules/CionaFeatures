import h5py
import numpy as np
from scipy.interpolate import UnivariateSpline

PARAM_SKEL = 'param_skel'
SKELETON = 'skeleton'
CURVATURE = 'curvature'
FULL_SHAPE = 'full_shape'
CONTOUR_DST = 'contour_dst'

class Features:
    def __init__(self, filename):
        self.features = h5py.File(filename,'r+')
        self.print_all()

    def print_all(self):
        try:
            for f in self.features:
                print(f)
        except Exception:
            print('File not loaded properly!')

    def get_skeleton(self):
        return self.features[SKELETON]

    def get_curvature_spline(self):
        '''
        Calculate the curvature using univariate splines. This method is slower and can fail
        badly if the fit does not work, so I am only using it as testing
        :return: curvatures
        '''
        if CURVATURE in self.features:
            return self.features[CURVATURE]
        try:
            skeleton = self.features[SKELETON]
        except:
            print('Missing skeleton info')
        def _curvature_fun(x_d, y_d, x_dd, y_dd):
            return (x_d * y_dd - y_d * x_dd) / (x_d * x_d + y_d * y_d) ** 1.5

        def _spline_curvature(skel):
            if np.any(np.isnan(skel)):
                return np.full(skel.shape[0], np.nan)

            x = skel[:, 0]
            y = skel[:, 1]
            n = np.arange(x.size)

            fx = UnivariateSpline(n, x, k=5)
            fy = UnivariateSpline(n, y, k=5)

            x_d = fx.derivative(1)(n)
            x_dd = fx.derivative(2)(n)
            y_d = fy.derivative(1)(n)
            y_dd = fy.derivative(2)(n)

            curvature = _curvature_fun(x_d, y_d, x_dd, y_dd)
            return curvature

        curvatures_fit = np.array([_spline_curvature(skel) for skel in skeleton])
        self.features[CURVATURE] = curvatures_fit
        return curvatures_fit

    def get_contour_dim(self):
        """
        create inter-contour distance
        :return:
        """
        if CONTOUR_DST in self.features:
            return self.features[CONTOUR_DST]

        skeleton = self.get_skeleton()
        len_s = len(skeleton)
        contour_left = self.features['contour_side1']
        contour_right = self.features['contour_side2']
        print(contour_left.shape)
        cnt_length = [np.sum((c[0] - c[1]) ** 2, axis=1) ** 0.5 for c in zip(contour_left, contour_right)]
        self.features[CONTOUR_DST] = cnt_length
        print('DIM',cnt_length[0])
        return cnt_length

    def get_contour_shape(self):
        """
        create the full-closed shape
        :return:
        """
        if FULL_SHAPE in self.features:
            return self.features[FULL_SHAPE]

        contour_left = self.features['contour_side1']
        contour_right = self.features['contour_side2']
        shape = [np.append(c[0], c[1][::-1], axis=0) for c in zip(contour_left, contour_right)]
        self.features[FULL_SHAPE] = shape
        return shape



    def get_parametrize(self):
        """
        create vectorized coverage of skeleton and store it as PARAM_SKEL in hd5
        :return: vectorized coverage of the skeleton
        """
        if PARAM_SKEL in self.features:
            return self.features[PARAM_SKEL]

        skel = self.features[SKELETON]
        #get vectors along central skeleton
        vec_dir = np.array([np.diff(b, axis=0) for b in skel])
        #get length of each segment
        vec_matrix = np.array([[self._ptl(a) for a in np.diff(b, axis=0)] for b in skel])
        #normalize vectors in vec_dir
        vec_dir_x = vec_dir[:, :, 0] / vec_matrix
        vec_dir_y = vec_dir[:, :, 1] / vec_matrix
        # dev_front = np.sum(vec_dir_x[1:10] * vec_dir_x[0:9]) + np.sum(vec_dir_y[1:10] * vec_dir_y[0:9])

        # get cumulative lengths along the skeleton
        vec_matrix_sum = np.cumsum(vec_matrix, axis=1)
        param_skel = vec_matrix_sum / vec_matrix_sum[:, -1].reshape(-1, 1)
        #store to feature file as param_skel
        self.features[PARAM_SKEL] = param_skel
        return param_skel

    def _ptl(self,a):
        return (np.sqrt(np.dot(a, a)))

    def _get_point_along_skel(self,t, id):
        try:
            param_skel = self.features[PARAM_SKEL]
            skel = self.features[SKELETON]
            point_x = np.interp([t], np.concatenate(([0], param_skel[id])), skel[id, :, 0])
            point_y = np.interp([t], np.concatenate(([0], param_skel[id])), skel[id, :, 1])
            return np.array([point_x[0], point_y[0]])
        except:
            return None



if __name__== "__main__":
    import os
    print(os.getcwd())
    source_folder = os.getcwd()+os.path.join(r'\\..\\data\\')
    hdf5_features_1 = source_folder + '20180823_124209_1_15m0s_None_None_None_skeletons.hdf5'
    f = Features(hdf5_features_1)
    f.get_contour_dim()