import numpy as np
from sklearn.cluster import KMeans
import ctypes
from ctypes import c_bool, c_double, c_int
import os

class DIMM:
    """
    ### Description

    A Dirichlet Multinomial Mixture Model in Python.

    ### Notes

    Please first compile the `./clibs/pyDIMM_libs.c` by the instructions in the head of that file.

    Check updates on https://github.com/Jck-R/pyDIMM

    """

    def __init__(
        self, 
        observe_data:np.ndarray, 
        n_components:int, 
        alpha_init:str = 'kmeans', 
        prior_label:np.ndarray = None, 
        print_log:bool = False, 
        max_sum_alpha:float = 1e10, 
        ):
        """
        ### Brief

        Initialize a DIMM instance.

        ### Parameters

        observe_data: A [n_cells * n_genes] ndarray showing the gene expression dataset to be trained by DIMM. n_genes and n_cells will be stored in `self.n_genes` and `self.n_cells`.

        n_components: A positive integer (>0) showing the components of the DIMM instance. i.e., how many Dirichlet Multinomial Models in this DIMM.

        alpha_init: A string to indicate how to initialize the prior label of the cells in datasets, which will affect the initialization of `alpha` and `pie`. We provide with 3 methods choice.
        - 'kmeans': Use `sklearn.KMeans` method to initialize the prior label.
        - 'random': Randomly initialize the cells into `n_components` equal parts of prior labels.
        - 'manual': Manually set the prior label of cells. You should input parameter `prior_label` if choose this method.

        prior_label: A [n_cells] ndarray showing the manually set prior_label of the cells in `observe_data`. The label should be integer in range (0 ~ `n_components`-1).

        print_log: A boolean variable indicating whether to print log information when available in each method.

        max_sum_alpha: A float number indicating the maximum value of alpha sum when initializing the alpha using Ronning's method. This is in case of the extreme situation such as all cells are the same so that the alpha could be infinitely large. We set the default value to be 1e10. Don't change it if you are not clear.

        ### Returns

        None.
        
        """
        assert(type(observe_data) == np.ndarray), 'Input `observe_data` is not numpy.ndarray type.'

        self.__DIMM_libpath = os.path.join(os.path.dirname(__file__), 'clibs', 'pyDIMM_libs.so')
        self.__lib = ctypes.cdll.LoadLibrary(self.__DIMM_libpath)
        self.__lib.EM_with_1dArr.argtypes = [c_int, c_int, c_int, ctypes.POINTER(c_double), ctypes.POINTER(c_double), ctypes.POINTER(c_double), c_int, c_double, c_double, c_double, ctypes.POINTER(c_double), ctypes.POINTER(c_double), c_bool]
        self.__lib.predict_with_1dArr.argtypes = [c_int, c_int, c_int, ctypes.POINTER(c_double), ctypes.POINTER(c_double), ctypes.POINTER(c_double), ctypes.POINTER(c_double), ctypes.POINTER(c_double)]
        
        self.observe_data = observe_data.astype('float64').copy()
        self.n_components = n_components
        self.alpha_init = alpha_init
        self.prior_label = prior_label
        self.print_log = print_log
        self.max_sum_alpha = max_sum_alpha

        self.n_genes = self.observe_data.shape[1]
        self.n_cells = self.observe_data.shape[0]
        self.__res_vec = np.zeros(3).astype('float64')

        self.model = dict()
        self.model['alpha'] = np.empty(shape=[self.n_components, self.n_genes], dtype='float64')
        self.model['pie'] = np.zeros(self.n_components, dtype='float64')
        self.model['delta'] = np.zeros([self.n_components, self.n_cells], dtype='float64').T #Pay attention to the .T here. It transpose the matrix in Python but not in c_POINTER (in memory).
        self.model['loglik'] = self.__res_vec[0]
        self.model['AIC'] = self.__res_vec[1]
        self.model['BIC'] = self.__res_vec[2]

        self.__init_alpha_pie(alpha_init, prior_label)

    def __init_alpha_pie(
        self, 
        init_mode, 
        prior_label, 
        ):
        if (init_mode == 'LoadingFromFile'):
            return
        elif (init_mode == 'kmeans'):
            cluster_cell = KMeans(n_clusters=self.n_components, ).fit_predict(self.observe_data)
        elif (init_mode == 'random'):
            cluster_col = np.arange(self.n_cells)
            np.random.shuffle(cluster_col)
            cluster_col = np.array_split(cluster_col, self.n_components)
            cluster_cell = np.zeros(self.n_cells)
            for k in range(self.n_components):
                cluster_cell[cluster_col[k]] = k
        elif (init_mode == 'manual'):
            assert (len(prior_label) == self.n_cells), 'Length of prior_label not equal to n_genes.'
            cluster_cell = prior_label
        else:
            print('alpha_init method only accept \'kmeans\', \'random\', \'manual\'.')
            assert(False)
        for k in range(self.n_components):
            self.model['pie'][k] = len(np.where(cluster_cell==k)[0])/len(cluster_cell)
            data_k = self.observe_data[np.where(cluster_cell==k)[0], :].T
            if (data_k.sum() == 0):
                self.model['alpha'][k] = np.zeros(self.n_genes)
                continue
            p_cluster = data_k.sum(axis=1)/data_k.sum()#p
            zero_idx = np.argwhere(np.all(data_k[..., :] == 0, axis=0))
            data_k = np.delete(data_k, zero_idx, axis=1)
            p_cell = data_k/data_k.sum(axis=0)#ppp
            p_E = p_cell.mean(axis=1)#pp
            if (p_cell.shape[1] == 1):
                p_var = p_cell.var(axis=1, ddof=0)#In case that only one cell in a cluster.
            else:
                p_var = p_cell.var(axis=1, ddof=1)
            p_var[np.where(p_var==0)] = p_var.mean()
            sum_alpha = 0
            for i in range(self.n_genes - 1):#Why G-1???
                # print(p_E[i]*(1-p_E[i])/p_var[i] - 1, p_E[i], p_var[i], i)
                if ((p_E[i] != 0) and (p_var[i] != 0)):#In case that varience are all zero.
                    tmp = p_E[i]*(1-p_E[i])/p_var[i] - 1
                    tmp = max(tmp, 1e-4)
                    sum_alpha = sum_alpha + (1/(self.n_genes-1))*np.log(tmp)
            sum_alpha = np.exp(sum_alpha)
            if (sum_alpha == 1):#In case that varience are all zero.
                sum_alpha = self.max_sum_alpha
            sum_alpha = min(sum_alpha, self.max_sum_alpha)#In case that alpha is too big, so that loglik calculation will overflow.
            self.model['alpha'][k] = sum_alpha * p_cluster
        self.model['alpha'] = self.model['alpha'] + 1e-4
        if (self.print_log):
            print("Initialize alpha successfully. Alpha = \n", self.model['alpha'])
            print("Initialize pie successfully. Pie = \n", self.model['pie'])
        
    def EM(
        self, 
        max_iter:int, 
        max_loglik_tol:float = 0,
        max_alpha_tol:float = 0,
        max_pie_tol:float = 0, 
        save_log:bool = False,
        ):
        """
        ### Brief

        Call C functions to run EM algorithm.

        ### Parameters

        max_iter: An integer showing the maximum iteration round. EM will stop immediately when iterate more than max_iter times.

        max_loglik_tol: An float number showing the maximum log_likelihood tolerance. EM will stop immediately when the difference of `log_likelihood` between the recent two rounds is smaller than this value.

        max_alpha_tol: An float number showing the maximum alpha tolerance (L2 norm). EM will stop immediately when the difference of `alpha` between the recent two rounds is smaller than this value.

        max_pie_tol: An float number showing the maximum pie tolerance (L2 norm). EM will stop immediately when the difference of `pie` between the recent two rounds is smaller than this value.

        ### Returns

        None.

        """
        ob_data_p = self.observe_data.ctypes.data_as(ctypes.POINTER(c_double))
        alpha_p = self.model['alpha'].ctypes.data_as(ctypes.POINTER(c_double))
        pie_p = self.model['pie'].ctypes.data_as(ctypes.POINTER(c_double))
        res_vec_p = self.__res_vec.ctypes.data_as(ctypes.POINTER(c_double))
        delta_p = self.model['delta'].ctypes.data_as(ctypes.POINTER(c_double))
        if (self.print_log):
            print('Call C functions to run EM algorithm. This may take a long time. Please wait...')
        #void EM_with_1dArr(int model_size, int n_cells, int n_genes, double *ob_data_1d, double *alpha_1d, double *pie, int max_iter, double max_pie_tol, double max_loglik_tol, double *res_vec, double **delta)
        self.__lib.EM_with_1dArr(c_int(self.n_components), c_int(self.n_cells), c_int(self.n_genes), ob_data_p, alpha_p, pie_p, c_int(max_iter), c_double(max_pie_tol), c_double(max_loglik_tol), c_double(max_alpha_tol), res_vec_p, delta_p, c_bool(save_log))
        self.model['loglik'] = self.__res_vec[0]
        self.model['AIC'] = self.__res_vec[1]
        self.model['BIC'] = self.__res_vec[2]
        if (self.print_log):
            print('EM finished.')
            
    def get_model(
        self, 
        ):
        """
        ### Brief

        Get all the DIMM model info in this class instance as a dictionary.
        
        Note: Ensure that you have runned `DIMM()` initialization function and `DIMM.EM` method so that there exists a DIMM model and dataset in this class instance.

        ### Parameters

        None.

        ### Returns

        res:A dictionary with 6 keys: `'alpha'`, `'pie'`, `'delta'`, `'loglik'`, `'AIC'`, `'BIC'`.
        - res['alpha'] is a [`self.n_components` * `self.n_genes`] ndarray, showing the alpha vector of each Dirichlet model.
        - res['pie'] is a [`self.n_components`] ndarray, showing the percentage of each Dirichlet model in the whole DIMM.
        - res['delta'] is a [`self.n_components` * `self.n_cells`] ndarray, with the item in position [k, c] showing the probability of cell c being sampled from model k.
        - res['loglik'] is a float64 number, showing the logarithm of likelihood of the DIMM given the dataset.
        - res['AIC'] is a float64 number, showing the AIC value of the DIMM given the dataset. 
        - res['BIC'] is a float64 number, showing the BIC value of the DIMM given the dataset. 

        """
        return self.model.copy()

    def predict(
        self, 
        cells:np.ndarray, 
        ):
        """
        ### Brief

        Predict the input cells dataset on the DIMM model trained before in the class instance.
        
        Note: Ensure that you have runned `DIMM.EM` method so that there exists a model in this class instance before call this `DIMM.predict` method.
        
        ### Parameters

        cells: A [C * `self.n_genes`] ndarray, showing the gene expression of C cells. Note that the columns must be `self.n_genes`.
        
        ### Returns

        res: A dictionary with 2 keys: 'label' and 'delta'.
        - res['label'] is an [C] length ndarray, with the item in position [c] showing the which model does cell c come from most likely. The model serial number 0 refer to `self.get_model()['alpha'][0]`.
        - res['delta'] is a [C * `self.n_components`] ndarray, with the item in position [c, k] showing the probability of cell c being sampled from model k among all models (so every row should sum to 1).
        - res['log_prob'] is a [C * `self.n_components`] ndarray, with the item in position [c, k] showing the log_probability of cell c being sampled from Dirichlet Multinomial model k (be independent to Mixture).

        """
        assert(type(cells) == np.ndarray), 'Input `cells` is not numpy.ndarray type.'
        assert (cells.shape[1] == self.n_genes), 'The columns of input cell (feature numbers) is not equal to self.n_genes.'
        cells = cells.astype('float64').copy()
        n_cells = cells.shape[0]
        res = dict()
        delta = np.zeros([self.n_components, n_cells], dtype='float64')
        log_prob = np.zeros([self.n_components, n_cells], dtype='float64')
        cells_p = cells.ctypes.data_as(ctypes.POINTER(c_double))
        alpha_p = self.model['alpha'].ctypes.data_as(ctypes.POINTER(c_double))
        pie_p = self.model['pie'].ctypes.data_as(ctypes.POINTER(c_double))
        delta_p = delta.ctypes.data_as(ctypes.POINTER(c_double))
        log_prob_p = log_prob.ctypes.data_as(ctypes.POINTER(c_double))
        #void predict_with_1dArr(int model_size, int n_cells, int n_genes, double *data_1d, double *alpha_1d, double *pie, double *res, double *delta_1d)
        self.__lib.predict_with_1dArr(c_int(self.n_components), c_int(n_cells), c_int(self.n_genes), cells_p, alpha_p, pie_p, delta_p, log_prob_p)
        label = np.argmax(delta, axis=0)
        res['label'] = label
        res['delta'] = delta.T
        res['log_prob'] = log_prob.T
        return res

    def save(
        self, 
        file_path, 
        ):
        """
        ### Brief

        Save the DIMM instance in .npy file.

        ### Parameters

        file_path: A string that indicate the file saving path.

        ### Returns

        None.

        """
        init_params = dict()
        init_params['observe_data'] = self.observe_data
        init_params['n_components'] = self.n_components
        init_params['alpha_init'] = self.alpha_init
        init_params['prior_label'] = self.prior_label
        init_params['print_log'] = self.print_log
        init_params['max_sum_alpha'] = self.max_sum_alpha
        info = dict()
        info['init_params'] = init_params
        info['dimm_model'] = self.get_model()
        np.save(file_path, info)
        
    @classmethod
    def load(
        cls, 
        file_path, 
        ):
        """
        ### Brief

        Load the DIMM file.

        Note: The file must be saved by a DIMM instance with `save()` method.

        ### Parameters

        file_path: A string that indicate the path of the file to be loaded.

        ### Returns

        A dimm instance with the information in the file.

        """
        info = np.load(file_path, allow_pickle=True).item()
        assert('dimm_model' in info.keys()), 'Load failed. Your file is not DIMM model file.'

        observe_data = info['init_params']['observe_data']
        n_components = info['init_params']['n_components']
        alpha_init = info['init_params']['alpha_init']
        prior_label = info['init_params']['prior_label']
        print_log = info['init_params']['print_log']
        max_sum_alpha = info['init_params']['max_sum_alpha']
        
        dimm_load = cls(observe_data=observe_data, n_components=n_components, alpha_init='LoadingFromFile', prior_label=prior_label, print_log=print_log, max_sum_alpha=max_sum_alpha)
        dimm_load.model = info['dimm_model'].copy()
        dimm_load.alpha_init = alpha_init

        return dimm_load

    @classmethod
    def sample(
        cls, 
        alpha:np.ndarray, 
        pie:np.ndarray,
        N_multinomial:int, 
        size:int,
        ):
        """
        ### Brief

        Sample from given parameters of a Dirichlet Multinomial Mixture Model.
        
        ### Parameters

        alpha: A [n_alpha * n_genes] ndarray showing n_alpha of alpha vectors.

        pie: A [n_alpha] length ndarray showing the probability of each alpha vector being sampled.

        N_multinomial: An integer indicating the N in multinomial distribution.

        size: An integer showing how many sets of sampled data you want.

        ### Returns

        A [size * alpha.shape[1]] ndarray data of sampled data.

        """
        assert(pie.sum() == 1), 'pie do not sum to 1.'
        data = np.zeros([size, alpha.shape[1]])
        for i in range(size):
            alpha_sampled_idx = np.random.choice(a=np.arange(len(pie)), p=pie)
            p = np.random.dirichlet(alpha[alpha_sampled_idx])
            data[i, :] = np.random.multinomial(n=N_multinomial, pvals=p)
        return data