# v 2021.04.12
# A generalized version of channel wise Saab
# modified from https://github.com/ChengyaoWang/PixelHop-_c-wSaab/blob/master/cwSaab.py
# Note: Depth goal may not achieved if no nodes's energy is larger than energy threshold or too few SaabArgs/shrinkArgs, (warning generates)

import numpy as np 
import copy
import gc, time
import heapq
from saab import Saab

def EFDM(x, y):
    """Exact Feature Distribution Matching
        x is content features, y is style features
    """
    index_x = np.argsort(x)  # Indices that would sort x
    sorted_y = np.sort(y)  # Sorted y values

    inverse_index = np.argsort(index_x) # ranking
    result = sorted_y[inverse_index]
    
    return result

def StyleSwap(C, S, Shrink, invShrink, win, stride=1):

    def Conv(X, K, shrinkArg):
        assert ('func' in shrinkArg.keys()), "shrinkArg must contain key 'func'!"

        # Extract a set of patches for content feats and style feats
        patch_X = shrinkArg['func'](X, shrinkArg)
        patch_K = shrinkArg['func'](K, shrinkArg)

        # normalized style patch
        norm_K = patch_K / np.linalg.norm(patch_K, axis=-1, keepdims=True)

        S1 = list(patch_X.shape)
        S2 = list(norm_K.shape)
        patch_X = patch_X.reshape(-1, S1[-1])
        norm_K = norm_K.reshape(-1, S2[-1])
        
        transformed = patch_X @ norm_K.transpose()
        transformed = transformed.reshape(S1[0],S1[1],S1[2],-1)
        # print(transformed.shape)
        # # (1, 380, 508, 193040)
        # # (1, 188, 252, 47376)
        # # (1, 92, 124, 11408)

        # # Extract a set of patches for content feats and style feats
        # patch_X = shrinkArg['func'](X[...,:5], shrinkArg)
        # patch_K = shrinkArg['func'](K[...,:5], shrinkArg)

        # # normalized style patch
        # norm_K = patch_K / np.linalg.norm(patch_K, axis=-1, keepdims=True)

        # S1 = list(patch_X.shape)
        # S2 = list(norm_K.shape)
        # patch_X = patch_X.reshape(-1, S1[-1])
        # norm_K = norm_K.reshape(-1, S2[-1])
        
        # transformed = patch_X @ norm_K.transpose()
        # transformed = transformed.reshape(S1[0],S1[1],S1[2],-1)
        # print(transformed.shape)
            
        return transformed, patch_K.reshape(-1, S2[-1])
        # ret = shrinkArg['func'](K, shrinkArg)
        # return transformed, ret.reshape(-1, ret.shape[-1])
    
    def ChannelWiseArgmax(input):
        # find the index of the max value along the last dimension
        indices = np.argmax(input, axis=-1)
        # set all elements to 0
        input.fill(0)
        # set the values at the max indices to 1
        np.put_along_axis(input, indices[..., np.newaxis], 1, axis=-1)
        return input
    
    def BijMatching(input):
        # Determine the number of channels
        num_channels = input.shape[-1]
        # print("copying")
        # input_copy = copy.deepcopy(input)
        # print("copied")

        # global sort in descending order
        t1 = time.time()
        print("begin argsort")
        idx_b, idx_h, idx_w, idx_c = np.unravel_index(np.argsort(-input, axis=None), input.shape)
        print("argsort time", time.time()-t1)
        skip_ch = set()
        skip_po = set()

        # Reset input to zero
        input.fill(0)

        # Pop elements from the heap and set the corresponding index in input to 1
        count = idx = 0
        t1 = time.time()
        print("begin BijMatching")
        while count < num_channels:
            # if count % 1000 == 0:
            #     print(count)
            batch, channel, i, j = idx_b[idx], idx_c[idx], idx_h[idx], idx_w[idx]
            while channel in skip_ch or f"{i}_{j}" in skip_po:
                idx += 1
                batch, channel, i, j = idx_b[idx], idx_c[idx], idx_h[idx], idx_w[idx]
            input[batch,i,j,channel] = 1
            skip_ch.add(channel)
            skip_po.add(f"{i}_{j}")
            count += 1
        print("BijMatching time", time.time()-t1)
        return input
    
    def DeConv(X, K, shrinkArg):
        assert ('func' in shrinkArg.keys()), "shrinkArg must contain key 'func'!"
        
        S1 = list(X.shape)
        X = X.reshape(-1, S1[-1])

        transformed = X @ K
        transformed = transformed.reshape(S1[0],S1[1],S1[2],-1)

        # not consider for "SaabArg['num_AC_kernels'] != -1"

        transformed = shrinkArg['invfunc'](transformed, shrinkArg)
        
        return transformed
    
    arg = {'invfunc': invShrink, 'func': Shrink, 'win':win, 'stride': stride, 'pool': 1, 'pad': 0}
    
    # print("C.shape, S.shape", C.shape, S.shape)
    Correlation, SPatch = Conv(C, S, arg)
    # print("Correlation.shape, SPatch.shape", Correlation.shape, SPatch.shape)
    # OneHot = BijMatching(Correlation)
    OneHot = ChannelWiseArgmax(Correlation)
    # print("OneHot.shape", OneHot.shape)
    Swapped = DeConv(OneHot, SPatch, arg)
    # print("Swapped.shape", Swapped.shape)

    assert Swapped.shape == C.shape, "Wrong implementation"

    return Swapped, OneHot

# at later layer of hops, the eigenvalues are small and cause problem
def WCT(fc, fs, alpha=1):
    """
        fc: [C, feats_num]
        fs: [C, feats_num]
    """
    C, feats_num = fc.shape

    # Compute whitening transform
    Cm = np.mean(fc, axis=1, keepdims=True)
    fc_centered = fc - Cm
    # np.linalg.eig((fc_centered@fc_centered.T)/(feats_num-1)) == np.linalg.eig(np.cov(fc_centered))
    Ceigval, Cnormeigvec = np.linalg.eig((fc_centered@fc_centered.T)/(feats_num-1))
    # Dc = np.diag(1.0 / np.sqrt(Ceigval + 1e-5))
    Dc = np.diag(1.0 / np.sqrt(Ceigval))
    whitening = Cnormeigvec@Dc@Cnormeigvec.T

    # Compute coloring transform
    Sm = np.mean(fs, axis=1, keepdims=True)
    fs_centered = fs - Sm
    Seigval, Snormeigvec = np.linalg.eig((fs_centered@fs_centered.T)/(feats_num-1))
    # Ds = np.diag(np.sqrt(Seigval + 1e-5))
    Ds = np.diag(np.sqrt(Seigval))
    coloring = Snormeigvec@Ds@Snormeigvec.T
    
    # Apply transforms
    whitened = whitening@fc_centered
    colored = alpha * (coloring@whitened + Sm) + (1 - alpha) * fc

    return colored

def StyleTransfer(transferred, X, args, shape, Shrink, invShrink):
    """
        transferred: deepcopyed X
        X: the Saab feature maps
    """
    b, h, w, c = shape
    transmethod = args['transmethod']
    SwapWin = args['SwapWin']
    OneHot = None # only for StyleSwap
    if transmethod == "EFDM_spatial":
        # in spatial
        for idx in range(c):
            transferred[0,...,idx] = EFDM(X[0,...,idx].reshape(-1), X[1,...,idx].reshape(-1)).reshape((h, w))
    elif transmethod == "EFDM_spectral":
        # in spectral
        for ver in range(h):
            for hor in range(w):
                transferred[0,ver,hor,:] = EFDM(X[0,ver,hor,:], X[1,ver,hor,:])
    elif transmethod == "EFDM_combined":
        # in spatial and spectral
        s = 5
        for ver in range(0, h, s):
            for hor in range(0, w, s):
                transferred[0,ver:ver+s,hor:hor+s,:] = EFDM(X[0,ver:ver+s,hor:hor+s,:].reshape(-1), X[1,ver:ver+s,hor:hor+s,:].reshape(-1)).reshape((s if ver+s<=h else h-ver, s if hor+s<=w else w-hor, c))
    elif transmethod == "StyleSwap":
        t = 2
        while t:
            transferred[0], OneHot = StyleSwap(transferred[0][None,...], transferred[1][None,...], Shrink, invShrink, SwapWin, 1)
            t -= 1
            print(t)
    elif transmethod == "WCT":
        transferred[0] = np.moveaxis(WCT(np.moveaxis(transferred[0], -1, 0).reshape((c,h*w)), np.moveaxis(transferred[1], -1, 0).reshape((c,h*w))).reshape((c,h,w)), 0, -1)
    
    return transferred, OneHot

# Python Virtual Machine's Garbage Collector
def gc_invoker(func):
    def wrapper(*args, **kw):
        value = func(*args, **kw)
        gc.collect()
        time.sleep(0.5)
        return value
    return wrapper

class cwSaab():
    def __init__(self, depth=1, TH1=0.01, TH2=0.005, SaabArgs=None, shrinkArgs=None, load=False):
        self.trained = False
        self.split = False
        self.traindata_transformed = []
        self.transformed = []
        
        if load == False:
            assert (depth > 0), "'depth' must > 0!"
            assert (SaabArgs != None), "Need parameter 'SaabArgs'!"
            assert (shrinkArgs != None), "Need parameter 'shrinkArgs'!"
            self.depth = (int)(depth)
            self.shrinkArgs = shrinkArgs
            self.SaabArgs = SaabArgs
            self.par = {}
            self.bias = {}
            self.TH1 = TH1
            self.TH2 = TH2
            self.Energy = {}
        
            if depth > np.min([len(SaabArgs), len(shrinkArgs)]):
                self.depth = np.min([len(SaabArgs), len(shrinkArgs)])
                print("       <WARNING> Too few 'SaabArgs/shrinkArgs' to get depth %s, actual depth: %s"%(str(depth),str(self.depth)))

    @gc_invoker
    def SaabTransform(self, X, saab, layer, train=False):
        '''
        Get saab features. 
        If train==True, remove leaf nodes using TH1, only leave the intermediate node's response
        '''
        shrinkArg, SaabArg = self.shrinkArgs[layer], self.SaabArgs[layer]
        assert ('func' in shrinkArg.keys()), "shrinkArg must contain key 'func'!"
        X = shrinkArg['func'](X, shrinkArg)
        S = list(X.shape)
        X = X.reshape(-1, S[-1])
        
        if SaabArg['num_AC_kernels'] != -1:
            S[-1] = SaabArg['num_AC_kernels']
            
        transformed = saab.transform(X)
        transformed = transformed.reshape(S[0],S[1],S[2],-1)
        
        # while training only non-leaf node can go next layer
        # while inference, each layer contains both intermediate and leaf nodes
        if train==True and self.SaabArgs[layer]['cw'] == True: # remove leaf nodes
            transformed = transformed[:, :, :, saab.Energy>self.TH1]
            
        return transformed
    
    @gc_invoker
    def invSaabTransform(self, X, saab, layer):
        '''
        inverse Saab transform, only at inference period for backward
        '''
        shrinkArg, SaabArg = self.shrinkArgs[layer], self.SaabArgs[layer]
        assert ('func' in shrinkArg.keys()), "shrinkArg must contain key 'func'!"
        
        S = list(X.shape)
        X = X.reshape(-1, S[-1])

        transformed = saab.invTransform(X)
        transformed = transformed.reshape(S[0],S[1],S[2],-1)

        # not consider for "SaabArg['num_AC_kernels'] != -1"

        transformed = shrinkArg['invfunc'](transformed, shrinkArg)
        
        return transformed
    
    @gc_invoker
    def SaabFit(self, X, layer, bias=0):
        '''Learn a saab model'''
        shrinkArg, SaabArg = self.shrinkArgs[layer], self.SaabArgs[layer]
        assert ('func' in shrinkArg.keys()), "shrinkArg must contain key 'func'!"
        X = shrinkArg['func'](X, shrinkArg)
        S = list(X.shape)
        X = X.reshape(-1, S[-1])
        saab = Saab(num_kernels=SaabArg['num_AC_kernels'], needBias=SaabArg['needBias'], bias=bias)
        saab.fit(X)
        return saab

    @gc_invoker
    def discard_nodes(self, saab):
        '''Remove discarded nodes (<TH2) from the model'''
        energy_k = saab.Energy
        discard_idx = np.argwhere(energy_k<self.TH2)
        saab.Kernels = np.delete(saab.Kernels, discard_idx, axis=0) 
        saab.Energy = np.delete(saab.Energy, discard_idx)
        saab.num_kernels -= discard_idx.size
        return saab

    @gc_invoker
    def cwSaab_1_layer(self, X, train, bias=None):
        '''cwsaab/saab transform starting for Hop-1'''
        if train == True:
            saab_cur = []
            bias_cur = []
        else:
            saab_cur = self.par['Layer'+str(0)]
            bias_cur = self.bias['Layer'+str(0)]
        transformed, eng = [], []

        if self.SaabArgs[0]['cw'] == True: # channel-wise saab
            S = list(X.shape)
            S[-1] = 1
            X = np.moveaxis(X, -1, 0)
            for i in range(X.shape[0]):
                X_tmp = X[i].reshape(S)
                if train == True:
                    # fit
                    saab = self.SaabFit(X_tmp, layer=0)
                    # remove discarded nodes
                    saab = self.discard_nodes(saab)
                    # store
                    saab_cur.append(saab)
                    bias_cur.append(saab.Bias_current)
                    eng.append(saab.Energy)
                    # transformed feature
                    transformed.append(self.SaabTransform(X_tmp, saab=saab, layer=0, train=True))
                else:
                    if len(saab_cur) == i:
                        break
                    transformed.append(self.SaabTransform(X_tmp, saab=saab_cur[i], layer=0))
            transformed = np.concatenate(transformed, axis=-1)
        else: # saab, not channel-wise
            if train == True:
                saab = self.SaabFit(X, layer=0)
                saab = self.discard_nodes(saab)
                saab_cur.append(saab)
                bias_cur.append(saab.Bias_current)
                eng.append(saab.Energy)
                transformed = self.SaabTransform(X, saab=saab, layer=0, train=True)
            else:
                transformed = self.SaabTransform(X, saab=saab_cur[0], layer=0)
                
        if train == True:
            self.par['Layer0'] = saab_cur
            self.bias['Layer'+str(0)] = bias_cur
            self.Energy['Layer0'] = eng
                
        return transformed
    
    @gc_invoker
    def inv_cwSaab_1_layer(self, X, bias=None):
        '''inverse cwsaab/saab transform starting for Hop-1'''
        saab_cur = self.par['Layer'+str(0)]
        bias_cur = self.bias['Layer'+str(0)]
        recovered = []

        if self.SaabArgs[0]['cw'] == True: # channel-wise saab
            S = list(X.shape)
            S[-1] = 1
            X = np.moveaxis(X, -1, 0)
            for i in range(X.shape[0]):
                X_tmp = X[i].reshape(S)
                if len(saab_cur) == i:
                    break
                recovered.append(self.invSaabTransform(X_tmp, saab=saab_cur[i], layer=0))
            recovered = np.concatenate(recovered, axis=-1)
        else: # saab, not channel-wise
            recovered = self.invSaabTransform(X, saab=saab_cur[0], layer=0)
                
        return recovered

    @gc_invoker
    def cwSaab_n_layer(self, X, train, layer):
        '''cwsaab/saab transform starting from Hop-2'''
        output, eng_cur, ct, pidx = [], [], -1, 0
        S = list(X.shape)
        saab_prev = self.par['Layer'+str(layer-1)]
        bias_prev = self.bias['Layer'+str(layer-1)]

        if train == True:
            saab_cur = []
            bias_cur = []
        else:
            saab_cur = self.par['Layer'+str(layer)]
        
        if self.SaabArgs[layer]['cw'] == True: # channel-wise saab
            S[-1] = 1
            X = np.moveaxis(X, -1, 0)
            for i in range(len(saab_prev)):
                for j in range(saab_prev[i].Energy.shape[0]):
                    if train==False:
                        ct += 1 # helping index
                    if saab_prev[i].Energy[j] < self.TH1: # this is a leaf node
                        continue
                    else: # this is an intermediate node
                        if train==True:
                            ct += 1
                        
                    self.split = True
                    X_tmp = X[ct].reshape(S)
                    
                    if train == True:
                        # fit
                        saab = self.SaabFit(X_tmp, layer=layer, bias=bias_prev[i])
                        # children node's energy should be multiplied by the parent's energy
                        saab.Energy *= saab_prev[i].Energy[j]
                        # remove the discarded nodes
                        saab = self.discard_nodes(saab)
                        # store
                        saab_cur.append(saab)
                        bias_cur.append(saab.Bias_current)
                        eng_cur.append(saab.Energy) 
                        # get output features
                        out_tmp = self.SaabTransform(X_tmp, saab=saab, layer=layer, train=True)
                    else:
                        out_tmp = self.SaabTransform(X_tmp, saab=saab_cur[pidx], layer=layer)
                        pidx += 1 # helping index
                        
                    output.append(out_tmp)
                    
                    # Clean the Cache
                    X_tmp = None
                    gc.collect()
                    out_tmp = None
                    gc.collect()
            
            if len(output) == 0:
                print("largest energy is", saab_prev[0].Energy[0])
            output = np.concatenate(output, axis=-1)
                    
        else: # saab, not channel-wise
            if train == True:
                # fit
                saab = self.SaabFit(X, layer=layer, bias=bias_prev[0])
                # remove the discarded nodes
                saab = self.discard_nodes(saab)
                # store
                saab_cur.append(saab)
                bias_cur.append(saab.Bias_current)
                eng_cur.append(saab.Energy)
                # get output features
                output = self.SaabTransform(X, saab=saab, layer=layer, train=True)
            else:
                output = self.SaabTransform(X, saab=saab_cur[0], layer=layer)

        if train == True:   
            if self.split == True or self.SaabArgs[0]['cw'] == False:
                self.par['Layer'+str(layer)] = saab_cur
                self.bias['Layer'+str(layer)] = bias_cur
                self.Energy['Layer'+str(layer)] = eng_cur
        
        return output
    
    @gc_invoker
    def inv_cwSaab_n_layer(self, X, layer):
        '''cwsaab/saab transform starting from Hop-2'''
        output, eng_cur, ct, pidx = [], [], -1, 0
        S = list(X.shape)
        saab_prev = self.par['Layer'+str(layer-1)]
        bias_prev = self.bias['Layer'+str(layer-1)]

        saab_cur = self.par['Layer'+str(layer)]
        
        if self.SaabArgs[layer]['cw'] == True: # channel-wise saab
            X = np.moveaxis(X, -1, 0)
            # print(len(saab_cur))
            for i in range(len(saab_cur)):
                c = saab_cur[i].Energy.shape[0]
                # print("-", c)
                
                X_tmp = np.moveaxis(X[i*c:i*c+c], 0, -1)
                
                out_tmp = self.invSaabTransform(X_tmp, saab=saab_cur[i], layer=layer)
                    
                output.append(out_tmp)
                
                # Clean the Cache
                X_tmp = None
                gc.collect()
                out_tmp = None
                gc.collect()
                    
            output = np.concatenate(output, axis=-1)
                    
        else: # saab, not channel-wise
            output = self.invSaabTransform(X, saab=saab_cur[0], layer=layer)

        return output
    
    def fit(self, X):
        print("layer 0 input size", X.shape)
        '''train and learn cwsaab/saab kernels'''
        X = self.cwSaab_1_layer(X, train=True)
        self.traindata_transformed.append(X)
        print('=' * 45 + '>c/w Saab Train Hop 1')
        for i in range(1, self.depth):
            print(f"layer {i} input size", X.shape)
            X = self.cwSaab_n_layer(X, train = True, layer = i)
            self.traindata_transformed.append(X)
            if (self.split == False and self.SaabArgs[i]['cw'] == True):
                self.depth = i
                print("       <WARNING> Cannot futher split, actual depth: %s"%str(i))
                break
            print('=' * 45 + f'>c/w Saab Train Hop {i+1}')
            self.split = False
        self.trained = True

    def transform(self, X, styletransfer=False, transferlayer=[0,1,2], args={}):
        '''
        Get feature for all the Hops
        Parameters
        ----------
        X: Input image (N, H, W, C), C=1 for grayscale, C=3 for color image
        Returns
        -------
        output: a list of transformed feature maps
        '''
        assert (self.trained == True), "Must call fit first!"
        output = []

        # b, h, w, c = X.shape
        # transferred = copy.deepcopy(X)
        # transferred[0] = StyleSwap(transferred[0][None,...], transferred[1][None,...], self.shrinkArgs[0]['func'], self.shrinkArgs[0]['invfunc'], 7, 1)
        # transferred[0] = (transferred[0] - transferred[0].min()) / (transferred[0].max() - transferred[0].min())
        # import matplotlib.pyplot as plt
        # plt.imsave("oriSwap.png", transferred[0])

        X = self.cwSaab_1_layer(X, train = False)
        if styletransfer and (0 in transferlayer):
            transferred = copy.deepcopy(X)
            transferred, _ = StyleTransfer(transferred, X, args, X.shape, self.shrinkArgs[0]['func'], self.shrinkArgs[0]['invfunc'])
            output.append(transferred)
            self.transformed.append(transferred)
        else:
            output.append(X)
            self.transformed.append(X)
        
        for i in range(1, self.depth):
            X = self.cwSaab_n_layer(X, train=False, layer=i)
            if styletransfer and (i in transferlayer):
                transferred = copy.deepcopy(X)
                transferred, OneHot = StyleTransfer(transferred, X, args, X.shape, self.shrinkArgs[0]['func'], self.shrinkArgs[0]['invfunc'])
                output.append(transferred)
                self.transformed.append(transferred)
            else:
                output.append(X)
                self.transformed.append(X)

        # # copy high freq maps in higher layers (only little improvement on overall color)
        # if args['transmethod'] == 'StyleSwap':
        #     print("start to recover high freq")
        #     assert OneHot.shape[0] == 1
        #     indices = np.argmax(OneHot, axis=-1)[0].ravel()

        #     s = 1
        #     win = args['SwapWin']
        #     for i in range(self.depth-2, -1, -1):
        #         print("now at", i)
        #         feats = self.transformed[i]
        #         s *= 2
        #         win *= 2
        #         shrinkArg = {'invfunc': self.shrinkArgs[0]['invfunc'], 'func': self.shrinkArgs[0]['func'], 'win':win, 'stride': s, 'pool': 1, 'pad': 0}
        #         patch_style = shrinkArg['func'](feats[1][None,...], shrinkArg)
        #         b, h, w, c = patch_style.shape                
        #         assert b == 1
        #         patch_style = patch_style.reshape(-1, c)
        #         patch_style = patch_style[indices]
        #         patch_style = patch_style.reshape(b, h, w, c)
        #         patch_style = shrinkArg['invfunc'](patch_style, shrinkArg)
        #         feats[0] = (feats[0] + patch_style) / 2
        #         # feats[0] = patch_style
        #         self.transformed[i] = feats
        #         output[i] = feats
            
        return output
    
    def invTransform(self, X):
        '''
        Inverse features for all the Hops
        Parameters
        ----------
        X: last layer features (N, H, W, C)
        Returns
        -------
        output: a list of recoverd feature maps
        '''
        assert (self.trained == True), "Must call fit first!"
        output = []

        # print(f"inverse feature input", X.shape)
        
        for i in range(self.depth-1, 0, -1):
            if i != self.depth-1:
                tmp_X = copy.deepcopy(self.transformed[i])
                saab_cur = self.par['Layer'+str(i)]
                indices = []
                for j in range(len(saab_cur)):
                    idx = saab_cur[j].Energy>self.TH1
                    # print(idx)
                    indices.append(idx)
                indices = np.concatenate(indices)
                # print(indices)
                tmp_X[..., indices] = X
                output.append(tmp_X)
                # print("inverse feature combined", tmp_X.shape)
                X = self.inv_cwSaab_n_layer(tmp_X, layer=i)
            else:
                X = self.inv_cwSaab_n_layer(X, layer=i)
            # print(f"inverse feature at layer {i}", X.shape)

        tmp_X = copy.deepcopy(self.transformed[0])
        saab_cur = self.par['Layer'+str(0)]
        indices = []
        for j in range(len(saab_cur)):
            idx = saab_cur[j].Energy>self.TH1
            # print(idx)
            indices.append(idx)
        indices = np.concatenate(indices)
        # print(indices)
        # tmp_X[0,...] = tmp_X[1,...] # only for keeping style maps in the first layer
        tmp_X[..., indices] = X
        output.append(tmp_X)
        # print("inverse feature combined", tmp_X.shape)
        X = self.inv_cwSaab_1_layer(tmp_X)
        output.append(X)
            
        return output
    
    def transform_singleHop(self, X, layer=0):
        '''
        Get feature for a single Hop

        Parameters
        ----------
        X: previous Hops output (N, H1, W1, C1)
        layer: Hop index (start with 0)
        
        Returns
        -------
        output: transformed feature maps (N, H2, W2, C2)
        '''
        assert (self.trained == True), "Must call fit first!"
        if layer==0:
            output = self.cwSaab_1_layer(X, train = False)
        else:
            output = self.cwSaab_n_layer(X, train=False, layer=layer)
            
        return output
    
# if __name__ == "__main__":
#     import warnings
#     warnings.filterwarnings("ignore")
#     from sklearn import datasets
#     from skimage.util import view_as_windows
    
#     # example callback function for collecting patches and its inverse
#     def Shrink(X, shrinkArg):
#         win = shrinkArg['win']
#         stride = shrinkArg['stride']
#         ch = X.shape[-1]
#         X = view_as_windows(X, (1,win,win,ch), (1,stride,stride,ch))
#         return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)

#     # read data
#     print(" > This is a test example: ")
#     digits = datasets.load_digits()
#     X = digits.images.reshape((len(digits.images), 8, 8, 1))
#     print(" input feature shape: %s"%str(X.shape))

#     # set args
#     SaabArgs = [{'num_AC_kernels':-1, 'needBias':False, 'cw': False},
#                 {'num_AC_kernels':-1, 'needBias':True, 'cw':True}] 
#     shrinkArgs = [{'func':Shrink, 'win':2, 'stride': 2}, 
#                 {'func': Shrink, 'win':2, 'stride': 2}]

#     print(" -----> depth=2")
#     cwsaab = cwSaab(depth=2, TH1=0.001,TH2=0.0005, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs)
#     cwsaab.fit(X)
#     output1 = cwsaab.transform(X)
#     output2 = cwsaab.transform_singleHop(X)
