# %%
import tracemalloc

import time
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from tensorflow.keras.datasets import mnist, fashion_mnist
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

def loadMNIST():
    # Load the MNIST dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Flatten the images
    X_train = X_train[:10,:,:,None]
    X_test = X_test[:10,:,:,None]

    # Normalize the data
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    return X_train, y_train[:10], X_test, y_test[:10]

def loadFashionMNIST():
    # Load the FashionMNIST dataset
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    # Flatten the images
    X_train = X_train[:10,:,:,None]
    X_test = X_test[:10,:,:,None]

    # Normalize the data
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    return X_train, y_train[:10], X_test, y_test[:10]

def trainXGBoost(data, max_depth=6, n_estimators=100, learning_rate=0.1, random_state=42, verbose=False):
    X_train, y_train, X_test, y_test = data

    model = XGBRegressor(
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        n_jobs=-1,  # Use all available CPU cores
        random_state=random_state,
        colsample_bytree=1.0,
        subsample=0.8,
        min_child_weight=5,
        gamma=5
    )

    XGB_runtime = {}
    # Train the model
    st = time.time()
    model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=verbose)
    ed = time.time()
    print("XGBoost training time is :", (ed-st), "s")
    XGB_runtime['XGBtrain'] = ed-st

    # Make predictions
    st = time.time()
    y_pred_train = model.predict(X_train)
    ed = time.time()
    print("XGBoost inference time on train data is :", (ed-st), "s")
    XGB_runtime['XGBinferOnTrain'] = ed-st

    st = time.time()
    y_pred = model.predict(X_test)
    ed = time.time()
    print("XGBoost inference time on test data is :", (ed-st), "s")
    XGB_runtime['XGBinferOnTest'] = ed-st

    return y_pred, y_pred_train, XGB_runtime

import os
from skimage.util import view_as_windows
from pixelhop import Pixelhop
from skimage.measure import block_reduce
import warnings, gc
from PIL import Image

def Shrink(X, shrinkArg):
    # print("shrink input", X.shape)
    #---- max pooling----
    pool = shrinkArg['pool']
    X = block_reduce(X, (1,pool,pool,1), np.max)

    #---- neighborhood construction
    win = shrinkArg['win']
    stride = shrinkArg['stride']
    pad = shrinkArg['pad']
    # zero padding
    X = np.pad(X, ((0,0),(pad,pad),(pad,pad),(0,0)), mode='constant')
    # split patches
    c = X.shape[-1]
    X = view_as_windows(X, (1, win, win, c), step=(1, stride, stride, c))
    # print(X.shape)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], c*win*win)
    # print("shrink output", X.shape)
    return X

def invShrink(X, shrinkArg):
    # print("inverse shrink input", X.shape)

    # reconstruct the original shape
    win = shrinkArg['win']
    c = X.shape[-1] // (win * win)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], win, win, c)

    # reverse neighborhood construction
    stride = shrinkArg['stride']
    pad = shrinkArg['pad']
    h, w = (X.shape[1]-1) * stride + win - 2 * pad, (X.shape[2]-1) * stride + win - 2 * pad
    # print(h, w)
    output = np.zeros((X.shape[0], h+2*pad, w+2*pad, c))
    count = np.zeros((X.shape[0], h+2*pad, w+2*pad, c))
    for i in range(X.shape[1]):
        for j in range(X.shape[2]):
            output[:, i*stride:i*stride+win, j*stride:j*stride+win, :] += X[:, i, j, :, :, :]
            count[:, i*stride:i*stride+win, j*stride:j*stride+win, :] += 1
    output /= count

    # remove padding
    if pad > 0:
        output = output[:, pad:-pad, pad:-pad, :]

    # reverse max pooling (no effect if pool=1)
    pool = shrinkArg['pool']
    output = np.repeat(np.repeat(output, pool, axis=1), pool, axis=2)

    # print("inverse shrink output", output.shape)
    return output

# example callback function for how to concate features from different hops
def Concat(X, concatArg):
    return X

def invConcat(X, concatArg):
    return X

def saveFeats(out):
    base_dir = "hop_output"

    # create the base directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    for layidx, layer in enumerate(out):
        print("layer", layidx+1)
        # create a directory for the current layer
        layer_dir = os.path.join(base_dir, f"layer_{layidx+1}")
        if not os.path.exists(layer_dir):
            os.makedirs(layer_dir)
        # loop over each tensor in the output
        for batch_idx, tensor in enumerate(layer, 1):
            # get the number of channels in the current tensor
            num_channels = tensor.shape[2]
            print("batch", batch_idx, "has num_channels", num_channels)
            
            # create a directory for the current batch
            batch_dir = os.path.join(layer_dir, f"batch_{batch_idx}")
            if not os.path.exists(batch_dir):
                os.makedirs(batch_dir)
            
            # loop over each channel in the current tensor
            for channel_idx in range(num_channels):
                # extract the features of the current channel
                channel_features = tensor[:, :, channel_idx]

                # normalize channel_features to range 0-1 for display purpose
                channel_features = (channel_features - channel_features.min()) / (channel_features.max() - channel_features.min())

                # save the channel features to a png file
                file_path = os.path.join(batch_dir, f"features_{channel_idx}.png")
                plt.imsave(file_path, channel_features, cmap='gray')

def acquire_image(img_path, style_img_path):
    # Open the image
    img = Image.open(img_path)
    style = Image.open(style_img_path)

    # # Calculate new size preserving aspect ratio
    w, h = img.size
    img = img.resize((w, h))
    w, h = img.size
    # if w > h:
    #     new_h = img_size
    #     new_w = int(img_size * w / h)
    # else:
    #     new_w = img_size
    #     new_h = int(img_size * h / w)

    # # Resize the image
    # img = img.resize((768, 512))
    style = style.resize((w,h))

    # Convert the image to a NumPy array
    img = np.array(img)
    if len(img.shape) == 2:
        img = img[...,None]
    if img.shape[-1] == 4:
        img = img[...,:3]
    style = np.array(style)
    if len(style.shape) == 2:
        style = style[...,None]
    if style.shape[-1] == 4:
        style = style[...,:3]

    # # Swap color channels from RGB to BGR
    # img = img[:, :, ::-1]

    # Convert the image to float32 and scale it to [0, 1]
    img = img.astype(np.float32) / 255
    style = style.astype(np.float32) / 255

    # # Subtract ImageNet mean
    # mean = np.array([0.40760392, 0.45795686, 0.48501961])
    # img = img - mean

    # # Multiply by 255 to unnormalize the pixel values
    # img = img * 255

    return np.concatenate([img[None,...], style[None,...]], axis=0)

def predSizeAndCheck(shape, args):
    """Return True if inversable else return False
    """
    b, h, w, c = shape
    print("[prediction] layer 0 input", "h", h, "w", w)
    for idx, arg in enumerate(args):
        win, stride, pool, pad = arg['win'], arg['stride'], arg['pool'], arg['pad']
        if pool > 1:
            return False
        h = (h + 2 * pad - win) / stride + 1
        w = (w + 2 * pad - win) / stride + 1
        print(f"[prediction] layer {idx} output", "h", h, "w", w)
        if h % 1 or w % 1:
            return False
    return True

def main(
    TH1=0.005, 
    TH2=0.001, 
    cw=True, 
    savePNG=False,
    saveModel=False,
    pretrained=None,
    stylize=False,
    recoverName="",
    cimg="",
    simg="",
    transferArgs={}
):
    np.random.seed(2023)
    warnings.filterwarnings("ignore")

    tracemalloc.start()

    # X_train, y_train, X_test, y_test = dataloader()
    # print(f"X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")
    img = acquire_image(cimg, simg)

    # -----------Module 1: set PixelHop parameters-----------
    SaabArgs = [
        {'num_AC_kernels':-1, 'needBias':False, 'cw': False},
        {'num_AC_kernels':-1, 'needBias':True, 'cw': cw},
        {'num_AC_kernels':-1, 'needBias':True, 'cw': cw},
        # {'num_AC_kernels':-1, 'needBias':True, 'cw': cw},
        # {'num_AC_kernels':-1, 'needBias':True, 'cw': cw}
        ] 
    shrinkArgs = [
        {'invfunc': invShrink, 'func': Shrink, 'win':3, 'stride': 1, 'pool': 1, 'pad': 1}, 
        {'invfunc': invShrink, 'func': Shrink, 'win':2, 'stride': 2, 'pool': 1, 'pad': 0},
        {'invfunc': invShrink, 'func': Shrink, 'win':2, 'stride': 2, 'pool': 1, 'pad': 0},
        # {'invfunc': invShrink, 'func': Shrink, 'win':2, 'stride': 2, 'pool': 1, 'pad': 0},
        # {'invfunc': invShrink, 'func': Shrink, 'win':2, 'stride': 2, 'pool': 1, 'pad': 0},
        ]
    concatArg = {'func':Concat, 'invfunc':invConcat}

    assert predSizeAndCheck(img.shape, shrinkArgs), "Not inversable!"


    # -----------Module 1: Train PixelHop -----------
    hops = Pixelhop(depth=len(SaabArgs), TH1=TH1, TH2=TH2, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs, concatArg=concatArg)

    runtime = {}

    if pretrained:
        print(f"loading pretrained from: {pretrained}")
        hops.load(pretrained)
    else:
        # record start time
        st = time.time()
        hops.fit(img)
        ed = time.time()
        print("Hop training time is :", (ed-st), "s")
        runtime['HopTrain'] = ed-st

    # for key in hops.par:
    #     print(key)
    #     for saab in hops.par[key]:
    #         print(saab.Energy)
    #     print()

    # --------- Module 2: get only Hop 3 feature for both training set and testing set -----------
    st = time.time()
    train_outputs = hops.transform(
        img, 
        styletransfer=stylize, 
        transferlayer=[2], 
        args=transferArgs
    )
    ed = time.time()
    print("Hop inference time on train data is :", (ed-st), "s")
    runtime['HopInferOnTrain'] = ed-st

    for each in train_outputs:
        print(each.shape)

    # inverse feature maps
    st = time.time()
    recover_outputs = hops.invTransform(train_outputs[-1])
    ed = time.time()
    print("Hop inverse time on train data is :", (ed-st), "s")
    runtime['HopInverseOnTrain'] = ed-st

    for each in recover_outputs[::-1]:
        print(each.shape)

    for idx, each in enumerate(recover_outputs[::-1]):
        if idx == 0:
            print("[MSE LOSS] input", np.mean((img - each)**2))
        else:
            print(f"[MSE LOSS] layer{idx}", np.mean((train_outputs[idx-1] - each)**2))
    
    recovered = recover_outputs[-1]
    recovered[0] = (recovered[0] - recovered[0].min()) / (recovered[0].max() - recovered[0].min())
    # recovered[1] = (recovered[1] - recovered[1].min()) / (recovered[1].max() - recovered[1].min())
    if recovered[0].shape[-1] == 1:
        Image.fromarray((recovered[0,...,0]*255).astype(np.uint8)).save(f"{recoverName}")
    else:
        plt.imsave(f"{recoverName}", recovered[0])
    # plt.imsave(f"{recoverName}_2.png", recovered[1])

    # st = time.time()
    # test_outputs = hops.transform(X_test)
    # ed = time.time()
    # print("Hop inference time on test data is :", (ed-st), "s")
    # runtime['HopInferOnTest'] = ed-st

    # for i in range(len(train_outputs)):
    #     print(f"Layer {i} dimension: {train_outputs[i].shape}")

    # # calculate model size
    # params_num = 0
    # if cw:
    #     for i in range(len(train_outputs)):
    #         bias_num = train_outputs[i-1].shape[-1] if i > 0 else 0
    #         params_num += 3 * 3 * train_outputs[i].shape[-1] + bias_num
    # else:
    #     for i in range(len(train_outputs)):
    #         last_ch = train_outputs[i-1].shape[-1] if i > 0 else 1
    #         params_num += last_ch * 3 * 3 * train_outputs[i].shape[-1] + (1 if i > 0 else 0)
    # print(f"Model size: {params_num}")

    # for each in hops.traindata_transformed:
    #     print(each.shape)

    if savePNG:
        saveFeats(train_outputs)
        # saveFeats(recover_outputs)
        
    if saveModel and not pretrained:
        try:
            print("saving model.")
            cname = cimg.split('/')[-1].split('.')[0]
            sname = simg.split('/')[-1].split('.')[0]
            hops.save(f"./models/{TH1}_{TH2}_{cw}_{cname}_{sname}")
        except:
            print("save model fail")

    # train_hop3_feats = train_outputs[-1]
    # test_hop3_feats = test_outputs[-1]
    
    # # --------- Module 2: standardization
    # STD = np.std(train_hop3_feats, axis=0, keepdims=1)
    # train_hop3_feats = train_hop3_feats/STD
    # test_hop3_feats = test_hop3_feats/STD

    # train_hop3_feats = train_hop3_feats.reshape(train_hop3_feats.shape[0], -1)
    # test_hop3_feats = test_hop3_feats.reshape(test_hop3_feats.shape[0], -1)

    # print(f"Flatten dimension: {train_hop3_feats.shape}")
    
    # displaying the memory
    print(tracemalloc.get_traced_memory())
    # stopping the library
    tracemalloc.stop()

    print("Complete!")

    return runtime, (hops, train_outputs)

# %%
if __name__ == '__main__':
    contents = os.listdir('./images/contents')
    styles = os.listdir('./images/styles')
    c = 'Tuebingen_Neckarfront.png'
    s = 'vangogh.png'
    save_path = f"./{c}_{s}.png"
    runtime, miscel = main(
        TH1=0.005, # for leaf node
        TH2=0, # for discard node
        cw=True, 
        savePNG=False,
        saveModel=False,
        # pretrained='./models/0.005_0_True_Tuebingen_Neckarfront_vangogh',
        pretrained=None,
        stylize=True,
        recoverName=save_path,
        cimg=f"./images/contents/{c}",
        simg=f'./images/styles/{s}',
        transferArgs = {
            'transmethod': "StyleSwap", 
            'SwapWin': 5 # only for styleSwap
        }
    )
    
    # contents = os.listdir('./images/contents')
    # styles = os.listdir('./images/styles')
    # for c in contents:
    #     for s in styles:
    #         for w in [1, 3, 5, 7, 11]:
    #             save_path = f"./fullSwap/{c}_{s}_win{w}.png"
    #             if os.path.isfile(save_path):
    #                 print(f"skipping {save_path}")
    #                 continue
    #             model_path = f"./models/{0.005}_{0}_{True}_{c.split('.')[0]}_{s.split('.')[0]}"
    #             saveModel = True
    #             pretrained = None
    #             if os.path.isfile(model_path + '.pkl'):
    #                 saveModel = False
    #                 pretrained = model_path
    #                 print("using pretrained model")
    #             runtime, miscel = main(
    #                 TH1=0.005, # for leaf node
    #                 TH2=0, # for discard node
    #                 cw=True, 
    #                 savePNG=False,
    #                 saveModel=saveModel,
    #                 # pretrained='./models/0.005_0_True_both',
    #                 pretrained=pretrained,
    #                 stylize=True,
    #                 recoverName=save_path,
    #                 cimg=f"./images/contents/{c}",
    #                 simg=f'./images/styles/{s}',
    #                 transferArgs = {
    #                     'transmethod': "StyleSwap", 
    #                     'SwapWin': w # only for styleSwap
    #                 }
    #             )
    #             del runtime
    #             del miscel
    #             gc.collect()
    #             time.sleep(0.5)


    # print(runtime)
    # model, trfeat = miscel
    # trfeat.shape

    # """
    # zip hop_output.zip -r hop_output/
    # rm hop_output -rf
    # rm hop_output.zip
    # """
# %%
def direct_stylize_cw():
    img = acquire_image('./images/house.jpg', 512)

    import copy
    b, h, w, c = img.shape
    transferred = copy.deepcopy(img)
    for idx in range(c):
        transferred[0,...,idx] = EFDM(img[0,...,idx].reshape(-1), img[1,...,idx].reshape(-1)).reshape((h, w))
    plt.imsave('./direct_stylized.png', transferred[0])

def direct_stylize():
    img = acquire_image('./images/house.jpg', 512)

    import copy
    b, h, w, c = img.shape
    transferred = copy.deepcopy(img)
    
    transferred[0] = EFDM(img[0].reshape(-1), img[1].reshape(-1)).reshape((h, w, c))
    plt.imsave('./direct_stylized.png', transferred[0])

# not working
def pred_latent():
    # example of automatically choosing the loss function for regression
    from sklearn.datasets import make_regression
    from xgboost import XGBRegressor
    from PIL import Image
    import numpy as np

    out = miscel[1]

    # define dataset
    C = np.moveaxis(out[-1][0].reshape(-1, 432), -1, 0)
    S = np.moveaxis(out[-1][1].reshape(-1, 432), -1, 0)
    C /= np.max(C)
    S /= np.max(S)
    y = np.random.normal(size=(C.shape))
    X = np.random.normal(size=(C.shape))
    print(X.shape, y.shape)

    def custom_objective(y_true, y_pred):
        print(y_true.shape, y_pred.shape)
        
        closs = (y_pred - C)**2 / 2
        # sloss = np.mean(y_pred.reshape(-1,1)*y_pred + S.reshape(-1,1)*S + y_pred.reshape(-1,1)*S)
        sloss = np.mean((y_pred@(y_pred.T) - S@(S.T))**2)
        grad = closs + sloss
        hess = np.ones_like(y_pred)*y_pred
        print(grad.shape, hess.shape)
        return grad.reshape(-1), hess.reshape(-1)

    # define the model
    model = XGBRegressor(objective=custom_objective)
    # fit the model
    model.fit(X, y, verbose=True)
    # summarize the model loss function
    print(model.objective)

    pred = model.predict(X)
    print(pred.shape)
    import matplotlib.pyplot as plt
    for idx, ch in enumerate(pred):
        ch = (ch - ch.min()) / (ch.max() - ch.min())
        plt.imsave(f'./xgb/{idx}.png', ch.reshape((96,128)))

def energyplot(c, s):
    hop = miscel[0]
    plt.plot(np.cumsum(hop.Energy['Layer0'][0]))
    plt.savefig(f"./enerygyPlot/{c}_{s}.png")

def showAllEnergy():
    for each in os.listdir('./models'):
        hops = Pixelhop(depth=None, TH1=None, TH2=None, SaabArgs=None, shrinkArgs=None, concatArg=None, load=True)
        hops.load(f'./models/{each[:-4]}')
        print(hops.Energy['Layer0'][0][:3])