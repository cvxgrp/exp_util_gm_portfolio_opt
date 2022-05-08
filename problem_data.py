import numpy as np

pi = np.array([0.66692381, 0.00762527, 0.13608766, 0.18936325])

k = len(pi)

mus = np.array([[ 2.34811066e-03,  1.06886284e-02,  1.06246963e-02,
         1.15581444e-02,  1.02078508e-02,  1.00976546e-02,
         9.90576551e-03,  9.53753858e-03,  3.37888168e-03,
         2.94725976e-03,  2.95533696e-03],
       [ 3.67571429e-04,  4.63470286e-01,  4.77257243e-01,
         5.67999857e-01,  2.27814857e-01,  2.57110641e-01,
         2.27951429e-01,  6.16867143e-01,  2.10714286e-02,
         6.31114285e-03,  6.73142856e-03],
       [ 4.31843309e-03,  4.47558929e-02,  5.05010681e-02,
         5.74217244e-02,  3.79041580e-02,  3.88409956e-02,
         3.68089790e-02,  5.49110559e-02,  1.43029093e-02,
         1.20242985e-02,  1.48667003e-02],
       [ 2.55887557e-03, -3.24722751e-02, -3.40766541e-02,
        -3.39945943e-02, -2.54479021e-02, -2.68715024e-02,
        -2.59494883e-02, -3.48931294e-02,  3.53694267e-03,
         2.78885118e-03,  4.08751271e-03]])

n = len(mus[0])

Sigmas = np.array([[[ 5.14068349e-06, -2.62394876e-06, -2.37152292e-06,
         -1.42107482e-06, -2.86107086e-06, -2.58078883e-06,
         -2.15168161e-06, -1.78075918e-06,  8.50299457e-07,
          3.12051279e-06,  1.56454324e-06],
        [-2.62394876e-06,  1.73686463e-03,  1.62498288e-03,
          1.72749346e-03,  1.22534366e-03,  1.27401514e-03,
          1.19631176e-03,  1.72207956e-03,  6.86793781e-05,
         -1.66609837e-05,  8.90626531e-07],
        [-2.37152292e-06,  1.62498288e-03,  2.05176368e-03,
          2.06451605e-03,  1.31835375e-03,  1.33094976e-03,
          1.19896362e-03,  2.21392125e-03,  4.44715007e-05,
         -2.27528194e-05, -2.27934485e-05],
        [-1.42107482e-06,  1.72749346e-03,  2.06451605e-03,
          2.20418592e-03,  1.27694710e-03,  1.32603451e-03,
          1.19357645e-03,  2.25037307e-03,  4.63334370e-05,
         -2.34103585e-05, -1.88578720e-05],
        [-2.86107086e-06,  1.22534366e-03,  1.31835375e-03,
          1.27694710e-03,  1.28906085e-03,  1.19777752e-03,
          1.16522603e-03,  1.34673641e-03,  8.07413058e-05,
          2.24822672e-07,  3.09380200e-05],
        [-2.58078883e-06,  1.27401514e-03,  1.33094976e-03,
          1.32603451e-03,  1.19777752e-03,  1.16167728e-03,
          1.11703757e-03,  1.37264407e-03,  7.79045610e-05,
         -8.33723153e-07,  2.50299338e-05],
        [-2.15168161e-06,  1.19631176e-03,  1.19896362e-03,
          1.19357645e-03,  1.16522603e-03,  1.11703757e-03,
          1.09966194e-03,  1.22680647e-03,  7.49485365e-05,
         -9.74591132e-07,  2.43572649e-05],
        [-1.78075918e-06,  1.72207956e-03,  2.21392125e-03,
          2.25037307e-03,  1.34673641e-03,  1.37264407e-03,
          1.22680647e-03,  2.49931076e-03,  3.97825152e-05,
         -2.92367024e-05, -3.12872220e-05],
        [ 8.50299457e-07,  6.86793781e-05,  4.44715007e-05,
          4.63334370e-05,  8.07413058e-05,  7.79045610e-05,
          7.49485365e-05,  3.97825152e-05,  2.04723233e-04,
          8.99867761e-05,  2.06230808e-04],
        [ 3.12051279e-06, -1.66609837e-05, -2.27528194e-05,
         -2.34103585e-05,  2.24822672e-07, -8.33723153e-07,
         -9.74591132e-07, -2.92367024e-05,  8.99867761e-05,
          7.50892659e-05,  1.18728727e-04],
        [ 1.56454324e-06,  8.90626531e-07, -2.27934485e-05,
         -1.88578720e-05,  3.09380200e-05,  2.50299338e-05,
          2.43572649e-05, -3.12872220e-05,  2.06230808e-04,
          1.18728727e-04,  2.81576980e-04]],

       [[ 1.08592167e-06,  1.16592809e-05,  1.40781876e-05,
          1.16056425e-05,  7.85376424e-06,  1.14676885e-05,
          9.83866449e-06,  1.88489469e-05, -1.28622653e-06,
          1.22681649e-06,  1.62732775e-06],
        [ 1.16592809e-05,  2.35871181e-02,  1.24989814e-02,
          1.97899790e-02,  9.84948916e-03,  1.27091274e-02,
          1.18407070e-02,  8.28265048e-03, -5.50589520e-04,
          2.21843860e-04,  8.09886040e-04],
        [ 1.40781876e-05,  1.24989814e-02,  1.96099341e-02,
          2.33609191e-02,  4.05857537e-03,  6.06065545e-03,
          5.00976560e-03,  2.58993306e-02,  1.76164419e-03,
          5.53979869e-04,  3.64576830e-04],
        [ 1.16056425e-05,  1.97899790e-02,  2.33609191e-02,
          3.17140983e-02,  4.76634941e-03,  7.94584418e-03,
          6.83232396e-03,  2.82272182e-02,  1.98236750e-03,
          4.65524543e-04,  6.15125510e-04],
        [ 7.85376424e-06,  9.84948916e-03,  4.05857537e-03,
          4.76634941e-03,  7.38456577e-03,  8.10123729e-03,
          7.50641015e-03,  9.51130840e-04, -7.01398733e-04,
          3.06652073e-04,  3.19199255e-04],
        [ 1.14676885e-05,  1.27091274e-02,  6.06065545e-03,
          7.94584418e-03,  8.10123729e-03,  9.34303759e-03,
          8.63652788e-03,  2.84051202e-03, -6.92871018e-04,
          3.53392595e-04,  4.89059907e-04],
        [ 9.83866449e-06,  1.18407070e-02,  5.00976560e-03,
          6.83232396e-03,  7.50641015e-03,  8.63652788e-03,
          8.02848259e-03,  1.58692130e-03, -7.19780302e-04,
          3.23201964e-04,  5.23654058e-04],
        [ 1.88489469e-05,  8.28265048e-03,  2.58993306e-02,
          2.82272182e-02,  9.51130840e-04,  2.84051202e-03,
          1.58692130e-03,  3.92261094e-02,  3.02674939e-03,
          5.55334315e-04, -2.23121923e-04],
        [-1.28622653e-06, -5.50589520e-04,  1.76164419e-03,
          1.98236750e-03, -7.01398733e-04, -6.92871018e-04,
         -7.19780302e-04,  3.02674939e-03,  4.77836327e-04,
          1.11911776e-04,  1.25411027e-04],
        [ 1.22681649e-06,  2.21843860e-04,  5.53979869e-04,
          4.65524543e-04,  3.06652073e-04,  3.53392595e-04,
          3.23201964e-04,  5.55334315e-04,  1.11911776e-04,
          1.95281039e-04,  3.76420239e-04],
        [ 1.62732775e-06,  8.09886040e-04,  3.64576830e-04,
          6.15125510e-04,  3.19199255e-04,  4.89059907e-04,
          5.23654058e-04, -2.23121923e-04,  1.25411027e-04,
          3.76420239e-04,  9.37240699e-04]],

       [[ 1.14813145e-05, -2.54255893e-05, -4.39833164e-05,
         -5.73374947e-05, -1.83098110e-05, -1.71114431e-05,
         -1.70875832e-05, -5.44767387e-05,  4.23107293e-05,
          2.53136836e-05,  4.15864405e-05],
        [-2.54255893e-05,  1.59933132e-03,  9.17814823e-04,
          9.82958540e-04,  1.18955299e-03,  1.26511920e-03,
          1.22871742e-03,  1.35899829e-03,  1.36258928e-04,
          1.07644917e-04,  1.70808864e-04],
        [-4.39833164e-05,  9.17814823e-04,  3.02081921e-03,
          2.50807144e-03,  1.47331752e-03,  1.22583589e-03,
          1.08593744e-03,  3.61126708e-03, -3.22995529e-05,
         -1.90093370e-05, -1.45191788e-05],
        [-5.73374947e-05,  9.82958540e-04,  2.50807144e-03,
          2.66853452e-03,  8.90217080e-04,  8.60392735e-04,
          7.00582983e-04,  3.34460902e-03, -5.12806466e-05,
         -4.68302315e-05, -4.71568453e-05],
        [-1.83098110e-05,  1.18955299e-03,  1.47331752e-03,
          8.90217080e-04,  2.14010936e-03,  1.75474786e-03,
          1.77838585e-03,  1.62678480e-03,  1.26812045e-04,
          1.38681579e-04,  2.25252307e-04],
        [-1.71114431e-05,  1.26511920e-03,  1.22583589e-03,
          8.60392735e-04,  1.75474786e-03,  1.59510680e-03,
          1.62242099e-03,  1.42842884e-03,  1.57508936e-04,
          1.37009066e-04,  2.31851662e-04],
        [-1.70875832e-05,  1.22871742e-03,  1.08593744e-03,
          7.00582983e-04,  1.77838585e-03,  1.62242099e-03,
          1.71015456e-03,  1.22954245e-03,  1.55002525e-04,
          1.35618347e-04,  2.23564230e-04],
        [-5.44767387e-05,  1.35899829e-03,  3.61126708e-03,
          3.34460902e-03,  1.62678480e-03,  1.42842884e-03,
          1.22954245e-03,  4.94921458e-03, -1.41681116e-04,
         -7.60727058e-05, -1.38798859e-04],
        [ 4.23107293e-05,  1.36258928e-04, -3.22995529e-05,
         -5.12806466e-05,  1.26812045e-04,  1.57508936e-04,
          1.55002525e-04, -1.41681116e-04,  8.98343303e-04,
          4.53255508e-04,  9.37140761e-04],
        [ 2.53136836e-05,  1.07644917e-04, -1.90093370e-05,
         -4.68302315e-05,  1.38681579e-04,  1.37009066e-04,
          1.35618347e-04, -7.60727058e-05,  4.53255508e-04,
          2.98026439e-04,  5.11473621e-04],
        [ 4.15864405e-05,  1.70808864e-04, -1.45191788e-05,
         -4.71568453e-05,  2.25252307e-04,  2.31851662e-04,
          2.23564230e-04, -1.38798859e-04,  9.37140761e-04,
          5.11473621e-04,  1.07984527e-03]],

       [[ 9.99087309e-06,  2.36017553e-05, -2.96295476e-06,
          1.45338046e-05, -6.62817342e-06, -1.42514490e-06,
         -1.95064052e-06, -6.44160437e-06, -3.12413416e-05,
         -8.32274138e-06, -3.00040965e-05],
        [ 2.36017553e-05,  1.00685144e-02,  8.91676019e-03,
          9.95094547e-03,  5.94903459e-03,  6.51420384e-03,
          6.02880783e-03,  1.01191541e-02,  4.53178790e-04,
         -2.02996613e-04, -2.75609339e-04],
        [-2.96295476e-06,  8.91676019e-03,  9.48237266e-03,
          1.00141181e-02,  6.31087457e-03,  6.61893042e-03,
          6.09357499e-03,  1.09015316e-02,  6.54342190e-04,
         -1.46505269e-04, -1.16645554e-04],
        [ 1.45338046e-05,  9.95094547e-03,  1.00141181e-02,
          1.11161489e-02,  6.35258872e-03,  6.83892764e-03,
          6.27148418e-03,  1.17248989e-02,  6.07787408e-04,
         -1.74265423e-04, -1.98377564e-04],
        [-6.62817342e-06,  5.94903459e-03,  6.31087457e-03,
          6.35258872e-03,  5.39414303e-03,  5.29301300e-03,
          5.04173864e-03,  6.96852631e-03,  5.86980189e-04,
         -8.10889634e-05,  7.99398616e-05],
        [-1.42514490e-06,  6.51420384e-03,  6.61893042e-03,
          6.83892764e-03,  5.29301300e-03,  5.36467035e-03,
          5.06993426e-03,  7.36696616e-03,  5.60117767e-04,
         -1.01798070e-04, -2.11861342e-06],
        [-1.95064052e-06,  6.02880783e-03,  6.09357499e-03,
          6.27148418e-03,  5.04173864e-03,  5.06993426e-03,
          4.83634337e-03,  6.76603853e-03,  5.21998619e-04,
         -9.54499621e-05,  9.74571907e-06],
        [-6.44160437e-06,  1.01191541e-02,  1.09015316e-02,
          1.17248989e-02,  6.96852631e-03,  7.36696616e-03,
          6.76603853e-03,  1.35392684e-02,  7.15526440e-04,
         -1.93488815e-04, -2.32200180e-04],
        [-3.12413416e-05,  4.53178790e-04,  6.54342190e-04,
          6.07787408e-04,  5.86980189e-04,  5.60117767e-04,
          5.21998619e-04,  7.15526440e-04,  1.15191863e-03,
          2.52284668e-04,  8.78141255e-04],
        [-8.32274138e-06, -2.02996613e-04, -1.46505269e-04,
         -1.74265423e-04, -8.10889634e-05, -1.01798070e-04,
         -9.54499621e-05, -1.93488815e-04,  2.52284668e-04,
          2.28937784e-04,  4.17732205e-04],
        [-3.00040965e-05, -2.75609339e-04, -1.16645554e-04,
         -1.98377564e-04,  7.99398616e-05, -2.11861342e-06,
          9.74571907e-06, -2.32200180e-04,  8.78141255e-04,
          4.17732205e-04,  1.37985385e-03]]])