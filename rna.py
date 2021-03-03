import numpy as np
import numpy.linalg as la
from scipy.signal import convolve2d

class SVM:

    def __init__(self, x, y):
        # X = (xij): mxn
        # X:
        #
        #   x11 x12 ... x1n -> x1
        #   x21 x22 ... x2n -> x2
        #   ... ... ... ...
        #   xi1 xi2 ... xin -> xi
        #   ... ... ... ...
        #   xm1 xm2 ... xmn -> xm
        #
        self._x = x
        m, n = x.shape
        self._m = m
        self._n = n
        # Vetores de Convolução
        # Cv: mx(m-1)
        #
        #    1  1 ...  1 -> ones(1,m-1)
        #   -1  0 ...  0 
        #    0 -1 ...  0 -> -eye(m-1)
        #   .. .. ... .. 
        #    0  0 ... -1 
        #
        Cv = np.concatenate([
            np.ones([1,m-1]),
            -np.eye(m-1)
        ])
        # Matriz de diferenças ponto a ponto
        # F = (fij): (m(m-1)/2)xn
        #
        #      x11-x21     x12-x22    ...     x1n-x2n    -> diff x1 e x2
        #      x21-x31     x22-x32    ...     x2n-x3n    -> diff x2 e x3
        #        ...         ...      ...       ...
        #   x(m-1)1-xm1  x(m-1)2-xm2  ...  x(m-1)n-xmn   -> diff x(m-1) e xm  __ diferenças de 1
        #      x11-x31     x12-x32    ...     x1n-x3n    -> diff x1 e x3
        #      x21-x41     x22-x42    ...     x2n-x4n    -> diff x2 e x4
        #        ...         ...      ...       ...
        #   x(m-2)1-xm1     x22-xm2   ...     x2n-xmn    -> diff x(m-2) e xm  __ diferenças de 2
        #        ...         ...      ...       ...
        #        ...         ...      ...       ...
        #        ...         ...      ...       ...
        #      x11-xm1     x12-xm2    ...     x1n-xmn    -> diff x1 e xm      __ diferenças de (m-1)
        #
        F = np.zeros((1,n))
        for i in range(m-1):
            # Convolução da iteração
            # c: (i+2)x1
            #
            #    1
            #    0 \
            #   ..  > i zeros
            #    0 /
            #   -1
            #
            c = np.array([Cv[0:i+2,i]]).T
            F = np.concatenate([ F, convolve2d(x,c, mode='valid') ])
        F = F[1:,:]
        # D: Matriz de ditãncias euclidianas entre os pontos de x.
        # D = (dij)mxm ; dij = dist(xi,xj)
        # dist(xi,xj) = √∑k (xik - xjk)²
        # 
        # Dado que a matriz F já contém as diferenças entre xik e xjk para todo
        # i != j para todo k. Então:
        # 
        # S: (m(m-1)/2)x1
        #
        #   d11
        #   d23
        #   ...
        #   d(m-1)m
        #   d13
        #   d24
        #   ...
        #   d(m-2)m
        #   ...
        #   ...
        #   ...
        #   d1m
        #
        S = np.sqrt(np.sum(F ** 2, axis=1))
        # Agora deve-se deixar D na forma de matriz quadrada:
        # Note: dij = dji, então a matriz é simétrica
        # D:
        #
        #    0   d12  d13 ... d1m
        #   d12   0   d23 ... d2m
        #   d13  d23   0  ... d3m
        #   ...  ...  ... ... ...
        #   d1m  d2m  d3m ... dmn
        #
        D = np.zeros((m,m))
        # Ajustar endereços da matrix triangular em que será acrescentado:
        #
        #  i\j | 0    1    2    3    4  ...   m-1
        #  ----+----------------------------------
        #    0 | .    1    m   2m   3m  ... (m(m-1)/2)
        #    1 | .    .    2  m+1 2m+1  ...  ...
        #    2 | .    .    .    3  m+2  ...  ...
        #    3 | .    .    .    .    4  ...  ...
        #   .. | .    .    .    .    .  ... 2m-1
        #   .. | .    .    .    .    .  ...  m-1
        #  m-1 | .    .    .    .    .  ...  .
        #
        trindex = np.triu_indices(m,1)
        sortindex = np.argsort(np.subtract(trindex[1],trindex[0]))
        index = tuple([ arr[sortindex] for arr in trindex ])
        D[index] = S
        D = D + D.T
        # Para calcular as diferenças entre todos os pontos de xi e xj, i!= j é
        # utilizada a opeção de convolução
        H = np.e ** -D
        self._p = la.solve(H, y)

    def __call__(self, x):
        # X (de teste): qxn
        #
        #   x11 x12 ... x1n -> x1
        #   x21 x22 ... x2n -> x2
        #   ... ... ... ...
        #   xi1 xi2 ... xin -> xi
        #   ... ... ... ...
        #   xq1 xq2 ... xqn -> xq
        #
        q, n = x.shape
        m = self._m
        if n != self._n:
            raise Exception("X dimension not match with trained")
        # Repete cada vetor m vezes para facilitar o cálculo da distancia
        # euclidiana.
        # 
        #   xt: mqxn
        #   
        #   x1
        #   x1
        #   ..
        #   x2
        #   x2
        #   ..
        #   ..
        #   ..
        #   ..
        #   xq \
        #   xq  > m vezes
        #   .. /
        #
        X = x.values.repeat(m, axis=0)
        # Repete a matriz de treino q vezes (seguidas)
        # T: mqxn
        #
        #   x1
        #   ..
        #   xm
        #   x1
        #   ..
        #   xm
        #   ..
        #   ..
        #   x1
        #   ..
        #   xm
        #
        T = np.tile(self._x.T, q).T
        # Matrix de diferenças
        F = X - T
        # Matriz de distâncias
        S = np.sqrt(np.sum(F ** 2, axis=1))
        D = S.reshape((q,m))
        # Matriz de ativação
        # H: qxm
        H = np.e ** -D
        r = H @ self._p
        k = r > 0
        y = k + np.logical_not(k) * -1
        return y

