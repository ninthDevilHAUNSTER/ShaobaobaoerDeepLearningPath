## Affine 层逆向传播理解



### 矩阵求导

$$ Y = X · W + B$$

- ∵
  - X  N*2
  - W 2*3
  - B 3*N
- ∴
  - X·W  N*3
  - Y N*3
- 以上是正向的过程



- ∵

  - DEFINE $$\cfrac{\partial L}{\partial Y}$$ AS L
  - L.size == Y.size == N,3
  - $X^T$ 2*N
  - $W^T$ 3*2

- ∴

  - 加法反向传播得

    - L  	N*3

  - 乘法反向传播

    - $\cfrac{\partial L}{\partial X} = L * W^T $

      N,2 = N,3 * 3,2

    - $\cfrac{\partial L}{\partial W} = X^T * L $

      2,3 =  2,N * N,3

### 参考图

书P148



