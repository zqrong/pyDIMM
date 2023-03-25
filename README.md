# pyDIMM
A Python Dirichlet Multinomial Mixture Model.

## Quick Start

You can install pyDIMM by pip.

```
pip install pyDIMM
```

pyDIMM needs certain version of scikit-learn package to work. We recommend you to have `scikit-learn==1.1.3`, which has been tested to work normally.

After that, you can import pyDIMM and use it to fit a Dirichlet Multinomial Mixture model.

```python
import numpy as np
import pyDIMM

X = np.random.randint(1,100,size=[200,100])

dimm = pyDIMM.DirichletMultinomialMixture(
    n_components=3,
    tol=1e-3,
    max_iter=100,
    verbose=2,
    pytorch=0
).fit(X)

print('Alphas:', dimm.alphas)
print('Weights:', dimm.weights)

label = dimm.predict(X)
print('Prediction label:', label)
```

## Contact

[Ziqi Rong](https://www.zqrong.com) ([ziqirong@umich.edu](mailto:ziqirong.umich.edu))