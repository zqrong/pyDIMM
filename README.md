# pyDIMM
A Python Dirichlet Multinomial Mixture Model.

## Ready

**Typically, if you import pyDIMM in your program, clibs will be automatically compiled, and you can skip this part.**

We need to first compile the files in `clibs`. The makefile has been provided.
```console
cd clibs && make
```

Also, you can compile it by yourself using gcc. Compile `./clibs/pyDIMM_libs.c` by the instructions in the head of that file, and then you will get a `./clibs/pyDIMM_libs.so` file.  
```console
cd clibs && gcc -lm -shared -fPIC -o pyDIMM_libs.so pyDIMM_libs.c
```

Check the files now.
```
└───pyDIMM
    |
    ├───pyDIMM
    |   ├───__init__.py
    |   ├───class_DIMM.py
    |   └───clibs
    |       ├───makefile
    |       ├───pyDIMM_libs.c
    |       └───pyDIMM_libs.so
    |
    └───Some other files...
```

## How to use

All the methods are based on the class `DIMM`. You need an instance of `pyDIMM.DIMM` to get started.

```python
import pyDIMM
```

- Example 1
    ```python
    dimm_0 = pyDIMM.DIMM(observe_data=your_data, n_components=3, alpha_init='kmeans')
    ```
- Example 2
    ```python
    dimm_1 = pyDIMM.DIMM(observe_data=your_data, n_components=5, alpha_init='manual', prior_label=your_label, print_log=True)
    ```

### Train (by EM algorithm)

Use EM algorithm to train the model. The EM algorithm is written in C (yes, it's the code in `pyDIMM.c`, we use `ctypes` to implement that.).

- Example
    ```python
    dimm_0.EM(max_iter=100, max_loglik_tol=1e-3, max_alpha_tol=1)
    ```

OK, the DIMM is already trained now. We need to get the result back. All the result information is in one dictionary.

- Example
    ```python
    result_0 = dimm_0.get_model()
    print(result_0)
    ```
    you're supposed to see
    ```
    {
    'alpha': array([...]),
    'pie': array([...]),
    'delta': array([...]),
    'loglik': ...,
    'AIC': ...,
    'BIC': ...
    }
    ```

That's all! You get it.  

>Note:  
Once you get the trained DIMM model, the result parameters are stored in the DIMM instance. So every time you want to retrieve the result back, just call the `get_model()` method. (Only if you don't change the instance before, such as call `EM()` again. That will of course change the result stored.)

### Predict

Sometimes you are not only want to fit a DIMM, but also want to use this model to predict some other data (If you don't want, forget it). Fortunately, we have the method `predict()`.

- Example
    ```python
    predict_res = dimm_0.predict(another_data)
    ```

Then you'll get the predict result `label` and `delta` in the `predict_res` dictionary. Find the detail explanation in the doc in codes.

### Save & Load

All information of your DIMM instance can be saved to `.npy` file and then can be loaded anytime and anywhere.
- Example
    ```python
    dimm_0.save('./dimm_0_file')
    ```
    After this, a new file called `dimm_0_file.npy`(the postfix .npy is automatically added) will appear in your current folder. You can read from the file later.
    ```python
    dimm_load = pyDIMM.DIMM.load('./dimm_0_file.npy')
    ```
    
## Contact

Ziqi Rong <rongziqi@sjtu.edu.cn>
