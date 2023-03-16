# Sliding Filter For AWGN Denosing

This package contains the [`sliding filter`](./sliding_filter.py) for removal of
additive stationary white Gaussian noise (AWGN) from 2D/3D image using the sliding 
window approach. 
Unlike traditional way of implementation using for-loops for local filtering (see vid1),
this implementation does not use any for-loop (see vid2), making the code way faster!










### Denoising demo
simply run the following command to see a denoising demo

```python
    python3 sliding_filter.py 
```





### Demo of the traditional procedure vs. the implemented one


## Disclaimer
Copyright (C) 2023    Nasser Eslahi

Closures is provided under the [`MIT LICENSE`](./LICENSE).


## Feedback
If you have any comment, suggestion, or question, please do contact
 [Nasser Eslahi](https://orcid.org/0000-0002-1134-9318)
\
 nasser.eslahi@tuni.fi
\
nasser.eslahi@gmail.com