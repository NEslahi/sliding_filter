# Sliding Filter For AWGN Denosing

This package contains the [`sliding filter`](./sliding_filter.py) for removal of
additive stationary white Gaussian noise (AWGN) from 2D/3D images using the sliding 
window approach. 
Unlike a traditional way of implementation using for-loops for local filtering,
this implementation does not use any for-loop, making the code way faster!

NOTE: The [`sliding filter`](./sliding_filter.py) accepts a variety of sparsifying/decorrelating
transforms for the local transform-domain filtering step. This transforms, which are separable,
can be selected from the same family or composed of different families (see
the details of `transform_type` in the [syntax](#syntax) section for more information).

</br></br>


### Background :book:
The sliding filtering of a noisy image $z=x+\eta$, where $\eta\sim\mathcal{N}(0,\sigma^2)$, 
proceeds as the following:
</br>
>Create two null-arrays $\hat{x}^w$ and $w$ as size as $z$, respectively, representing 
the accumulated denoised blocks/cubes and accumulated weights (buffers)
>> For each block extracted from $z$ do
>>> Apply a transform-domain filtering to the $i$-th extracted block/cube 
```math
\hat{x}_i^{w} =\mathcal{T}_{d\textrm{D}}^{-1}\Big(\Upsilon\big(\mathcal{T}_{d\textrm{D}}(z_i),~\lambda\sigma\big)\Big)\,,
```
>>> where $\mathcal{T}_{d\textrm{D}}$ is a $d$-dimensional sparsifying/decorrelating transform 
($d=2$ for blocks and $d=3$ for cubes), $\Upsilon$ is a nonlinear shrinkage operator (e.g.,
soft- or hard-shrinkage), and $\lambda>0$ is a thresholding factor.
\
>>> Create a buffer block/cube (i.e. array of all ones)
```math
w_i \equiv 1
```

>> Add $\hat{x}_i^{w}$ to its corrsponding position in $\hat{x}$
\
>> Add $\hat{w}_i$ to its corrsponding position in $w$
>
> Compensating the effect of accumulation
```math
\hat{x} = \hat{x}^{w}\oslash{w}
```


</br></br>

### Syntax :scroll:
function interface
```python
    sliding_filter(noisy_img, noise_std, transform_type=None, partition_size=None, step_size=None, shrinkage_type=None)
    """
    Args:
    - noisy_img (ndarray): 2D/3D noisy image (2D: monochromatic; 3D: RGB, hyperspectral, ...)

    - nois_std   (float) : noise standard deviation    
      
    - transform_type (list/tuple of str) : sparsifying/decorrelating transform
              the string can be 'dct', 'fft', 'hadamard', or anything that is listed by
              'pywt.wavelist()' (e.g., 'haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey'),
              identity transform, 'DCrand'-- an orthonormal transform with a DC, and
              all the other basis elements of random nature. The transform in the list/tuple
              of transform_type can be different (combination of separable transforms), e.g., 
                    transform_type = ('bior1.5','haar','dct')[:noisy_img] 
              or the same, e.g.,
                    transform_type = ('dct',)*len(noisy_img)
              
    - partition_size (list/tuple of int): size of each extracted block/cube.
               Note that for a chosen wavelet as transform_type, its corresponding partition 
               size must be dyadic 2^K.       

    - step_size (str, or list/tuple of int),               
              if the type is either 'str' , then step_size takes the following options:
              - "sliding"  : filter operates on "fully"-overlapped blocks/cubes (i.e. the
              two adjacent blocks/cubes have one pixel slide toward a dimension)                
              - "distinct" : filter operates on non-overlapping parsing blocks/cubes
              if the type is list or tuple, then step_size indicates the step-size (stride)
               toward each dimension.
               
    - shrinkage_type (str) : type of shrinkage operator, which can be either of the following:
              - "h" or "hard"   : hard-shrinkage (default)
              - "s" or "soft"   : soft-shrinkage

              
    Returns:
    - denoised_img (ndarray): 2D/3D denoised image.
    """
```


</br></br>



### Denoising demo :mag: 
simply run the following command to see a denoising demo

```python
    python3 sliding_filter.py 
```
</br></br>




### Demo of the traditional procedure vs. the implemented one :bulb:

Traditional sliding filtering approach proceeding block-by-block 

https://user-images.githubusercontent.com/48449082/226089309-30db8c89-b9f2-4158-8ff8-89f502ec8148.mp4

</br></br>

Implemented approach using [`sliding filter`](./sliding_filter.py), where extraction/aggregation of blocks
is done by [`imreshape`](./imreshape.py) 

https://user-images.githubusercontent.com/48449082/225544822-81366711-1ff0-47ea-adac-c65c58e26e92.mp4



## Disclaimer :copyright:
Copyright &copy; 2023    Nasser Eslahi.    All right reserved.

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.


## Feedback :speaking_head: :mailbox_with_mail:
If you have any comment, suggestion, or question, please do contact
 [Nasser Eslahi](https://orcid.org/0000-0002-1134-9318) :snowman:
\
nasser.eslahi@tuni.fi
\
nasser.eslahi@gmail.com
