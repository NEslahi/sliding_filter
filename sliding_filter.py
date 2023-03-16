import numpy as np
from imreshape import *
from scipy.fft import fft, dct, dst
import pywt, warnings


def sliding_filter(noisy_img, noise_std, transform_type=None, partition_size=None, 
                   step_size=None, shrinkage_type=None)-> np.array:
    """
    sliding_filter performs (AWGN) denoising on the sets of 2D blocks or 3D cubes 
    extracted from the noisy image using sliding window.  
    
    * NOTE: My implementation of the sliding filter does not use any for-loop for sliding
     window, which makes the code way faster *

    * NOTE: The `sliding filter` accepts a variety of sparsifying/decorrelating transforms
    for the local transform-domain filtering step. This transforms, which are separable,can
    be selected from the same family or composed of different families (see the details of 
    `transform_type` in the following syntax section for more information). *
    
    
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
	
	------------------------------------------------------------------------------------

    Creator: Nasser Eslahi    (nasser.eslahi@{gmail.com, tuni.fi})          
    """

    if noisy_img.ndim not in [2, 3]:
        raise ValueError('the input image should be either 2D or 3D! Please see the syntax!')

    
    if partition_size is None:
        partition_size = tuple(min(x) for x in zip((8,)*noisy_img.ndim, noisy_img.shape))
    elif len(partition_size) not in [2, 3]:
        raise ValueError('our partitioning is either block-based or cube-based!')



    if step_size is None:
        step_size = tuple(max(x) for x in zip(
            (1,)*noisy_img.ndim, tuple(int(y/2) for y in partition_size)
            ))
    elif isinstance(step_size, str):
        if   step_size.lower() == "distinct":  # non-overlapping parsing
             step_size = partition_size
        elif step_size.lower() == "sliding":   # fully-overlapped parsing
             step_size = np.ones(partition_size.size)
        else:
            raise ValueError('the 5th input parameter (i.e. step_size) should be either\n'
                             'a 1x%d array of integers, indicating the step size, \n'
                             'or the following options:\n'
                             '"distinct": non-overlapping parsing\n'
                             '"sliding" : fully-overlapped parsing' % (noisy_img.ndim))


    if len(step_size) == 2:
        step_size = np.hstack((step_size, 1))
        
    if len(partition_size) == 2:
        partition_size = np.hstack((partition_size, 1))

    partition_size = np.array(partition_size)
    step_size      = np.array(step_size)    
        
    if step_size.size != partition_size.size:
        raise ValueError('Please check the selected step-size!\n'
                         'Please have a look at the syntax '
                         '%s' % (sliding_filter.__name__))


    if transform_type is None:
        transform_type = ('dct',)*len(partition_size) 

    
    if shrinkage_type is None:
        shrinkage_type = 'hard'
    elif shrinkage_type.lower() not in ['h', 'hard', 's', 'soft']:
        raise ValueError('This implementation accepts\n'
                         ' "h" or "hard" : hard-shrinkage, or\n'
                         ' "s" or "soft" : soft-shrinakge,\n'
                         'for the shrinkage_type')


    ## setting the sparsifying/decorrelating transform bases
    tm_0 = get_transform_matrix(partition_size[0], transform_type[0])
    tm_1 = get_transform_matrix(partition_size[1], transform_type[1])
    if partition_size[2] != 1:
        tm_2 = get_transform_matrix(partition_size[2], transform_type[2])
    else:
        tm_2 = np.array([1])


    ## extracting blocks/cubes from the noisy image
    dmy = imreshape('parse', noisy_img, partition_size, step_size)

    ## computing the 2D/3D spectra of all extracted blocks/cubes
    dmy = tm_0          @ reshp(dmy, (partition_size[0], -1))  
    dmy = tm_1.conj()   @ reshp( 
                            np.transpose( 
                                reshp(dmy, (partition_size[0], partition_size[1], -1)),
                            [2, 0, 1]), 
                          (-1, partition_size[1])).T

    dmy = tm_2.conj().T @ reshp(dmy.T, (partition_size[2], -1))
    dmy = reshp(dmy.T, (-1, np.prod(partition_size))).T


    # lambda_thr = np.sqrt(2*np.log(np.prod(partition_size))) # Donoho's universal thresholing factor:
    lambda_thr = 3  # thresholding factor based on 3-sigma rule
    threshold  = lambda_thr * noise_std  # threshold value
    if shrinkage_type in ["h", "hard"]:
        ## hard-thresholding step 
        # (exploiting sparsity by killing insignificant coefficients corresponding mostly to noise)
        dmy = dmy * (np.abs(dmy) > threshold)
    else:
        ## soft-thresholding step 
        dmy = np.sign(dmy) * np.maximum(np.abs(dmy) - threshold, 0)

    

    ## computing the 2D/3D inverse transform of all spectra (getting denoised blocks/cubes)
    dmy = tm_2.conj()   @ reshp(dmy.T, (-1, partition_size[2])).T
    dmy = tm_1.conj().T @ reshp(dmy,   (-1,partition_size[1])).T 
    dmy = tm_0.T        @ reshp(dmy,   (-1, partition_size[0])).T
    dmy = reshp(dmy, (np.prod(partition_size), -1))

    ## aggregating all extracted & denoised blocks/cubes into their original positions
    denoised_img = imreshape('aggregate', dmy, partition_size, step_size, noisy_img.shape)

    return denoised_img.squeeze()












def get_transform_matrix(N, transform_type, dec_levels=0):
    """
    get_transform_matrix creates forward and inverse transform matrices, which allow us 
    computing the spectra of the underlying signal. The forward transform matrix is normalized
    so that the l2-norm of each basis element is 1.

    Args:
        N (int): Size of the transform (for wavelets, must be 2^K)
        
        transform_type (str): sparsifying/decorrelating transform; the string can be 'dct',
         	'fft', 'hadamard', or anything that is listed by 'pywt.wavelist()' (e.g.,
         	'haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey'), identity transform,
         	'DCrand'-- an orthonormal transform with a DC, and all the other basis 
         	elements of random nature
          
        dec_levels (int): If a wavelet transform is generated, this is the desired
         	decomposition level. It must be in the range [0, log2(N)-1], where "0" implies
         	full decomposition. Default is 0.

    Returns:
        Tforward (ndarray): (N x N) Forward transform matrix
        #Tinverse (ndarray): (N x N) Inverse transform matrix
    """
    if dec_levels is None:
        dec_levels = 0

    if N == 1:
        Tforward = np.ones((1, 1))
    elif transform_type.lower() == 'hadamard':
        Tforward = hadamard(N)
    elif transform_type.lower() == 'dct':
        Tforward = dct(np.eye(N), axis=0, norm='ortho')
    elif transform_type.lower() == 'fft':
        Tforward = fft(np.eye(N))
    elif transform_type.lower() == 'dst':
        Tforward = dst(np.eye(N), axis=0, norm='ortho')
    elif transform_type.lower() == 'eye':
        Tforward = np.eye(N)
    elif transform_type.lower() == 'dcrand':
        x = np.random.randn(N, N)
        x[:, 0] = 1
        Q, _ = np.linalg.qr(x)
        if Q[0, 0] < 0:
            Q = -Q
        Tforward = Q.T
    else:  # a wavelet decomposition supported by 'pywt.wavedec' (e.g., 'haar', 'bior1.5', etc.)
                        
        ## SUPPORTED WAVELET FAMILIES
        # haar family: haar
        # db family: db1, db2, db3, db4, db5, db6, db7, db8, db9, db10, db11, db12, db13, db14, db15, db16, db1
        # sym family: sym2, sym3, sym4, sym5, sym6, sym7, sym8, sym9, sym10, sym11, sym12, sym13, sym14, sym15,
        # coif family: coif1, coif2, coif3, coif4, coif5
        # bior family: bior1.1, bior1.3, bior1.5, bior2.2, bior2.4, bior2.6, bior2.8, bior3.1, bior3.3, bior3.5
        # rbio family: rbio1.1, rbio1.3, rbio1.5, rbio2.2, rbio2.4, rbio2.6, rbio2.8, rbio3.1, rbio3.3, rbio3.5
        # dmey family: dmey

        ## MODE : [’zpd’, ’cpd’, ’sym’, ’ppd’, ’sp1’, ’per’]
        # 'zpd' : zero-padding
        # 'cpd' : constant-padding
        # 'sym' : symmetric-padding
        # 'ppd' : periodic-padding
        # 'sp1' : smooth-padding
        # 'per' : periodization  **we use this one**

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Tforward = np.zeros((N, N))
            for i in range(N):
                Tforward[:, i] = np.concatenate(np.array(
                                    pywt.wavedec(
                                        np.roll([1] + [0] * (N - 1), dec_levels + i), 
                                        transform_type, level=int(np.log2(N)),
                                        mode='periodization'),
                                    dtype=object)).reshape(-1,)                                   


    # Normalize the basis elements
    if transform_type.lower() != 'fft':
        Tforward = (Tforward.T * np.sqrt(1.0 / np.sum(Tforward ** 2, axis=1))).T

    # # Compute the inverse transform matrix
    # Tinverse = np.linalg.inv(Tforward)

    return Tforward



def hadamard(N):
    """ hadamard transform """
    Tforward = np.array([1])
    if N > 1:
        n = int(np.log2(N))
        H = np.array([[1, 1], [1, -1]])

        for i in range(n):
            Tforward = np.kron(Tforward, H)


def reshp(x, size_x):
    """ reshape the elements columnwise """
    return np.reshape(x, size_x, order='F')









if __name__ == "__main__":
    """    
	Test: An image (RGB or gray) is randomly selected from the set of available
	skimage.data images, which is then corrupted by an AWGN with a random standard
	deviation. The sliding filter is then applied for the task of denoising.
	Please feel free to play with the parameters! please check the syntax!
	
	"""

    from skimage import data
    import matplotlib.pyplot as plt 

    test_images = [
    'astronaut',
    'camera',
    'cell',
    'checkerboard',
    'chelsea',
    'clock',
    'coins',
    'coffee',
    'horse',
    'hubble_deep_field',
    'immunohistochemistry',
    'logo',
    'microaneurysms',
    'moon',
    'page',
    'rocket',
    'shepp_logan_phantom',
    'text'
    ]

    # Load and display an image (selected randomly) from the above list
    # idx = np.random.randint(0,len(test_images))
    idx = 1
    x = getattr(data, test_images[idx])()
    
    test_image_name = 'fake_cameraman!' if test_images[idx]=='camera' else test_images[idx]
    print(test_image_name)
    
    max_x = np.max(x)
    # randomely selected AWGN standard deviation (use your own desired value!)
    noise_std = np.random.uniform(.05, .5) * max_x
    
    # add noise to clean image
    z = x + np.random.normal(0, noise_std, x.shape) # noisy image
    PSNR_nsy = 10*np.log10(max_x**2/np.mean((x - z)**2))
    
    # denoising
    xhat = sliding_filter(z, noise_std) # denoised image
    # xhat = sliding_filter(z, noise_std, transform_type=('bior1.5','haar','dct')[:len(z)] )
    
    PSNR_den = 10*np.log10(max_x**2/np.mean((x - xhat)**2))

    print(f"PSNR of the noisy and denoised images are \
respectively {PSNR_nsy:.2f} (dB) and {PSNR_den:.2f} (dB)")

    

    def im_show(X, axis_idx, ttl):
        if len(X.shape) == 2:
            # Grayscale image
            axes[axis_idx].imshow(np.clip(X/max_x, 0, 1), cmap='gray')
        else:
            # Color image
            axes[axis_idx].imshow(np.clip(X/max_x, 0, 1))    
        axes[axis_idx].set_title(ttl)  
        

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12,4)) 
    im_show(x,    0, f'ground-truth\n range$\in$[{np.min(x):.1f},{np.max(x):.1f}]') 
    im_show(z,    1, f'noisy img\n PSNR={PSNR_nsy:.2f}(dB)') 
    im_show(xhat, 2, f'denoised img\n PSNR={PSNR_den:.2f}(dB)')
    for axi in axes.ravel(): axi.set_axis_off()
    fig.suptitle(test_image_name + f' (noise std: {noise_std:.2f})')
    plt.show() 
    




