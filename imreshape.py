import numpy as np

def imreshape(action, input_img, partition_size, step_size=None, size_full_img=None) -> np.array:
    """
    imreshape extracts 2D blocks or 3D cubes of specified size (partition_size) from
    the input signal (input_img) with specified step-size or stride (step_size) under 
    action="parse" or action="p".
    
    Furthermore, imreshape can constitute an image of full size (size_full_img)
    given the partitioned image under action="aggregate" or action="a".
    
    Args:
    - action: str, 
              the selected reshaping action, which can be either
              "parse" or "p" to partition the input 2D or 3D image into 2D blocks or 3D cubes, or
              "aggregate" or "a" to constitute a 2D or 3D image from the partitioned blocks or cubes.

    - input_img: numpy array, 
              under action="parse" or action="p", input_img is a 2D or 3D image to be partitioned. 
              under action="aggregate" or action="a", input_img is a partitioned image to be reshaped 
              back into a 2D or 3D image.

    - partition_size: list or tuple, 
              the size of each partitioning element, i.e. the the size of extracted blocks or cubes.
              the length of partition_size can be 2 (resp. 3) for block (resp. cube) extraction. 

    - step_size: list or tuple or str,               
              if the class is either 'str' , then step_size takes the following options:
              - "sliding"  : it does fully-overlapped parsing                
              - "distinct" : it does non-overlapping parsing
              if the class is list or tuple, then step_size indicates the step-size (stride) toward 
              each dimension.

    - size_full_img: list or tuple, 
              the size of full 2D or 3D image (to reshape the partitioned image into that size).
              
    Returns:
    - output_img: numpy array, 
              under action="parse" or action="p", output_img is a 2D partitioned image that each of 
              its columns represents an extracted 2D block or 3D cubes.
              under action="aggregate" or action='a', output_img is a 2D or 3D image constituted from 
              the 2D partitioned image.


    Creator: Nasser Eslahi    (nasser.eslahi@gmail.com)          
    """


    if input_img.ndim not in [2, 3]:
        raise ValueError('the input image should be either 2D or 3D!')
    
    if len(partition_size) not in [2, 3]:
        raise ValueError('our partitioning is either block-based or cube-based!')
    
    if step_size is None:
        step_size = np.maximum(np.round(partition_size/2), 1).astype(int)
    elif isinstance(step_size, str):
        if   step_size.lower() == "distinct":  # non-overlapping parsing
             step_size = partition_size
        elif step_size.lower() == "sliding":   # fully-overlapped parsing
             step_size = np.ones(partition_size.size)
        else:
            raise ValueError('the 3rd input parameter (i.e. step_size) should be either\n'
                             'a 1x%d array, indicating the step size, or the following options:\n'
                             '"distinct": non-overlapping parsing\n'
                             '"sliding" : fully-overlapped parsing' % (input_img.ndim))

    if len(step_size) == 2:
        step_size = np.hstack((step_size, 1))
        
    if len(partition_size) == 2:
        partition_size = np.hstack((partition_size, 1))

    partition_size = np.array(partition_size)
    step_size      = np.array(step_size)    
        
    if step_size.size != partition_size.size:
        raise ValueError('Please check the selected step-size!\n'
                         'Please have a look at the syntax '
                         '%s' % (imreshape.__name__))

    
    if action is None:
        action = "parse"
    
    actions = ["parse", "p", "aggregate", "a"]
    if action.lower() not in actions:
        raise ValueError('Your selected action ''%s'' (as the 1st input) does not exist!\n'
                         'We have the following actions:\n'
                         '  "parse"   or "p": partitioning the 2D/3D image into blocks/cubes\n'
                         '"aggregate" or "a": reshaping the partitioned blocks/cubes into a 2D/3D image.\n'
                         % (action))
    
    if action.lower() in ["parse", "p"]:
        input_img  = input_img.reshape(*input_img.shape, 1) if input_img.ndim == 2 else input_img
        output_img = im_parser(input_img, partition_size, step_size)
    else:
        if size_full_img is None:
            raise ValueError('The size of full image should be provided for aggregation!\n'
                             'Please see the example in the syntax %s' % (imreshape.__name__))
        elif len(size_full_img) not in [2, 3]:
            raise ValueError('Please provided a correct size of the full 2D or 3D image,\n'
                             'so that the partitioned input image would be reshaped into that size!')

        if len(size_full_img) == 2:
            size_full_img = np.hstack((size_full_img, 1))

        size_full_img = np.array(size_full_img)    
        indices = np.arange(np.array(size_full_img).prod()).reshape(size_full_img)
        partitioned_inds = im_parser(indices, partition_size, step_size)
        
        if input_img.size != partitioned_inds.size:
            raise ValueError('Either the size of the full image (i.e. the 5th input) '
                             'OR the partitioned image (i.e. the 1st input) '
                             'is wrongly selected.\n'
                             'Please see the example in the syntax %s' % (imreshape.__name__))
        
        output_img = im_aggregator(input_img, partitioned_inds, size_full_img)
    
    return output_img






def im_parser(fullSize_img, partition_size, step_size):
    """
    Extracts 2D blocks (resp. 3D cubes) from a 2D (resp. 3D) image, vectorizes them as columns 
    and then stacks them in a 2D matrix.

    Parameters:
    -----------
    fullSize_img : ndarray
        The full-size input image, which can be 2D or 3D array.

    partition_size : tuple
        A tuple of integers specifying the size of each partition (block or cube).
        If partition_size has three elements, then partitioning is cube-based, otherwise 
        it is block-based.
    
    step_size : tuple
        A tuple of integers specifying the step-size for sliding across each dimension.

    Returns:
    --------
    partitioned_img : ndarray
        A 2D matrix where each column represents a partitioned block or cube of the input image, 
        vectorized as a column.
    """

    s1, s2, s3 = partition_size
    Sz1, Sz2, Sz3 = fullSize_img.shape
    step1, step2, step3 = step_size

    # to make more efficient memory usage, one may assign a unique name to "ind_1st_element_depth", 
    # "ind_1st_element", "ind_plus_rows", "ind_both_rows_cols" and "all_parsing_indices"


    # indices of the first element in the depth dimension, starting from 0 which indicates the first slice
    ind_1st_element_depth = np.unique(np.concatenate((np.arange(0, Sz3-s3+1, step3), [Sz3-s3])))*Sz1*Sz2

    # indices of the first element within each partition (block or cube), ...
    # where each index is the representative of the index of top left corner ...
    # within each block or cube
    
    unique_1 = np.unique(np.concatenate((np.arange(0, Sz1-s1+1, step1),[Sz1-s1])))
    unique_2 = np.unique(np.concatenate((np.arange(0, Sz2-s2+1, step2),[Sz2-s2]))) * Sz1
    ind_1st_element = (unique_1 + unique_2[:, None] + ind_1st_element_depth[:, np.newaxis, np.newaxis]).transpose(2,1,0)
    
    # computing the row indices of partitions
    ind_plus_rows = (np.reshape(ind_1st_element,(1,-1), order='F') + np.arange(s1)[:,np.newaxis])[:, np.newaxis,:]

    # computing the row-and-column indices of the partition
    ind_both_rows_cols  = np.reshape(ind_plus_rows + np.arange(s2)[:, np.newaxis]*Sz1, (s1*s2,1, -1), order='F')

    # indices of partitioned elements 
    all_parsing_indices = ind_both_rows_cols + np.arange(s3)[np.newaxis, :, np.newaxis]*Sz1*Sz2

    # indices of partitioned elements (vectorized)
    all_parsing_indices = np.reshape(all_parsing_indices, (-1, all_parsing_indices.shape[2]), order='F')

    # extract the vectorized partitioned image from the computed indices
    partitioned_img     = fullSize_img.ravel('F')[all_parsing_indices]
    
    return partitioned_img






def im_aggregator(partitioned_img, partitioned_inds, size_full_img):
    # Calculate the mean of partitioned_img over each group in partitioned_inds
    # using the `np.bincount` and `np.divide` functions
    # Note that `np.bincount` is slower than "accumarray" in Matlab
    full_img = np.divide(np.bincount(partitioned_inds.flatten(), weights=partitioned_img.flatten()), 
                         np.bincount(partitioned_inds.flatten()))

    
    # Reshape the output into a 2D or 3D image based on size_full_img
    full_img = np.reshape(full_img, size_full_img)
    
    return full_img






if __name__ == "__main__":
    
    img = np.random.random(size=(256,223,16))

    partition_size = (31,25,12)
    step_size  = (8,7,6)

    print(f' extracting cubes of size {partition_size} with stride {step_size}, \
from the input image of size {img.shape}')
    partitioned_img = imreshape("p", img, partition_size, step_size) 


    print(f' aggregating the extracted cubes to reshape back the original image')
    reshaped_img = imreshape("a", partitioned_img, partition_size, step_size, img.shape) 
