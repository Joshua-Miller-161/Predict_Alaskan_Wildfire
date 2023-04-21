def Funnel(start_size, end_size, r=np.e):
    sizes = [start_size]

    i = 1
    while ((round(start_size / r**i)) > end_size):
        sizes.append(round(start_size / r**i))
        i += 1

    sizes.append(end_size)

    return sizes


def MaxPool2D(x, pool_size=(2, 2), strides=None, axes=(1,2), padding='valid'):
    '''
    Made by ChatGPT - Downsamples a 3D numpy array along the specified axes by taking every nth element.
    :param x: ndarray, The input array to downsample. Must have shape (dim1, spat_dim1, spat_dim2).
    :param pool_size: tuple, The number of elements aggregated and then sent to the maximum.
    :param strides: idk, 
    :param axes: tuple, The axes on which to downsample. Default is (1, 2).
    :return: ndarray
        The downsampled array, with shape (dim1, new_spat_dim1, new_spat_dim2), where
        new_spat_dim1 = ceil(spat_dim1 / downsample_rate) and
        new_spat_dim2 = ceil(spat_dim2 / downsample_rate)
    '''
    if strides is None:
        strides = pool_size

    if padding == 'same':
        pad_height = max(pool_size[0] - x.shape[1] % pool_size[0], 0)
        pad_width = max(pool_size[1] - x.shape[2] % pool_size[1], 0)
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        x = np.pad(x, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')

    out_height = (x.shape[1] - pool_size[0]) // strides[0] + 1
    out_width = (x.shape[2] - pool_size[1]) // strides[1] + 1

    out = np.zeros((x.shape[0], out_height, out_width))

    for i in range(out_height):
        for j in range(out_width):
            out[:, i, j] = np.max(x[:, i*strides[0]:i*strides[0]+pool_size[0], j*strides[1]:j*strides[1]+pool_size[1]], axis=axes)

    return out


def Scale(data, reference, method='standard'):
    '''
    This function scales the data, either using the StandardScaler or MaxAbsScaler
    :param data: nparray, the data to be scaled 
    :param reference: nparray, what to use as a reference for the scaler, must be same size as data
    :param method: str, either 'standard' or 'maxabs', chooses the scaling method
    '''
    data = np.array(data)
    reference = np.array(reference)

    orig_shape = np.shape(data)

    if method == 'standard':
        scaler = preprocessing.StandardScaler() # Makes mean = 0, stdev = 1
    elif method == 'maxabs':
        scaler = preprocessing.MaxAbsScaler() # Scales to range [-1, 1], best for sparse data
    else:
        raise ValueError('Invalid method specified. Allowed values are "standard" and "maxabs".')

    scaler.fit(reference.ravel().reshape(-1, 1)) # Flatten array in order to obtain the mean over all the space and time

    scaled_data = scaler.transform(data.ravel().reshape(-1, 1))

    return scaled_data.reshape(orig_shape)


def DownSample(data, downsample_rate, axis, delete=False):
    '''
    Made by ChatGPT - Extracts data points separated by skip along the given axis
    :param data: ndarray, the data
    :param skip: int, the number of elements that are skiped when downsampling
    :param axis: int, the axis on which to downsample
    :param delete: bool, whether or not to delete the original data in order to save memory
    '''
    slices       = [slice(None)] * data.ndim
    slices[axis] = slice(None, None, downsample_rate)
    new_data     = data[tuple(slices)]
    
    print('         Orig. shape :', np.shape(data), "----> new shape :", np.shape(new_data))

    if delete:
        del(data)

    return new_data


def ExtractNumberfromFilename(filename):
    '''
    From ChatGPT
    :param filename, str: the name of the file
    '''
    numbers = []
    current_number = ""

    for i in range(len(filename)):
        if (((filename[i]).isdigit() and (filename[i+1].isdigit())) or ((filename[i]).isdigit() and (filename[i-1].isdigit()))):
            current_number += filename[i]
        
        elif current_number:
            numbers.append(int(current_number))
            current_number = ""

    if current_number:
        numbers.append(int(current_number))

    return int("".join(str(n) for n in numbers))


def FindFiles(path, target_phrases):
    correct_files = []
    for filename in os.listdir(path):
        bools = np.zeros(len(target_phrases))

        for i in range(len(target_phrases)):
            if (target_phrases[i] in filename):
                bools[i] = 1

        if reduce(lambda x, y: x*y, bools): # Check if all phrases were found
            correct_files.append(os.path.join(path, filename))

    return correct_files


def FlattenAx(data, axes):
    '''
    Flattens the specified axes of an array
    :param data: nparray, the data
    :param axes: tuple, MUST BE CONSECUTIVE, the axes that will be flattened together
    '''
    assert (len(axes) > 1), "Must have more than one axis"
    assert (data.ndim >= len(axes)), 'Not enough dimensions in data'

    orig_shape = np.shape(data)
    new_shape  = []

    i = 0
    while (i <= len(orig_shape) - 1):
        if i in axes:
            new_shape.append(int(orig_shape[i] * orig_shape[i+1]))
            i += 2
            #print(i, new_shape, orig_shape)
        else:
            new_shape.append(int(orig_shape[i]))
            i += 1
            #print(i, new_shape, orig_shape)
    
    return data.reshape(tuple(new_shape))