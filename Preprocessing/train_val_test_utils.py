import numpy as np
import os
import extraction_funcs
import misc

def Split(data, i, hist_period_len, days_to_predict, skip):
    '''
    This separates a chunk of data into a historical period and a future period. The time points in the historical period
    may be separated by "skip" in order to allow a deeper look into the past without sacrificing memory
    :param data:, nparray, the full dataset to be split
    :param i: int, the location to begin the historical period
    :param hist_period_len: int, how far forward to move from the starting point i
    :param days_to_predict: int, how many consecutive timesteps after hist_period_len the model should predict
    :param skip: int, how many days are skipped between timesteps in the historical period, must divide hist_period_len
    '''
    # Assume data has shape (time, space)
    if (data.ndim > 1):
        #print(np.shape(data[:, i:(i + hist_period_len):skip]), np.shape(data[:, (i + hist_period_len):(i + hist_period_len + days_to_predict)]))
        return data[i:(i + hist_period_len):skip, ...], data[(i + hist_period_len):(i + hist_period_len + days_to_predict), ...]
    else:
        #print(np.shape(data[i:(i + hist_period_len)]), np.shape(data[(i + hist_period_len):(i + hist_period_len + days_to_predict)]))
        return data[i:(i + hist_period_len):skip], data[(i + hist_period_len):(i + hist_period_len + days_to_predict)]
    

def Hist_and_Target(data, hist_period_len, days_to_predict, skip, shuf=False, shuffle_idx=None):
    '''
    This creates sequences of historical periods and future targets on which to train and test the model
    :param data:, nparray, The full dataset to be split
    :param i: int, The location to begin the historical period
    :param hist_period_len: int, How far forward to move from the starting point i
    :param days_to_predict: int, How many consecutive timesteps after hist_period_len the model should predict
    :param skip: int, How many days are skipped between timesteps in the historical period, must divide hist_period_len
    :param shuf: bool, If true, this will shuffle the order of the sequences, not the individual days inside them
    :param shuffle_idx:, 1-D nparray, This is an array of integers, 0 - number of sequences, which must be randomly shuffled
                                     and used to shuffle the order of the sequences
    '''
    if (data.ndim > 1):
        # Assume data has dimensions (time, space)
        data_shape = np.shape(data)


        shape_h = (np.shape(data)[0] - hist_period_len - days_to_predict,) + (int(hist_period_len / skip),) + np.shape(data)[1:]
        #print("data_shape =", data_shape, ", shape_h =", shape_h)
        shape_t = (np.shape(data)[0] - hist_period_len - days_to_predict,) + (days_to_predict,) + np.shape(data)[1:]
        #print("data_shape =", data_shape, ", shape_t =", shape_t)

        histories = np.ones(shape_h, dtype=float)
        targets = np.ones(shape_t, dtype=float)

        ''' Go sequential through the data to extract the histories and targets '''
        for i in range(np.shape(data)[0] - hist_period_len - days_to_predict):
            h, t = Split(data, i, hist_period_len, days_to_predict, skip)
            #print("i =", i, ", np.shape(data) =", np.shape(data), ", shape_h =", shape_h, ", h =", np.shape(h), ", shape_t =", shape_t, ", t =", np.shape(t))

            histories[i, ...], targets[i, ...] = h, t
            
            
        ''' Shuffle the individual histories and targets so they aren't in sequential order '''
        if (shuf == True):
            histories = [histories[i, ...] for i in shuffle_idx]
            targets   = [targets[i, ...] for i in shuffle_idx]
            
            return np.array(histories), np.array(targets)

        else:
            return np.array(histories), np.array(targets)
        
    else:
        histories = np.ones((np.shape(data)[0] - hist_period_len - days_to_predict,
                             int(hist_period_len / skip)),
                             dtype=float)
        targets = np.ones((np.shape(data)[0] - hist_period_len - days_to_predict,
                           days_to_predict),
                           dtype=float)

        ''' Go sequential through the data to extract the histories and targets '''
        for i in range(np.shape(data)[0] - hist_period_len - days_to_predict):
            histories[i, :], targets[i, :] = Split(data, i, hist_period_len, days_to_predict, skip)

        ''' Shuffle the individual histories and targets so they aren't in sequential order '''
        if (shuf == True):
            histories = [histories[i, :] for i in shuffle_idx]
            targets   = [targets[i, :] for i in shuffle_idx]
            return np.array(histories), np.array(targets)

        else:
            return np.array(histories), np.array(targets)


def CombineLocations(loc_names, folder_name, var_name, orig_shape, total_histories, 
                     total_targets, base_path, fire_pix_val,
                     hist_period_len, days_to_predict, skip,
                     shuffle=False, shuffle_idx=None, downsample_rate=1):
    '''
    This is where all of the preoprocessing functions are employed. The particular variable determined by folder_name and var name
    from each of the locations in loc_names is extracted, scaled, downsampled, and turned into sequences of histories and targets.
    The histories and targets from all of the locations in loc_names are concatenated into the total_histories and total_targets arrays respectively.
    :param loc_names: list of str, The names of folders for each location of the data you want to preprocess
    :param folder name: str, The name of the folder which contains the variable you want
    :param var_name: str, The name of the variable in that folder you want to preprocess
    :param orig_shape: tuple, The unflattened shape of the .nc files, (time, spat_dim1, spat_dim2)
    :param total_histories: ndarray, Must have dimensions (365 - hist_period_len - days_to_predict, hist_period_len, len(loc_names) * number of spatial samples in data).
    :param total_targets: ndarray, Must have dimensions (365 - hist_period_len - days_to_predict, days_to_predict, len(loc_names) * number of spatial samples in data)
    :param base_path: str, The absolute path to the folder that contains all of the data
    :param fire_pix_val: float, The value that every MaxFRP value from the MODIS fire .tif files will be set to if that MaxFRP value is greater than one
    :param hist_period_len: int, How far forward to move from the starting point i
    :param days_to_predict: int, How many consecutive timesteps after hist_period_len the model should predict
    :param skip: int, How many days are skipped between timesteps in the historical period, must divide hist_period_len
    :param shuffle: bool, If true, this will shuffle the order of the histories and targets created by Hist_and_Target according to shuffle_idx
    :param shuffle_idx: (1-D ndarray), Must be a permuation of integers 0 - np.shape(histories)[0]. Dictates how the histories and targets arrays will be shuffled
    :param downsample_rate: int, Must divide the len of the axis on which it's called. By what factor the data's resolution is reduced.
                                 1 MEANS NO DOWNSAMPLING, 2 means take every other element, 3 etc.
    '''
    #======================================================================================================================
    #======================================================================================================================
    #======================================================================================================================
    if (folder_name == 'MODIS_Fire'):
        print("############### loc_idx =", 0, loc_names[0], "###############")
        # --------------------------------------------------------------------------
        path = os.path.join(base_path, os.path.join(loc_names[0], folder_name))
        # --------------------------------------------------------------------------
        var_big_flat = extraction_funcs.ExtractFire(path, orig_shape, var_name, make_binary=True, max_val=fire_pix_val)

        #print('var_big_flat =', np.shape(var_big_flat), ', orig_shape =', orig_shape)
        #var_big = var_big_flat.reshape(orig_shape[:-1] + (-1,))
        var_big = var_big_flat.reshape((np.shape(var_big_flat)[0],) + orig_shape[1:])
        #print('var_big =', np.shape(var_big), ', orig_shape =', orig_shape)
        # --------------------------------------------------------------------------
        var = misc.MaxPool2D(var_big, pool_size=(downsample_rate, downsample_rate))

        # --------------------------------------------------------------------------
        var_flat = misc.FlattenAx(var, (1, 2))

        # --------------------------------------------------------------------------
        var_sum = np.sum(var_flat, axis=-1)

        # --------------------------------------------------------------------------
        print('Is MODIS :', orig_shape, np.shape(var_big_flat), np.shape(var_big_flat), np.shape(var), np.shape(var_flat), np.shape(var_sum))
        # --------------------------------------------------------------------------
        for loc_idx in range(1, len(loc_names)):
            print("############### loc_idx =", loc_idx, loc_names[loc_idx], "###############")
            # --------------------------------------------------------------------------
            path = os.path.join(base_path, os.path.join(loc_names[loc_idx], folder_name))
            # --------------------------------------------------------------------------
            var_big_flat = extraction_funcs.ExtractFire(path, orig_shape, var_name, make_binary=True, max_val=fire_pix_val)

            #print('var_big_flat =', np.shape(var_big_flat), ', orig_shape =', orig_shape)
            #var_big = var_big_flat.reshape(orig_shape[:-1] + (-1,))
            var_big = var_big_flat.reshape((np.shape(var_big_flat)[0],) + orig_shape[1:])
            #print('var_big =', np.shape(var_big), ', orig_shape =', orig_shape)
            # --------------------------------------------------------------------------
            var = misc.MaxPool2D(var_big, pool_size=(downsample_rate, downsample_rate))

            # --------------------------------------------------------------------------
            var_flat = misc.FlattenAx(var, (1, 2))

            # --------------------------------------------------------------------------
            var_sum += np.sum(var_flat, axis=-1)
            # --------------------------------------------------------------------------
            print('Is MODIS :', orig_shape, np.shape(var_big_flat), np.shape(var_big_flat), np.shape(var), np.shape(var_flat), np.shape(var_sum))
            # --------------------------------------------------------------------------
        var_bi = np.zeros_like(var_sum)
        var_bi[var_sum > 0] = 1

        histories, targets = Hist_and_Target(var_bi, hist_period_len, days_to_predict, skip, shuffle, shuffle_idx)
        print("~~~~~~~IN CombineLocations~~~~~~~~~histories =", np.shape(histories), ", targets =", np.shape(targets))       
        total_histories[:, :] = histories
        total_targets[:, :] = targets
    #======================================================================================================================
    #======================================================================================================================
    #======================================================================================================================
    else: 
        for loc_idx in range(0, len(loc_names)):
            print("############### loc_idx =", loc_idx, loc_names[loc_idx], "###############")

            path = os.path.join(base_path, os.path.join(loc_names[loc_idx], folder_name))
        
            filenames = FindFiles(path, ['krig_grid', var_name])

            for filename in filenames:
                print('filename :', filename)

                if (filename.endswith('.nc') or filename.endswith('.xlsx') or filename.endswith('.csv')):
                    
                    if (filename.endswith('.nc')):
                        lon_big, lat_big, var_big = extraction_funcs.Extract_netCDF4(filename, var_name)
                        # --------------------------------------------------------------------------
                        lon = misc.DownSample(misc.DownSample(lon_big, downsample_rate, 0), downsample_rate, 1)
                        lat = misc.DownSample(misc.DownSample(lat_big, downsample_rate, 0), downsample_rate, 1)
                        var = misc.DownSample(misc.DownSample(var_big, downsample_rate, 1), downsample_rate, 2)
                        # --------------------------------------------------------------------------
                        var_flat = misc.FlattenAx(var, (1, 2))

                        print('Is NC :', np.shape(var_big), np.shape(var), np.shape(var_flat))
                        # --------------------------------------------------------------------------
                        histories, targets = Hist_and_Target(var, hist_period_len, days_to_predict, skip, shuffle, shuffle_idx)
                        print("~~~~~~~IN CombineLocations~~~~~~~~~histories =", np.shape(histories), ", targets =", np.shape(targets), ', loc_idx =', loc_idx)
                        total_histories[:, :, (loc_idx * np.shape(histories)[2]) : ((loc_idx+1) * np.shape(histories)[2]), :] = histories
                        total_targets[:, :, (loc_idx * np.shape(targets)[2]) : ((loc_idx+1) * np.shape(targets)[2]), :] = targets

                    elif ((folder_name == 'Land_Fire') and filename.endswith('.csv')):
                        var_big_flat = extraction_funcs.ExtractLandFire(filename, orig_shape, var_name)
                        #print('var_big_flat =', np.shape(var_big_flat), ', orig_shape =', orig_shape)

                        var_big = var_big_flat.reshape((np.shape(var_big_flat)[0],) + orig_shape[1:])
                        #var_big = var_big_flat.reshape(orig_shape[:-1] + (-1,))
                        #print('var_big =', np.shape(var_big), ', orig_shape =', orig_shape)
                        # --------------------------------------------------------------------------
                        var = misc.DownSample(misc.DownSample(var_big, downsample_rate, 1), downsample_rate, 2)
                        #print('var =', np.shape(var))
                        # --------------------------------------------------------------------------
                        var_flat = misc.FlattenAx(var, (1, 2))

                        print('Is LandFire :', orig_shape, np.shape(var_big_flat), np.shape(var_big_flat), np.shape(var), np.shape(var_flat))
                        # --------------------------------------------------------------------------
                        histories, targets = Hist_and_Target(var, hist_period_len, days_to_predict, skip, shuffle, shuffle_idx)
                        print("~~~~~~~IN CombineLocations~~~~~~~~~histories =", np.shape(histories), ", targets =", np.shape(targets), ', loc_idx =', loc_idx)
                        total_histories[:, :, (loc_idx * np.shape(histories)[2]) : ((loc_idx+1) * np.shape(histories)[2]), :] = histories
                        total_targets[:, :, (loc_idx * np.shape(targets)[2]) : ((loc_idx+1) * np.shape(histories)[2]), :] = targets

                    else:
                        print(filename, ' not found')
                    
                else:
                    print(filename, ' is invalid file type, valid : (.nc, .csv, .xlsx)')


def CombineLocations_lonlattime(loc_names_train, folder_name, var_name, orig_shape, 
                                total_histories, total_targets, base_path,
                                hist_period_len, days_to_predict, skip,
                                shuffle=False, shuffle_idx=None, downsample_rate=1):
    '''
    This is where all of the preoprocessing functions are employed. The particular variable determined by folder_name and var name
    from each of the locations in loc_names_train is extracted, scaled, downsampled, and turned into sequences of histories and targets.
    The histories and targets from all of the locations in loc_names_train are concatenated into the total_histories and total_targets arrays respectively.
    :param loc_names_train: list of str, The names of folders for each location of the data you want to preprocess
    :param folder name: str, The name of the folder which contains the variable you want
    :param var_name: str, The name of the variable in that folder you want to preprocess
    :param orig_shape: tuple, The unflattened shape of the .nc files, (time, spat_dim1, spat_dim2)
    :param total_histories: ndarray, The first dimension must be len(loc_names_train) * histories for one variable. Where the sequences of histories
                                     for the varaible in all the locations in loc_names_train are kept.
    :param total_targets: ndarray, Same as total_histories but for the targets
    :param base_path: str, The absolute path to the folder that contains all of the data
    :param hist_period_len: int, How far forward to move from the starting point i
    :param days_to_predict: int, How many consecutive timesteps after hist_period_len the model should predict
    :param skip: int, How many days are skipped between timesteps in the historical period, must divide hist_period_len
    :param shuffle: bool, If true, this will shuffle the order of the histories and targets created by Hist_and_Target
    :param shuffle_idx: (1-D ndarray), Must be a permuation of integers 0 - np.shape(histories)[0]. Dictates how the histories and targets arrays will be shuffled
    :param downsample_rate: int, Must divide the len of the axis on which it's called. By what factor the data's resolution is reduced.
                                 1 MEANS NO DOWNSAMPLING, 2 means take every other element, 3 etc.
    '''
    for loc in range(0, len(loc_names_train)):
        path = os.path.join(base_path, os.path.join(loc_names_train[loc], folder_name))

        tmax_filenames = misc.FindFiles(path, ['krig_grid', 'tmax'])

        for tmax_filename in tmax_filenames:
            print('tmax_filename :', tmax_filename)
   
            if (tmax_filename.endswith('.nc')):

                tmax_lon_big, tmax_lat_big, tmax_big = extraction_funcs.Extract_netCDF4(tmax_filename, 'tmax')
                tmax_flat_big = misc.FlattenAx(tmax_big, (1, 2))

                tmax_lon = misc.DownSample(misc.DownSample(tmax_lon_big, downsample_rate, 0), downsample_rate, 1)
                tmax_lat = misc.DownSample(misc.DownSample(tmax_lat_big, downsample_rate, 0), downsample_rate, 1)
                tmax = misc.DownSample(misc.DownSample(tmax_big, downsample_rate, 1), downsample_rate, 2)
                tmax_flat = misc.FlattenAx(tmax, (1, 2))

                orig_shape_big = np.shape(tmax_big)
                flat_shape_big = np.shape(misc.FlattenAx(tmax_big, (1, 2)))

                orig_shape = np.shape(tmax)
                flat_shape = np.shape(tmax_flat)

                print('orig_shape_big =', orig_shape_big, ', flat_shape_big =', flat_shape_big, ', orig_shape =', orig_shape, ', flat_shape =', flat_shape)

                lon_flat = tmax_lon.ravel()
                lat_flat = tmax_lat.ravel()

                time = np.arange(0, orig_shape[0])

                lonlat_flat = np.array([lon_flat, lat_flat]).T

                lonlattime_flat = np.zeros((np.shape(time)[0],
                                           np.shape(lonlat_flat)[0],
                                           3), dtype=float)
                print('time =', np.shape(time), ', lonlat_flat =', np.shape(lonlat_flat), ', lonlattime_flat =', np.shape(lonlattime_flat))

                lonlat_flat_tiled = np.tile(np.expand_dims(lonlat_flat, axis=0), (np.shape(time)[0], 1, 1))
                print('lonlat_flat_tiled =', np.shape(lonlat_flat_tiled))

                lonlattime_flat[..., :2] = lonlat_flat_tiled

                for i in range(np.shape(lonlattime_flat)[0]):
                    for j in range(np.shape(lonlattime_flat)[1]):
                            lonlattime_flat[i][j][2] = time[i]
                
                lonlattime = np.reshape(lonlattime_flat, (np.shape(lonlattime_flat)[0],) + orig_shape[1:] + (np.shape(lonlattime_flat)[2],))
                print('===========================')
                print(np.shape(lonlattime))
                print(lonlattime[69, 0, 0, :])
                print(lonlattime[60, 0, 0, :])
                print(lonlattime[69, 10, 0, :])
                print(lonlattime[69, 0, 10, :])
                lonlattime = np.reshape(lonlattime_flat, (np.shape(lonlattime_flat)[0],) + orig_shape[1:] + (np.shape(lonlattime_flat)[2],))

                histories, targets = Hist_and_Target(lonlattime, hist_period_len, days_to_predict, skip, shuffle, shuffle_idx)
                print("~~~~~~~IN CombineLocations_lonlattime~~~~~~~~~histories =", np.shape(histories), ", targets =", np.shape(targets), ', loc =', loc)

                total_histories[:, :, (loc * np.shape(histories)[2]) : ((loc+1) * np.shape(histories)[2]), :, :] = histories
                total_targets[:, :, (loc * np.shape(targets)[2]) : ((loc+1) * np.shape(histories)[2]), :, :] = targets


            else:
                print(tmax_filename, ' is invalid file type, valid : .nc')