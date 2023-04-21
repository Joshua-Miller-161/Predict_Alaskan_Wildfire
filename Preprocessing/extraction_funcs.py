def Extract_netCDF4(path, var_name, scale=True):
    '''
    This will load lon, lat, and data from an nc file, and transpose it so the time dimension in the data is last
    :param path: str, path to the nc file
    :param var_name: str, the name of the variable to be extracted
    '''
    if path.endswith('nc'):
        f = nc.Dataset(path, "r")
        print(f.dimensions.keys())

        # Print the variables in the file
        print(f.variables.keys())

        lon = f.variables['lon']
        lon = np.array(lon[:])

        lat = f.variables['lat']
        lat = np.array(lat[:])

        var = f.variables[var_name]
        var = np.array(var[:])
        
        f.close()

        if scale:
            val = Scale(var, var)

        return lon, lat, var

    else:
        print("ERROR :", path, 'must be .nc')


def ExtractLandFire(path, target_arr_shape, var_name, scale=True):
    '''
    Gets the data from Land_Fire folder
    :param path: str, the full path to the file
    :param target_arr_shape: tuple, The shape all of the data should be in. The first dimension (time) is used to tile the LandFire data
    :param var_name: str, the name of the column in the file you want to pull
    '''
    if path.endswith('.csv'):
        df = pd.read_csv(path)


        #print('df.head() =', df.head())
        var = df[var_name].values
        
        if scale:
            print('SCALING')
            var = Scale(var, var)  #.reshape(np.shape(mod_fire)[:2])
                
        var_tiled = np.tile(var, (target_arr_shape[0], 1))
        print('Pulling ', path, ', var =', np.shape(var), ', var_tiled =', np.shape(var_tiled), ', target_shape =', target_arr_shape)

        return var_tiled

    else:
        print('ERROR :', path, 'must be .csv')


def ExtractFire(path, target_arr_shape, var_name, make_binary=False, max_val=1, scale=True):
    '''
    Searches through fire_path for .xlsx files, extracts their data, and places them into an array according to the
    index given in their filename
    :param path: str, Path to the folder where the .xlsx files are
    :param target_arr_shape: tuple, Used to calculate the shape of the array into which the fire data will be stored
    :param var_name: str, the name of the varaible you want to get
    :param make_binary: bool, If true, this will make every values greater than 0 equal to max_val
    :param max_val: int,
    :param scale: bool, If true, this will perform MaxAbsScaling on the data
    returns: np.array (spatial_dims, time_dim)
    '''

    mod_fire = np.zeros((target_arr_shape[0], target_arr_shape[1] * target_arr_shape[2]))
    
    for filename in os.listdir(path):
        if (filename.endswith('.xlsx')):
            day_idx = ExtractNumberfromFilename(filename)

            curr_var = pd.read_excel(os.path.join(path, filename))[var_name].values

            if make_binary:
                curr_var = [max_val if x > 0 else 0 for x in curr_var.ravel()]
                #curr_mod_fire = np.array(curr_mod_fire).reshape(target_arr_shape[0], target_arr_shape[1])

            elif scale:
                curr_var = Scale(curr_var, 'maxabs')

            print('Pulling ', filename, ", target_arr_shape =", target_arr_shape, ", mod_fire =", np.shape(mod_fire), ", curr_var =", np.shape(curr_var), ', sum = ', np.sum(curr_var))

            mod_fire[day_idx, ...] = curr_var

    return mod_fire