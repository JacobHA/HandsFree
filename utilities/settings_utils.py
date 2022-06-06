def read_index(filename):

    with open(filename, 'r') as f:
        
        idx_str = f.read()

        if idx_str == '':
            idx = 0
            print('No index found. Using default value (0).')


        else:
            idx = int(idx_str)
            print(f'Using chosen value ({idx}) for camera index.')


    return idx

