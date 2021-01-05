import platform
if platform.system() == 'Windows':
    print('Running on Windows. Parallel dataloader disabled.')
    num_workers = 0
else:
    print('Running on Linux. Parallel dataloader enabled.')
    num_workers = 4
