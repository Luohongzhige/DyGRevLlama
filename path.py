import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
WORK_PATH = os.path.join(BASE_PATH, 'work')
DATA_PATH = os.path.join(BASE_PATH, 'data')
TEMP_PATH = os.path.join(BASE_PATH, 'temp')
LOG_PATH = os.path.join(BASE_PATH, 'log')

def path_change(source, target):
    # add target to source
    exec('global {}; {} = os.path.join({}, target)'.format(source, source, source))

def create_folder(path, force_new = False):
    if force_new and os.path.exists(path):
            os.rmdir(path)
    if not os.path.exists(path):
        os.makedirs(path)

create_folder(DATA_PATH)
create_folder(WORK_PATH)
create_folder(LOG_PATH)
create_folder(TEMP_PATH, force_new=True)