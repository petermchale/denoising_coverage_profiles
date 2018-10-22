import os


def make_dir(dir_name):
    import errno

    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def append_log_file(log_file_name, string):
    """ Write diagnostic information to a log file"""

    with open(log_file_name, 'a') as flog:
        flog.write(string)
        flog.flush()  # flush the program buffer
        os.fsync(flog.fileno())  # flush the OS buffer


import errno


def silent_remove(filename):
    try:
        os.remove(filename)
    except OSError as e:  # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
            raise  # re-raise exception if a different error occurred


def down_sample(data, number_samples=1000):
    if number_samples < len(data):
        return data.sample(n=number_samples).sort_values('start')
    else:
        return data


import json


def get_args(trained_model_path):
    with open(os.path.join(trained_model_path, 'args.json'), 'r') as fp:
        return json.load(fp)


def named_tuple(dictionary):
    from collections import namedtuple
    return namedtuple('Struct', dictionary.keys())(*dictionary.values())
