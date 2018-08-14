import os


def clean_path(path):
    return path.strip('/')


def make_dir(dir_name):
    import errno

    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise



