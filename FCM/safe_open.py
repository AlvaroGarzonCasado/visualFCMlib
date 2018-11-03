import os
import itertools


def iter_incrementing_file_names(path):

    yield path
    prefix, ext = os.path.splitext(path)

    for i in itertools.count(start=1, step=1):

        yield prefix + ' ({0})'.format(i) + ext


def get_safe_name(path):

    for filename in iter_incrementing_file_names(path):

        if os.path.isfile(filename):
            pass

        else:
            return filename


def safe_open(path, mode):

    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY

    if 'b' in mode and platform.system() == 'Windows':
        flags |= os.O_BINARY

    for filename in iter_incrementing_file_names(path):

        try:
            file_handle = os.open(filename, flags)

        except OSError as e:

            if e.errno == errno.EEXIST:
                pass

            else:
                raise

        else:
            return os.fdopen(file_handle, mode)