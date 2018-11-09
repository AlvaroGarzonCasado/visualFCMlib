"""
This program is free software: you can redistribute it and/or modify it under 
the terms of the GNU General Public License as published by the Free Software 
Foundation, either version 3 of the License, or (at your option) any later 
version.

This program is distributed in the hope that it will be useful, but WITHOUT 
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with 
this program. If not, see http://www.gnu.org/licenses/

Author: Pablo Cano Marchal

"""

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