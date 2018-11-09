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

import logging
import time
from functools import wraps
import os

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M',
                    filename='fcm.log')

console = logging.StreamHandler()
console.setLevel(logging.INFO)

formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')

console.setFormatter(formatter)

logging.getLogger('').addHandler(console)

formatter2 = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
lf = logging.FileHandler(os.path.join(os.getcwd(), 'time.log'))
lf2 = logging.FileHandler(os.path.join(os.getcwd(), 'error_ev.log'))
lf3 = logging.FileHandler(os.path.join(os.getcwd(), 'visual_lib.log'))
print os.getcwd()
lf.setLevel(logging.DEBUG)
lf.setFormatter(formatter2)
lf2.setLevel(logging.INFO)
lf2.setFormatter(formatter2)
lf3.setLevel(logging.INFO)
lf3.setFormatter(formatter2)
logging.getLogger('').addHandler(lf)
logging.getLogger('').addHandler(lf2)
logging.getLogger('').addHandler(lf3)


logger = logging.getLogger(__name__)


def timethis(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info('%s - Time elapsed %s,', func.__name__, end - start)
        return result

    return wrapper


def log(func):


    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug('Entering %s', func.__name__)
        result = func(*args, **kwargs)
        logger.debug('Exiting %s', func.__name__)
        return result

    return wrapper
