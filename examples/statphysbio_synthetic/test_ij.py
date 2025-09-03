import os
import imagej
import sys
import time
import scyjava

import imagej.doctor
imagej.doctor.checkup()

# print(imagej.__version__)

# try:
#     os.environ.setdefault("CLIJ_OPENCL_ALLOWED_DEVICE_TYPE", "CPU")
#     ij = imagej.init()
#     ij_success = True
# except:
#     time.sleep(.5)
#     ij_success = False

# print(ij_success)

# os.environ['JAVA_HOME'] = os.sep.join(sys.executable.split(os.sep)[:-2] + ['jre'])

# ij = imagej.init()