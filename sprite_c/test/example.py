# NAME:
#         example.py
#
# PURPOSE:
#         call SPRITE exe and display outputs
#
# CALLING:
#         example


import subprocess
import os
import pyfits as fits
import matplotlib.pyplot as plt

data_path = "./"
outputs_path = "sprite_outputs/"
opt="-v -t24 -s4 -r2 -F -N -w resi.fits"
datafile = data_path+ "cube_test.fits"
output_image = "solution.fits"
subprocess.call(["../bin/run_sprite","-v","-t2","-s4","-r2","-F","-N","-w", "resi.fits",datafile,output_image,outputs_path])

solution = fits.getdata(outputs_path+output_image)

plt.imshow(solution)
plt.show()