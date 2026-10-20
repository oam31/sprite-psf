=====================================================================================
                      SPRITE - SParse Recovery of Instrumental rEsponse          
                                Library and tools                          
                                  v1.0  09/2014                           
=====================================================================================


SPRITE is a C++ package providing a super-resolution algorithm optimized for compact source images.

SPRITE is distributed under the CeCILL License (see LICENSE.CeCILL) in the hope that it 
will be usefull but without any warranty. 

This package contains a sample of 4 low resolution images derived from a zemax simulation of Euclid telescope optical PSF in the file cube_test.fits. 

Author: Ngolè Fred (ngolefred@yahoo.fr)

Copyright CEA 2014

===================
   Requirements
===================

In order to be able to compile the SPRITE package, the following softwares are required:

    - cmake 	     (http://cmake.org/cmake/resources/software.html)
    - cfitsio        (http://heasarc.gsfc.nasa.gov/fitsio/)
    - gcc
    - gsl      	 (http://www.gnu.org/software/gsl/)

The following softwares are optional:
    - IDL 
    - ipython    

===================
     Building
===================

Compile the package:

    $ cd build
    $ cmake ..
    $ make all
    $ make install  

===================
 Running the code
===================

The main executable is run_sprite located in the directory bin.
To get the program options, type under a terminal: ./run_sprite
A typical example of call is given in the IDL script example.pro and the python script example.py.
