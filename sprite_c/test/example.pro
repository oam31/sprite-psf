; NAME: 
;         example.pro
;
; PURPOSE: 
;         call SPRITE exe and display outputs
;
; CALLING:
;         example


data_path = './'
outputs_path = 'sprite_outputs/'
opt=' -t 24 -s 4 -r 2 -F -N -w resi.fits -v '
datafile = data_path+ 'cube_test.fits solution.fits '+outputs_path
com = '../bin/run_sprite '+  opt + datafile
spawn,com

solution = readfits(outputs_path+'solution.fits')

loadct,5
tvscl,solution

end
