%matplotlib inline
from matplotlib.pyplot import *
from matplotlib import pyplot as plt
import numpy as np
from importlib import reload

import vip_hci as vip
from hciplot import plot_frames, plot_cubes

from vip_hci.config import VLT_NACO
from vip_hci.fm import normalize_psf, cube_inject_companions
from vip_hci.psfsub import median_sub, pca, pca_annular, pca_annulus

from vip_hci.fits import open_fits, write_fits, info_fits
from vip_hci.metrics import significance, snr, snrmap
from vip_hci.var import fit_2dgaussian, frame_center

from vip_hci.preproc.cosmetics import cube_crop_frames, frame_crop
from vip_hci.preproc import frame_rotate
from vip_hci.config.utils_conf import pool_map, iterable

import training_set_generation_functions
reload(training_set_generation_functions)
from training_set_generation_functions import inject_random_fake_comp, make_mlar_plus, make_mlar_minus, get_fwhm, evaluate_snr, plot_patch, flux_interval, patch_rotation, patch_shift, patch_average, save_to_h5, read_h5, make_residual_cube, plot_10_random

# //////////////////////////////////////////////////////////////////////////
# Importing adi sequence and calculating its psf
print('Importing adi sequence and calculating its psf')
print(' ')
# //////////////////////////////////////////////////////////////////////////

adi=np.load('adi_seq.npy')
pa=np.load('pa.npy')
psf=np.load('psf.npy')
pxscale=np.load('pxscale.npy')

fwhm = get_fwhm(psf)

# /////////////////////////////////////////////////////////////////////////////////////////////////
#data set parameters (annulus, numbre of sample, numbre of test injections for flux-snr statistics,
# numbre of principal components for mlar patches, snr interval we wanna work in)
print('data set parameters')
print(' ')
# /////////////////////////////////////////////////////////////////////////////////////////////////

an_nbr=5
an_radius = an_nbr*fwhm
snr_interv=(1,3)
ncomp= 3
npr=6  # numbre of processors for multiprocessing
sample_nbr=300
flux_stat=150

# /////////////////////////////////////////////////////////////////////////////////////////////////////
#
# C+ class generation
#
print('C+ class')
print(' ')
# //////////////////////////////////////////////////////////////////////////////////////////////////////


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# first fake companions injection, then evaluating snr for those injecctions, also keeping injections inside set snr interval
print('first fake companion injection')
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inj_adi=[]                 #store adi cubes that have been injected with a satisfying S/N value
good_values=[]             # S/N, posx, posy corrseponding to adi cubes in inj_adi
snr_values=[]
patches_plus=[]
adi_with_fake_comp, flevel = inject_random_fake_comp(adi, psf, pxscale, pa, fwhm, an_radius, inj_nbr=flux_stat, nproc=npr)

inj_pos=[(i[1][0][1],i[1][0][0]) for i in adi_with_fake_comp] # [(x,y),....] xy injection position

adi_with_fake_comp=[i[0] for i in adi_with_fake_comp]
snr_res = evaluate_snr(adi_with_fake_comp, pa, fwhm, inj_pos, an_radius, nproc=npr)  #[[posy, posx,flux, array(flux_appertures), s/r],....]

for i in range(len(snr_res)):
    if snr_res[i][4]>snr_interv[0] and snr_res[i][4]<snr_interv[1]:
        inj_adi.append(adi_with_fake_comp[i])
        good_values.append([snr_res[i][4], snr_res[i][1], snr_res[i][0]])

snr_values=[[i[4],i[2]] for i in snr_res]

# /////////////////////////////////////////////////////////////////////////////////////////////
# snr-flux statistics, determining which flux interval corresponds to snr_interv
print('flux statistics')
# /////////////////////////////////////////////////////////////////////////////////////////////

snr_values=np.array(snr_values)
flevel=np.array([flevel])
# scatter(snr_values[:,1],snr_values[:,0])
snr_values = np.concatenate((snr_values,flevel.T), axis=1)
flux_interv = flux_interval(snr_interv, snr_values)

# ///////////////////////////////////////////////////////////////////////////////////////////
# first crop for adi sequence with right snr for their fake companion injection
print('first crop') 
# ///////////////////////////////////////////////////////////////////////////////////////////
for i in range(len(inj_adi)):
    mlar=make_mlar_plus(inj_adi[i], ncomp, pa, fwhm, (good_values[i][1],good_values[i][2]), an_radius, plot=False)
    patches_plus.append(mlar)

# ////////////////////////////////////////////////////////////////////////////////////////////
# Generating more mlar patches based on snr-flux statistics
print('more mlar patches')
# ////////////////////////////////////////////////////////////////////////////////////////////

diff=sample_nbr-len(mlar)
chunk_size=500
chunk_nbr=diff//chunk_size
last_ones= diff - chunk_nbr*chunk_size

# generating only 'chunk_size' numbre of samples to avoid memory error in python pooling
for j in range(chunk_nbr):
    print('making 500')
    adi_with_fake_comp, flevel = inject_random_fake_comp(adi, psf, pxscale, pa, fwhm, an_radius, inj_nbr=chunk_size, flx=flux_interv, nproc=npr)
    inj_pos=[(i[1][0][1],i[1][0][0]) for i in adi_with_fake_comp]
    adi_with_fake_comp=[i[0] for i in adi_with_fake_comp]

    for i in range(len(adi_with_fake_comp)):
        mlar=make_mlar_plus(adi_with_fake_comp[i], ncomp, pa, fwhm, inj_pos[i], an_radius, plot=False)
        patches_plus.append(mlar)
        
#remaining samples to reach sample_nbr
print('last ones')
adi_with_fake_comp, flevel = inject_random_fake_comp(adi, psf, pxscale, pa, fwhm, an_radius, inj_nbr=chunk_size, flx=flux_interv, nproc=npr)
inj_pos=[(i[1][0][1],i[1][0][0]) for i in adi_with_fake_comp]
adi_with_fake_comp=[i[0] for i in adi_with_fake_comp]

for i in range(len(adi_with_fake_comp)):
    mlar=make_mlar_plus(adi_with_fake_comp[i], ncomp, pa, fwhm, inj_pos[i], an_radius, plot=False)
    patches_plus.append(mlar)

# /////////////////////////////////////////////////////////////////////////////////////////////////////
#
# C- class generation
#
print(' ')
print('C- class')
# //////////////////////////////////////////////////////////////////////////////////////////////////////

# /////////////////////////////////////////////////////////////////////////////////////////////////////
# Croping random patches inside annulus 
# ////////////////////////////////////////////////////////////////////////////////////////////////////

chunk_size=500
chunk_nbr=sample_nbr//chunk_size
last_ones= sample_nbr - chunk_nbr*chunk_size

residual_cube=make_residual_cube(adi, pa, ncomp, fwhm, an_radius)

patches_minus=[]
for i in range(chunk_nbr):
    patches_minus+=make_mlar_minus(residual_cube, fwhm, an_radius, chunk_nbr, nproc=npr)
    
if last_ones!=0:
    patches_minus+=make_mlar_minus(residual_cube, fwhm, an_radius, last_ones, nproc=npr)
    

# /////////////////////////////////////////////////////////////////////////////////////////////////////
#
# Data augmentation
#
print(' ')
print('data augmentation')
# //////////////////////////////////////////////////////////////////////////////////////////////////////

patches_minus_aug=[]
patches_plus_aug=[]

# /////////////////////////////
# Shift
print('shift')
# //////////////////////////////

patches_minus_aug+=patch_shift(patches_minus, nproc=npr)
    
patches_plus_aug+=patch_shift(patches_plus, nproc=npr)

# /////////////////////////////
# Rotation
print('rotation')
# //////////////////////////////

patches_minus_aug+=patch_rotation(patches_minus, nproc=npr)
patches_plus_aug+=patch_rotation(patches_plus, nproc=npr)

# /////////////////////////////
# Average
print('average')
# //////////////////////////////

patches_minus_aug+=patch_average(patches_minus, nproc=npr)

patches_plus_aug+=patch_average(patches_plus, nproc=npr)


# /////////////////////////////////////////////////////////////////////////////////////////////////////
#
# Saving to H5 file
#
print(' ')
print('Saving data set')
# //////////////////////////////////////////////////////////////////////////////////////////////////////

save_to_h5(an_nbr, {'patches_plus':patches_plus, 'patches_plus_aug':patches_plus_aug, 'patches_minus':patches_minus, 'patches_minus_aug':patches_minus_aug})