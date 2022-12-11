
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


adi=np.load('adi_seq.npy')
pa=np.load('pa.npy')
psf=np.load('psf.npy')
pxscale=np.load('pxscale.npy')

fwhm = 3.7684223732796815     #issue with get_fwhm....
#plt.close()
an_nbr=9

for i in range(30):
    an_radius = an_nbr*fwhm
    print('start annulus: '+str(an_nbr))
    snr_interv=(1,3)
    ncomp= 3
    sample_nbr=30
    flux_stat=500
    
    inj_adi=[]                 #store adi cubes that have been injected with a satisfying S/N value
    good_values=[]             # S/N, posx, posy corrseponding to adi cubes in inj_adi
    snr_values=[]
    patches_plus=[]
    adi_with_fake_comp, flevel = inject_random_fake_comp(adi, psf, pxscale, pa, fwhm, an_radius, inj_nbr=flux_stat, nproc=6)
    print('done injection')
    inj_pos=[(i[1][0][1],i[1][0][0]) for i in adi_with_fake_comp] # [(x,y),....] xy injection position
    
    adi_with_fake_comp=[i[0] for i in adi_with_fake_comp]
    snr_res = evaluate_snr(adi_with_fake_comp, pa, fwhm, inj_pos, an_radius, nproc=6)  #[[posy, posx,flux, array(flux_appertures), s/r],....]
    print('done snr')
    #snr_values.append([snr_vale, f_source, sourcex, sourcey])
    for i in range(len(snr_res)):
        if snr_res[i][4]>snr_interv[0] and snr_res[i][4]<snr_interv[1]:
            inj_adi.append(adi_with_fake_comp[i])
            good_values.append([snr_res[i][4], snr_res[i][1], snr_res[i][0]])
    
    snr_values=[[i[4],i[2]] for i in snr_res]
    print('start statisctics')
    snr_values=np.array(snr_values)
    flevel=np.array([flevel])
    scatter(snr_values[:,1],snr_values[:,0])
    plt.savefig(str(an_nbr)+'L_D_500inj_from0_to30')
    plt.close()
    print('done statistics and save image')
    snr_values = np.concatenate((snr_values,flevel.T), axis=1)
    flux_interv = flux_interval(snr_interv, snr_values)
    
    with open('flux_stat_results.txt', 'a') as f:
        f.write(str(an_nbr)+'L_D:['+str(flux_interv[0])+', '+str(flux_interv_[1])+']\n')
        f.close()
    an_nbr+=1
    print('donefile writting')
