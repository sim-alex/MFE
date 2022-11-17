from matplotlib.pyplot import *
from matplotlib import pyplot as plt
import numpy as np
import copy
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
from vip_hci.preproc.recentering import frame_shift




def inject_random_fake_comp(adi, psf, pxscale, pangles, fwhm, rad_dists, flx='random'):
    
    norm_psf = normalize_psf(psf, fwhm=fwhm, model='gauss', imlib='opencv', interpolation='bicubic')       #Normalizing PSF
    if not flx == 'random':
        flevel= np.random.randint(flx[0], high=flx[1])
    else:
        flevel= np.random.randint(0, high=110)
    theta = np.random.rand()*360 #in degrees
    
    adi_with_fake_comp, inj_posyx = cube_inject_companions(adi, psf, pangles, flevel=flevel, plsc=pxscale, rad_dists=rad_dists, theta=theta, imlib='opencv', interpolation='bicubic', full_output=True)
    inj_posyx=inj_posyx[0]
    inj_posxy=(inj_posyx[1], inj_posyx[0])

    return adi_with_fake_comp, inj_posxy

def make_mlar_plus(adi_with_fake_comp, ncomp, pangles, fwhm, inj_posxy, an_radius, plot=False):
    MLAR=[]
    patch_width = np.ceil(2*fwhm) // 2 * 2 + 1    # rounding to the nearest odd integer
    patch_width = int(patch_width)
    rot_options={'imlib':'skimage', 'interpolation':'bicubic'}

    for i in range(1, ncomp+1):
        adi_pca = pca_annulus(adi_with_fake_comp, pangles, ncomp, 2*fwhm, an_radius, cube_ref=None, svd_mode='lapack', scaling=None, collapse='median', weights=None, collapse_ifs='mean', **rot_options)
        MLAR.append(frame_crop(adi_pca, patch_width, cenxy=inj_posxy, verbose=False, force=True))
    
    if plot:
        plot_frames(tuple(MLAR[i] for i in range(ncomp)))
        
    return MLAR
    
def get_fwhm(psf):

    DF_fit = fit_2dgaussian(psf, crop=True, cropsize=9, debug=True, full_output=True)
    fwhm = np.mean([DF_fit['fwhm_x'],DF_fit['fwhm_y']])   #using gaussian fit to get FWHM
    
    return fwhm
    
def evaluate_snr(adi_with_fake_comp, pangles, fwhm, inj_posxy, an_radius):
    
    residual= median_sub(adi_with_fake_comp, pangles, fwhm, asize=2*fwhm, mode='annular', delta_rot=0.5, radius_int= an_radius, nframes=4, imlib='skimage', interpolation='bicubic', verbose=False)
    
    sourcey, sourcex, f_source, fluxes, snr_vale = snr(residual, source_xy = inj_posxy, fwhm=fwhm, plot=False, full_output=True, verbose=False )
    
    return snr_vale, f_source, sourcex, sourcey
    
def plot_patch(mlar):
    plot_frames(tuple(mlar[i] for i in range(len(mlar))))
    return

def make_mlar_minus(adi, ncomp, pangles, fwhm, an_radius, sample_nbr):
    sample_set=[]
    patch_width = np.ceil(2*fwhm) // 2 * 2 + 1    # rounding to the nearest odd integer
    patch_width = int(patch_width)
    rot_options={'imlib':'skimage', 'interpolation':'bicubic'}

    residual_cube=[]
    for i in range(1, ncomp+1):
        adi_pca = pca_annulus(adi, pangles, ncomp, 2*fwhm, an_radius, cube_ref=None, svd_mode='lapack', scaling=None, collapse='median', weights=None, collapse_ifs='mean', **rot_options)
        residual_cube.append(adi_pca)
    residual_cube = np.array(residual_cube)
    
    for i in range(sample_nbr):
    
        theta = np.random.rand()*2*np.pi
        x = an_radius * np.cos(theta) + adi[0].shape[0]/2
        y = an_radius * np.sin(theta) + adi[0].shape[1]/2
        posxy= (x,y)
    
        sample_set.append(cube_crop_frames(residual_cube, patch_width, xy=posxy, force=True, verbose=False))
        
    return sample_set

def flux_interval(snr_interv, snr_values):
    low_flux=[]
    for i in range(len(snr_values)):
        if snr_values[i][0] <snr_interv[0]*1.2 and snr_values[i][0] >= snr_interv[0] :
            low_flux.append(snr_values[i][1])
    high_flux=[]
    for i in range(len(snr_values)):
        if snr_values[i][0] > snr_interv[1]*0.8 and snr_values[i][0] <= snr_interv[1]:
            high_flux.append(snr_values[i][1])

    low_flux=np.mean(low_flux)
    high_flux=np.mean(high_flux)
    
    return (low_flux, high_flux)

def patch_rotation(mlar):
    angle= np.random.rand()*2*np.pi
    
    for i in mlar:
        frame_rotate(i, angle, imlib='skimage', interpolation='bicubic')
    
    return mlar
    
    
def patch_shift(mlar):
    shift_x=0
    shift_y=0
    while shift_x==0 and shift_y==0:
        shift_x=np.random.randint(-2, high=3)
        shift_y=np.random.randint(-2, high=3)
    for i in mlar:
        frame_shift(i, shift_y, shift_x, imlib='opencv', interpolation='bicubic', border_mode='reflect')
    return mlar
    
    
def patch_average(mlar_set):

    patch1_index, patch2_index = np.random.choice(len(mlar_set), 2, replace=False)
    patch1=mlar_set[patch1_index]
    patch2=mlar_set[patch2_index]
    new_patch = []
    for i in range(len(patch1)):
        new_patch.append(np.mean( np.array([ patch1[i], patch2[i] ]), axis=0 ))

    return new_patch



