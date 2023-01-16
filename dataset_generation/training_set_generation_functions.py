from matplotlib.pyplot import *
from matplotlib import pyplot as plt
import numpy as np
import copy
import vip_hci as vip
import h5py
from hciplot import plot_frames, plot_cubes

from vip_hci.config import VLT_NACO
from vip_hci.fm import normalize_psf, cube_inject_companions
from vip_hci.psfsub import median_sub, pca, pca_annular, pca_annulus

from vip_hci.fits import open_fits, write_fits, info_fits
from vip_hci.metrics import significance, snr, snrmap
from vip_hci.var import fit_2dgaussian, frame_center

from vip_hci.preproc.cosmetics import cube_crop_frames, frame_crop
from vip_hci.preproc import frame_rotate, cube_shift, cube_derotate
from vip_hci.preproc.recentering import frame_shift

from vip_hci.config.utils_conf import pool_map, iterable



def inject(flevel, theta, adi, psf, pangles, pxscale, rad_dists):
    array_out, inject_pos_yx = cube_inject_companions(adi, psf, pangles, flevel=flevel, plsc=pxscale, rad_dists=rad_dists, theta=theta, imlib='opencv', interpolation='bicubic', full_output=True)  # array_out, inject_pos_yx
    res=[array_out, (inject_pos_yx[0][1], inject_pos_yx[0][0])]  #returning pos_xy
    return res


def inject_random_fake_comp(adi, psf, pxscale, pangles, fwhm, rad_dists, inj_nbr=1, flx='random', nproc=1):
    """
    From one adi sequence returns the same adi sequence with injected fake companion
    
    adi: adi sequence to inject fake companion into
    psf: psf liked to adi seuqence
    pxscale: pixel scale related to adi sequence
    pangles: paralactic angles related to adi sequence
    fwhm: full width half maximum related to adi sequence
    rad_dist: radial distance from star to inject fake companion at
    inj_nbr:numbre of injections to be made
    flx: luminosity flux of injected companion (random/...)
    nproc: processor number to be used if using multiprocessing
    
    output: injected adi sequence, (y,x) coordinates of fake companion injection
    """

    norm_psf = normalize_psf(psf, fwhm=fwhm, model='gauss', imlib='opencv', interpolation='bicubic')       #Normalizing PSF
    fluxes=[]
    if not flx == 'random':
        from_stat=[0,(395.3,544.7),(165.6,344.4),(142.1,267.8),(33.1,75.4),(31.4,65.7),(7.8,19.4),(8.28,14.7),(2.8,9.6),(1.5,5.96),(2.7,6.2),(2.6,4.7),(2.7,4.9),(2.1,5),(3.3,5.12),(3.4,6.7),(2,6),(3.5,6.9),(4.5,9.5),(2.9,8.6)]
        try:
            flevel= [np.random.randint(flx[0], high=flx[1]) for i in range(inj_nbr)]
        except:
            a,b = from_stat[int(rad_dists/fwhm)][0], from_stat[int(rad_dists/fwhm)][1]
            flevel= [np.random.randint(a, high=b) for i in range(inj_nbr)]
    else:
        flux_range=[0,(100,850),(70,600),(50,400),(10,250),(5,100), (0,90), (0,80), (0,50), (0,30)]+30*[(0,20)]
        flevel= [np.random.randint(flux_range[int(rad_dists/fwhm)][0], high=flux_range[int(rad_dists/fwhm)][1]) for i in range(inj_nbr)]     
    theta = [np.random.rand()*360 for i in range(inj_nbr)] #in degrees
    

    res=pool_map(nproc, inject, iterable(flevel), iterable(theta), adi, psf, pangles, pxscale, rad_dists)
    res=np.array(res, dtype=object)
    return res[:,0], np.array(res[:,1]), flevel # res(:,0): adi cubes with fake companion res(:,1): injection position of fake companion flevel: flux level of injected FC'''

def adi_mlar_pca(adi_with_fake_comp, pangles, ncomp, fwhm, an_radius):
    pca=[]
    rot_options={'imlib':'skimage', 'interpolation':'bicubic'}
    for i in range(1, ncomp+1):
        pca.append(pca_annulus(adi_with_fake_comp, pangles, i, 3*fwhm, an_radius, cube_ref=None, svd_mode='lapack', scaling=None, collapse='median', weights=None, collapse_ifs='mean', **rot_options))
    return pca

def crop(pos, residual_cube, patch_width):
    return cube_crop_frames(residual_cube, patch_width, xy=pos, force=True, verbose=False)

def make_mlar_plus(adi_with_fake_comp, ncomp, pangles, fwhm, inj_posxy, an_radius, nproc=1):
    
    """
    From injected ADI sequence, crop MLAR patch around fake companion injection position.
    
    adi_with_fake_comp: injected adi sequence
    nproc: processor number to be used if using multiprocessing
    fwhm: full width half maximum related to adi sequence
    pangles: paralactic angles related to adi sequence
    inj_posxy: (x,y) injection position of fake companion
    an_radius: radius of the annulus fake companion has been injected in
    
    output:
    generated MLAR patch (as a list)
    """
    patch_width = np.ceil(2*fwhm) // 2 * 2 + 1    # rounding to the nearest odd integer
    patch_width = int(patch_width)
    

    mlar_adi_pca= pool_map(nproc, adi_mlar_pca, iterable(adi_with_fake_comp),pangles, ncomp, fwhm, an_radius, nproc)
    MLAR= pool_map(nproc, crop, iterable(inj_posxy), iterable(mlar_adi_pca), patch_width)
        
    return MLAR
    
def get_fwhm(psf):
    
    """
    Computes full width at half maximum for a given ADI sequence with its linked PSF
    
    output: fwhm
    """
    DF_fit = fit_2dgaussian(psf, crop=True, cropsize=9, debug=False, full_output=True)
    fwhm = np.mean([DF_fit['fwhm_x'],DF_fit['fwhm_y']])   #using gaussian fit to get FWHM
    
    return fwhm

def pool_snr(residual, inj_posxy, fwhm):
    return snr(residual, source_xy = inj_posxy, fwhm=fwhm, plot=False, full_output=True, verbose=False )

def pool_median_sub(adi_with_fake_comp, pangles, fwhm, an_radius):
    return median_sub(adi_with_fake_comp, pangles, fwhm, asize=2*fwhm, mode='annular', delta_rot=0.5, radius_int= an_radius, nframes=4, imlib='skimage', interpolation='bicubic', verbose=False)

def evaluate_snr(adi_with_fake_comp, pangles, fwhm, inj_posxy, an_radius, nproc=1):
    
    """
    Evaluates signal to noise ratio from one ADI sequence with fake companion injection, knowing its (x,y) injection position, annulus radius of said position
    adi_with_fake_comp: adi sequence with injected fake companion
    pangles: paralactic angles linked to given ADI sequence
    fwhm: full width half maximum related to adi sequence
    inj_posxy: (x,y) injection position of fake companion
    an_radius: radius of the annulus fake companion has been injected in
    nproc: processor number to be used if using multiprocessing
    
    output: S/N at given pos_xy position
    """
#    residual= median_sub(adi_with_fake_comp, pangles, fwhm, asize=2*fwhm, mode='annular', delta_rot=0.5, radius_int= an_radius, nframes=4, imlib='skimage', interpolation='bicubic', verbose=False)
    residual= pool_map(nproc, pool_median_sub, iterable(adi_with_fake_comp), pangles, fwhm, an_radius)
    
    res= pool_map(nproc, pool_snr, iterable(residual), iterable(inj_posxy), fwhm)
    #sourcey, sourcex, f_source, fluxes, snr_vale = snr(residual, source_xy = inj_posxy, fwhm=fwhm, plot=False, full_output=True, verbose=False )
    return res
#    return snr_vale, f_source, sourcex, sourcey
    
def plot_patch(mlar):
    
    """
    From given MLAR patch, plots it
    """
    plot_frames(tuple(mlar[i] for i in range(len(mlar))))
    return

def make_residual_cube(adi, pangles, ncomp, fwhm, an_radius):
    residual_cube=[]
    rot_options={'imlib':'skimage', 'interpolation':'bicubic'}
    for i in range(1, ncomp+1):
        adi_pca = pca_annulus(adi, pangles, ncomp, 3*fwhm, an_radius, cube_ref=None, svd_mode='lapack', scaling=None, collapse='median', weights=None, collapse_ifs='mean', **rot_options)
        residual_cube.append(adi_pca)
    residual_cube = np.array(residual_cube)
    return residual_cube


def make_mlar_minus(residual_cube, fwhm, an_radius, sample_nbr, nproc=1):
    
    """
    From given adi sequence makes a C- MLAR patch set
    
    pangles: paralactic angles linked to given ADI sequence
    fwhm: full width half maximum related to adi sequence
    an_radius: radius of annulus to extract C- patch from
    sample_nbr: numbre of noise patch samples to be taken
    
    output
    MLAR C- patches set
    """
    
    sample_set=[]
    
    patch_width = np.ceil(2*fwhm) // 2 * 2 + 1    # rounding to the nearest odd integer
    patch_width = int(patch_width)
    
    pos=[]
    for i in range(sample_nbr):
        theta = np.random.rand()*2*np.pi
        x = an_radius * np.cos(theta) + residual_cube[0].shape[0]/2
        y = an_radius * np.sin(theta) + residual_cube[0].shape[1]/2
        pos.append((x,y))

    sample_set=pool_map(nproc, crop, iterable(pos), residual_cube, patch_width)
        
    return sample_set



def flux_interval(snr_interv, snr_values):
    
    """
    From a given S/R interval, gives corresponding lower and higher flux value for injecting companions in that S/R interval by making an
    average of flux intensities lying around lower snr value and higher snr value 
    
    snr_interv: (snr_low, snr_high)
    snr_values: (snr, flux) S/R and its corresponding flux intensity as evaluated by evaluate_snr() for a given injection
    
    output:
    (low_flux, high_flux) lower and higher flux value
    """
    
    low_flux=[]
    for i in range(len(snr_values)):
        if snr_values[i][0] <snr_interv[0]*1.2 and snr_values[i][0] >= snr_interv[0] :
            low_flux.append(snr_values[i][2])
    high_flux=[]
    for i in range(len(snr_values)):
        if snr_values[i][0] > snr_interv[1]*0.8 and snr_values[i][0] <= snr_interv[1]:
            high_flux.append(snr_values[i][2])

    low_flux=np.mean(low_flux)
    high_flux=np.mean(high_flux)
    
    return (low_flux, high_flux)


def rotation(patch, angle):
#    rot = [frame_rotate(i, angle, imlib='vip-fft', interpolation='lanczos4', cxy=None, border_mode='constant', edge_blend=None, interp_zeros=False, ker=1) for i in patch]
    each_frame_angle = angle*np.ones(len(patch))
    rot = cube_derotate(patch, each_frame_angle, border_mode='reflect', imlib='opencv')
    return rot


def patch_rotation(mlar, nproc=1):
    
    """
    from MLAR patch set, gives the same MLAR patch set with random rotation for each
    
    """
    angle=[]
    for i in range(len(mlar)):
        angle.append(np.random.rand()*360)
    res=pool_map(nproc, rotation, iterable(mlar), iterable(angle))
    
    return res
    
    
def shift(patch, xy_shift):
#    res = [frame_shift(i, xy_shift[1], xy_shift[0], imlib='opencv', interpolation='bicubic', border_mode='reflect') for i in patch]
    res = cube_shift(cube=patch, shift_x=xy_shift[0], shift_y=xy_shift[1],
                                       imlib='opencv', border_mode='reflect')
    return res
    
def patch_shift(mlar, nproc):
    
    """
    from MLAR patch set, returns the same MLAR patch set with each patch having all pixels shifted with random value of 1 or 2 pixels
    """
    shift_list=[]
    for i in range(len(mlar)):
        shift_x=0
        shift_y=0
        while shift_x==0 and shift_y==0:
            shift_x=np.random.randint(-2, high=3)
            shift_y=np.random.randint(-2, high=3)
            shift_list.append((shift_x,shift_y))

        res= pool_map(nproc, shift, iterable(mlar), iterable(shift_list))
        
    return res

def average(couple, mlar_set):
    patch1=mlar_set[couple[0]]
    patch2=mlar_set[couple[1]]
    new_patch = []
    for i in range(len(patch1)):
        new_patch.append(np.mean( np.array([ patch1[i], patch2[i] ]), axis=0 ))
    return new_patch
    
def patch_average(mlar_set, nproc=1):
    
    """
    from MLAR patch set, gives a new MLAR patch by averaging 2 randomly picked MLAR patches
    """
    couples=[]
    for i in range(len(mlar_set)):
        patch1_index, patch2_index = np.random.choice(len(mlar_set), 2, replace=False)
        couples.append((patch1_index, patch2_index))
    res = pool_map(nproc, average, iterable(couples), mlar_set)

    return res

def save_to_h5(annulus,dataset_dict):
    
    """
    saves to 'training_dataset' h5 file all data contained in dataset_dict under directory annulus+L_D
    eg: for a set of C+ and C- patches generated in 5 \lambda / D annulus, saves them in directory called '5L_D'
    
    dataset_dict: python dictionnary, each item is saved under a separate file  with name being the key
                  eg: {name_of_the_file: data_to_be_saved, another_name: other_data}
    output: /
    """
    
    h5file = h5py.File('training_dataset', 'w')
    h5group= h5file.create_group(str(annulus)+'L_D')
    
    for x, y in dataset_dict.items():
        h5dset = h5group.create_dataset(x, data=y)
    h5file.close()
    return

def dataset_name(name):
    print(name)
    striped_name=''
    count=-1
    for i in range(len(name)):
        while name[count] != '/':
            striped_name+=name[count]
            count-=1
    return striped_name

def read_h5(name, annulus):
    
    """
    Reads all file in 'name/annulusL_D'
    eg: to get C+ and C- MLAR patch from 5 \lambda / D annulus, we call read_h5('training_dataset', 5)
    
    output: list of (file_name, data_contained)
    """
    
    h5file = h5py.File(name, 'r')
    h5dset = h5file[str(annulus)+'L_D']
    return [(x, h5dset[x][()]) for x in h5dset]

def plot_10_random(mlar_seq):
    nbr=10
    ncomp=len(mlar_seq[0])
    if len(mlar_seq)<=10:
        nbr=len(mlar_seq)
        for j in range(nbr):
            plot_frames(tuple(mlar_seq[j][i] for i in range(ncomp)))
    else:
        for j in range(nbr):
            plot_frames(tuple(mlar_seq[np.random.randint(0, high=len(mlar_seq))][i] for i in range(ncomp)))
    return
