from matplotlib.pyplot import *
from matplotlib import pyplot as plt
import numpy as np
import time
from importlib import reload
'''
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

'''
from vip_hci.config.utils_conf import pool_map, iterable
import training_set_generation_functions
reload(training_set_generation_functions)
from training_set_generation_functions import inject_random_fake_comp, adi_mlar_pca, crop, make_mlar_minus, get_fwhm, evaluate_snr, plot_patch, flux_interval, patch_rotation, patch_shift, patch_average, save_to_h5, read_h5, make_residual_cube, plot_10_random

st1, st2, et1, et2= 0,0,0,0
def clock(evaluate_run_time, s_t):
    global st1, st2, et1, et2
    if evaluate_run_time and s_t=='start':
        st1=time.time()
        st2=time.process_time()
    if evaluate_run_time and s_t=='stop':
        et1=time.time()
        et2=time.process_time()
        print('Execution time:', et1-st1, 'seconds')
        print('CPU Execution time:', et2-st2, 'seconds \n')
    return
        

def flux_statistics(adi, pangles, psf, pxscale, annulus, sample_size, n_proc, evaluate_run_time=False, plot=False, saved_stat=True):
    
    if saved_stat:
        #use flux intervals previously computed over 500 samples
        from_stat=[0,(395.3,544.7),(165.6,344.4),(142.1,267.8),(33.1,75.4),(31.4,65.7),(7.8,19.4),(8.28,14.7),(2.8,9.6),(1.5,5.96),(2.7,6.2),(2.6,4.7),(2.7,4.9),(2.1,5),(3.3,5.12),(3.4,6.7),(2,6),(3.5,6.9),(4.5,9.5),(2.9,8.6)]
        return from_stat[annulus]
    
    fwhm = get_fwhm(psf)

#injecting fake companions
    clock(evaluate_run_time,'start')
        
    adi_with_fake_comp, inj_pos, flevel = inject_random_fake_comp(adi, psf, pxscale, pangles, fwhm, an_radius, inj_nbr=sample_size, flx='random', nproc=n_proc)
    
    print('injecting done')
    clock(evaluate_run_time, 'stop')
    
#evaluating signal to noise ratio for each injection
    clock(evaluate_run_time,'start')
        
    snr_res = evaluate_snr(adi_with_fake_comp, pa, fwhm, inj_pos, an_radius, nproc=6)  #[[posy, posx,flux, array(flux_appertures), s/r],....]
    
    print('snr done')
    clock(evaluate_run_time, 'stop')
    
#keeping values only for 1 < snr < 3
    good_values=[]
    for i in range(len(snr_res)):
        if snr_res[i][4]>snr_interv[0] and snr_res[i][4]<snr_interv[1]:
            good_values.append([snr_res[i][4], snr_res[i][1], snr_res[i][0]])
    snr_values=[[i[4],i[2]] for i in snr_res]

#making average
    snr_values=np.array(snr_values)
    flevel=np.array([flevel])
    if plot:
        scatter(snr_values[:,1],snr_values[:,0])
    snr_values = np.concatenate((snr_values,flevel.T), axis=1)
    flux_interv = flux_interval(snr_interv, snr_values)
    
    return flux_interv

def C_plus_class(adi, pangles, psf, pxscale, annulus, patch_numbre, n_proc, ncomp, data_aug=False, evaluate_run_time=False):
    
    fwhm = get_fwhm(psf)
    an_radius = annulus*fwhm
    patch_width = np.ceil(2*fwhm) // 2 * 2 + 1    # rounding to the nearest odd integer
    patch_width = int(patch_width)
    
    sample_size=500 #for statisticss
    flux_interv= flux_statistics(adi, pangles, psf, pxscale, annulus, sample_size, n_proc, evaluate_run_time=False, plot=False, saved_stat=True)
    
# injecting fake companions
    clock(evaluate_run_time, 'start')

    adi_with_fake_comp, inj_pos, flevel = inject_random_fake_comp(adi, psf, pxscale, pangles, fwhm, an_radius, inj_nbr=patch_numbre, flx=flux_interv, nproc=n_proc) #still need to include flux statisctics part
    print('injecting done')
    clock(evaluate_run_time, 'stop')

# making residual cubes
    clock(evaluate_run_time, 'start')

    adi_pca_cubes= pool_map(n_proc, adi_mlar_pca, iterable(adi_with_fake_comp), pangles, ncomp, fwhm, an_radius)
    adi_pca_cubes=np.array(adi_pca_cubes)
    
    print('PCA done')
    clock(evaluate_run_time, 'stop')

# cropping patches 
    clock(evaluate_run_time, 'start')
        
    MLAR= pool_map(n_proc, crop, iterable(inj_pos), iterable(adi_pca_cubes), patch_width)
    
    print('cropping done')
    clock(evaluate_run_time, 'stop')

# data augementation
    patches_plus_aug=[]
    if data_aug:
        print('data_augmentation \n')
        clock(evaluate_run_time, 'start')
            
        patches_plus_aug+=patch_rotation(MLAR, nproc=n_proc)
        print('rotation done')
        clock(evaluate_run_time, 'stop')
        
        clock(evaluate_run_time, 'start')
            
        patches_plus_aug+=patch_shift(MLAR, nproc=n_proc)
        
        print('shift done')
        clock(evaluate_run_time, 'stop')
        
        clock(evaluate_run_time, 'start')
            
        patches_plus_aug+=patch_average(MLAR, nproc=n_proc)
        
        print('average done')
        clock(evaluate_run_time, 'stop')
        
    return MLAR, patches_plus_aug

def C_minus_class(adi, pangles, psf, pxscale, annulus, patch_numbre, n_proc, ncomp, data_aug=False, evaluate_run_time=False):

    fwhm = get_fwhm(psf)
    an_radius = annulus*fwhm
    patch_width = np.ceil(2*fwhm) // 2 * 2 + 1    # rounding to the nearest odd integer
    patch_width = int(patch_width)   
    
# making residual cube (only 1 needed)    
    clock(evaluate_run_time, 'start')

    adi_pca_cubes= pool_map(n_proc, adi_mlar_pca, iterable([adi]), pangles, ncomp, fwhm, an_radius)
    adi_pca_cubes=np.array(adi_pca_cubes)
    print('PCA done')
    clock(evaluate_run_time, 'stop')   
    
# cropping random patches
    clock(evaluate_run_time, 'start')
    pos=[]
    for i in range(patch_numbre):
        theta = np.random.rand()*2*np.pi
        x = an_radius * np.cos(theta) + adi_pca_cubes[0][0].shape[0]/2
        y = an_radius * np.sin(theta) + adi_pca_cubes[0][0].shape[1]/2
        pos.append((x,y))

    sample_set=pool_map(n_proc, crop, iterable(pos), adi_pca_cubes[0], patch_width)
    
    print('cropping done')
    clock(evaluate_run_time, 'stop')
    
# data augementation
    patches_min_aug=[]
    if data_aug:
        print('data_augmentation \n')
        clock(evaluate_run_time, 'start')
            
        patches_min_aug+=patch_rotation(sample_set, nproc=n_proc)
        
        print('rotation done')
        clock(evaluate_run_time, 'stop')
        
        clock(evaluate_run_time, 'start')
            
        patches_min_aug+=patch_shift(sample_set, nproc=n_proc)
        
        print('shift done')
        clock(evaluate_run_time, 'stop')
        
        clock(evaluate_run_time, 'start')
            
        patches_min_aug+=patch_average(sample_set, nproc=n_proc)
        
        print('average done')
        clock(evaluate_run_time, 'stop')
    
    return sample_set, patches_min_aug

def ask_input():
    int_nbr=list(range(1,15))
    int_nbr=[str(i) for i in int_nbr]
    yes_or_no=['y','n']
    
    annulus=input('annulus ?')
    while annulus not in int_nbr:
        annulus=input('annulus ? (integre > 0)')
    annulus=int(annulus)
    
    ncomp=input('Numbre of principal components ?')
    while ncomp not in int_nbr:
        ncomp=input('Numbre of principal components ? (integre > 0)')
    ncomp=int(ncomp)
    
    patch_numbre=input('numbre of patches to make? (integre>0)')
    try:
        patch_numbre=int(patch_numbre)
    except:
        patch_numbre=input('numbre of patches to make? (integre>0)')
    patch_numbre=int(patch_numbre)
    
    data_aug=input('data augmentation ? (y/n)')
    while not data_aug in yes_or_no:
        data_aug=input('data augmentation ? (y/n)')
    if data_aug=='y':
        data_aug=True
    else:
        data_aug=False
        
    n_proc=input('multiprocessing? (integre>0)')
    while n_proc not in int_nbr[:9]:
        n_proc=input('multiprocessing? (integre>0)')
    n_proc=int(n_proc)

    evaluate_run_time=input('display runtime ? (y/n)')
    while not evaluate_run_time in yes_or_no:
        evaluate_run_time=input('display runtime ? (y/n)')
    if evaluate_run_time=='y':
        evaluate_run_time=True
    else:
        evaluate_run_time=False
    
    return annulus, patch_numbre, ncomp, data_aug, n_proc, evaluate_run_time

def main():

#loading ADI sequence
    adi=np.load('adi_sequence/adi_seq.npy')
    pangles=np.load('adi_sequence/pa.npy')
    psf=np.load('adi_sequence/psf.npy')
    pxscale=np.load('adi_sequence/pxscale.npy')    
    
#asking user for input
    annulus, patch_numbre, ncomp, data_aug, n_proc, evaluate_run_time = ask_input()

#generating patches   
    patches_plus, patches_plus_aug = C_plus_class(adi, pangles, psf, pxscale, annulus, patch_numbre, n_proc, ncomp, data_aug, evaluate_run_time)
    patches_minus, patches_minus_aug = C_minus_class(adi, pangles, psf, pxscale, annulus, patch_numbre, n_proc, ncomp, data_aug, evaluate_run_time)
#saving
    save_to_h5(annulus, {'patches_plus':patches_plus, 'patches_plus_aug':patches_plus_aug, 'patches_minus':patches_minus, 'patches_minus_aug':patches_minus_aug})
    plot_10_random(patches_plus)
    plot_10_random(patches_minus)
    return

if __name__ == "__main__":
    main()
    
