#absolute magnitude relation
import splat.empirical as spe
import numpy as np
from astropy.io import ascii
import splat

kirkpa2019pol={'2MASS H':{'pol':np.poly1d(np.flip([36.9714, -8.66856, 1.05122 ,-0.0344809])), 
                    'scatter':.67, 'range':[36, 44]}}
kirkpa2020pol={'2MASS H':{'pol':np.poly1d(np.flip([-6.918e01, 1.1863e+01,-5.4084e-1, 8.466e-03 ])), 
                    'scatter':.51, 'range':[36, 44]}}

kirkap202_teff_to_mag_pol={'2MASS H':{'coeffs':[1.2516e+04, -1.5666e+03, 6.7502e+01, -9.2430e-01, -1.9530e-03],\
                                    'coeff_unc':[1.0770e+03, 2.7058e+02, 2.4638e+01, 9.6594e-01, 1.3793e-02], \
                                    'range':[9, 25]}}

def k_clip_fit(x, y, sigma_y, sigma = 5, n=6):
    
    '''Fit a polynomial to y vs. x, and k-sigma clip until convergence
    hard-coded, returns mask array
    '''
    
    not_clipped = np.ones_like(y).astype(bool)
    n_remove = 1
    
    #use median sigma
    #median_sigma= np.nanmedian(sigma_y)
    
    while n_remove > 0:

        best_fit = np.poly1d(np.polyfit(x[not_clipped], y[not_clipped], n))
        
        norm_res = (np.abs(y - best_fit(x)))/(sigma_y)
        remove = np.logical_and(norm_res >= sigma, not_clipped == 1)
        n_remove = sum(remove)
        not_clipped[remove] = 0   
        
    return  not_clipped

def fit_with_nsigma_clipping(x, y, y_unc, n, sigma=3.):
    not_clipped = k_clip_fit(x, y, y_unc, sigma = sigma)
    return not_clipped, np.poly1d(np.polyfit(x[not_clipped], y[not_clipped], n))

best_dict={'2MASS J': {\
            'spt': [16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39], \
            'values': [10.36,10.77,11.15,11.46,11.76,12.03,12.32,12.77,13.51,13.69,14.18,14.94,14.90,14.46,14.56,15.25,14.54,14.26,13.89,14.94,15.53,16.78,17.18,17.75],\
            'rms': [0.30,0.30,0.42,0.34,0.18,0.15,0.21,0.24,0.28,0.25,0.60,0.20,0.13,0.71,0.5,0.12,0.06,0.16,0.36,0.12,0.27,0.76,0.51,0.5]},
        '2MASS H': {\
            'spt': [16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39], \
            'values': [9.76,10.14,10.47,10.74,11.00,11.23,11.41,11.82,12.45,12.63,13.19,13.82,13.77,13.39,13.62,14.39,13.73,13.67,13.57,14.76,15.48,16.70,17.09,17.51],\
            'rms': [0.30,0.31,0.43,0.35,0.23,0.21,0.25,0.29,0.3,0.30,0.62,0.31,0.20,0.73,0.5,0.18,0.15,0.24,0.40,0.24,0.37,0.78,0.5,0.5]},
       
        'PANSTARRS_R': {\
            'spt': [16,17,18,19,20,21,22,23,24,25,26,28], \
            'values': [15.37,16.76,17.74,18.14,18.37,18.74,19.02,19.61,20.60,20.74,21.21,22.88],\
            'rms': [0.43,0.50,0.51,0.37,0.31,0.28,0.29,0.39,0.56,0.37,0.78,0.5]},
        'PANSTARRS_I': {\
            'spt': [16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,35], \
            'values': [13.25,14.18,15.00,15.62,16.00,16.41,16.73,17.40,18.35,18.71,19.27,20.09,20.38,20.09,20.22,21.10,21.97,22.69],\
            'rms': [0.34,0.39,0.48,0.39,0.26,0.25,0.26,0.34,0.38,0.33,0.65,0.36,0.79,0.5,1.14,0.5,0.5,0.5]},
        'PANSTARRS_Z': {\
            'spt': [16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39], \
            'values': [12.27,12.98,13.61,14.17,14.52,14.93,15.30,15.88,16.68,16.94,17.35,18.18,18.10,17.69,17.98,18.84,18.26,18.08,18.02,19.20,19.82,21.17,21.52,21.82],\
            'rms': [0.32,0.34,0.45,0.37,0.25,0.23,0.24,0.21,0.33,0.29,0.62,0.26,0.22,0.73,0.5,0.21,0.23,0.25,0.39,0.22,0.32,0.78,0.52,0.5]},
        'PANSTARRS_Y': {\
            'spt': [16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39], \
            'values': [11.76,12.31,12.81,13.23,13.58,13.97,14.33,14.89,15.66,15.87,16.27,17.13,17.04,16.57,16.77,17.45,16.75,16.50,16.32,17.43,18.06,19.34,19.75,20.37],\
            'rms': [0.30,0.31,0.43,0.36,0.23,0.21,0.24,0.29,0.32,0.28,0.61,0.25,0.21,0.72,0.5,0.16,0.13,0.22,0.38,0.18,0.32,0.81,0.59,0.5]}}

BEST={}
for k in best_dict.keys():
    BEST[k]=fit_with_nsigma_clipping(np.array(best_dict[k]['spt']), 
                                      np.array(best_dict[k]['values']), 
                                      np.array(best_dict[k]['rms'])
                                     , 6, sigma=5.)

def make_mamajek_fit(spt):
    mamjk=ascii.read('/users/caganze/research/wisps/data/mamajek_relations.txt').to_pandas().replace('None', np.nan)
    js=mamjk.M_J.apply(float).values
    jminush=mamjk['J-H'].apply(float).values
    hs=js-jminush
    
    spts=mamjk.SpT.apply(splat.typeToNum).apply(float).values
    
    hsortedindex=np.argsort(hs)
    jsortedindex=np.argsort(js)
    
    hval=np.interp(spt,  spts[hsortedindex], hs[hsortedindex])
    jval=np.interp(spt,  spts[jsortedindex], js[jsortedindex])
    
    return ((jval, 0.4), (hval, 0.4))

def absolute_mag_kirkpatrick(spt, filt):
    if filt != '2MASS H':
        return np.nan
    else:
        pol=kirkpa2019pol[filt]['pol']
        unc=kirkpa2019pol[filt]['scatter']
        return pol(spt-30), unc

def get_teff_from_mag_ignore_unc(mag):
    corr=splat.photometry.vegaToAB('2MASS H')
    pol=np.poly1d(np.flip(kirkap202_teff_to_mag_pol['2MASS H']['coeffs']))
    #ignore objects outside the [9, 25] range
    rng=kirkap202_teff_to_mag_pol['2MASS H']['coeffs']
    res=None
    if np.isscalar(mag):
        #check outisde the range
        if np.logical_or(mag>rng[1], mag<rng[0]): mag=np.nan
        res=pol(mag)
    else:
        res=np.ones_like(mag)*np.nan
        bools=np.logical_or(mag>rng[1], mag<rng[0])
        res[bools]=pol(mag)[bools]
    return res+corr



def get_teff_from_mag(mag, mag_unc=0.0, flt='2MASS H'):
    nsample=int(1e4)
    #ignore coeffs_unc_for_no
    coff_unc=np.array(kirkap202_teff_to_mag_pol[flt]['coeff_unc'])#*(1e-3)
    coeffs=np.random.normal(np.array(kirkap202_teff_to_mag_pol[flt]['coeffs']),\
                            coff_unc,(nsample, 5))
    
    x=mag
    xerr=mag_unc
    res=(np.nan, np.nan)

    if np.isscalar(mag):
        x=np.random.normal(mag, xerr, nsample)
        vals=coeffs[:,0] + coeffs[:,1]*x+ coeffs[:,2]*(x**2)+coeffs[:,3]*(x**3)+coeffs[:,4]*(x**4)
        res=np.nanmedian(vals), np.nanstd(vals)
    else:
        if mag_unc==0.0:
            xerr=np.zeros_like(mag)
            
        x=np.random.normal(mag, xerr, (nsample, len(mag))).T
    
        vals=coeffs[:,0] + coeffs[:,1]*x+ coeffs[:,2]*(x**2)+coeffs[:,3]*(x**3)+coeffs[:,4]*(x**4)
        
        res=np.nanmedian(vals, axis=1), np.nanstd(vals, axis=1)

    return res


def get_abs_mag(spt, flter):
    m, munc=(np.nan, np.nan)
    if flter=='2MASS J':
        corr=splat.photometry.vegaToAB('2MASS J')
        #m, munc= spe.typeToMag(spt, '2MASS J')
        ((m, munc), (h, hunc))=make_mamajek_fit(spt)
        m= m+corr
        
    if flter=='2MASS H':
        corr=splat.photometry.vegaToAB('2MASS H')
        if np.isscalar(spt):
            if spt <36:
                #m=BEST['2MASS H'][1](spt)
                #munc= np.nanmedian(best_dict[k]['rms'])
                #m, munc=spe.typeToMag(spt, '2MASS H')
                ((j, junc), (m, munc))=make_mamajek_fit(spt)
                m= m+corr
            else:
                m, munc=absolute_mag_kirkpatrick(spt, '2MASS H')
                m= m+corr
        else:
            m=np.ones_like(spt)*np.nan

            #np.place(m, spt>36,(make_mamajek_fit(spt[spt<36])[1][0])+corr )
            #np.place(m, spt>=36, absolute_mag_kirkpatrick(spt[spt>=36], '2MASS H')[0]+corr)
          
            m[spt<36]= (make_mamajek_fit(spt[spt<36])[1][0])+corr
            m[spt>=36]= (absolute_mag_kirkpatrick(spt[spt>=36], '2MASS H')[0])+corr
            munc=(0.4**2+0.67**2)**0.5
               
            
    if flter=='PANSTARRS_R':
        corr=splat.photometry.vegaToAB('PANSTARRS_R')
        m=(BEST[k][1])(spt)+corr
        munc=np.nanmedian(best_dict[k]['rms'])
    return m, munc