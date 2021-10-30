#compare sdss colors to regular colors
def sd_absolute_mag(spt):
    return {'SDSS_I': spe.typeToMag(spt, 'SDSS_I', reference='zhang2013'), \
            'SDSS_Z': spe.typeToMag(spt, 'SDSS_Z', reference='zhang2013')}

def absolute_mag_color_schmidt(x, color_key):
    #x is a float
    coeffs= SCHMIDT_COLOR_RELATIONS[color_key]
    #linear regression 
    a0= np.random.normal(coeffs[0][0], coeffs[0][-1], 1000)
    a1=np.random.normal(coeffs[1][0], coeffs[1][-1], 1000)
    y= a0+a1*x
    #trim                  
    y=y[np.logical_and(x >coeffs[-1][0], x <coeffs[-1][-1])]
    return np.nanmean(y), np.nanstd(y)

def spt_color_schmidt(spt):
    res={}
    if (spt> 26 or spt <17):
        return {}
    else:
        unc=0.1 #assume for no
        for k in schmidt_rel.keys():
            spts=schmidt_rel[k]['spt']
            vals=schmidt_rel[k]['values']
            val=np.interp(spt, spts, vals)
            res.update({k: val })
        return res

    
    
#@np.vectorize
#def absolute_sdss(spt):
#    if (spt> 26 or spt <17):
#        return {'SDSS_I':(np.nan, np.nan), 'SDSS_Z': (np.nan, np.nan)}
#    else:
 #       colors= spt_color_schmidt(spt)

 #       mi= absolute_mag_color_schmidt(colors['SDSS_I-SDSS_Z'], 'i-z')
 #       mz= (mi[0]-colors['SDSS_I-SDSS_Z'], mi[-1])
 #       return {'SDSS_I':mi, 'SDSS_Z': mz}
