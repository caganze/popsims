################################
#filter profiles
#copied from splat, plus additional filters

##############################
FILTERS = { \
    '2MASS_J': {'file': 'j_2mass.txt', 'description': '2MASS J-band', 'zeropoint': 1594.0, 'method': 'vega', 'rsr': True, 'altname': []}, \
    '2MASS_H': {'file': 'h_2mass.txt', 'description': '2MASS H-band', 'zeropoint': 1024.0, 'method': 'vega', 'rsr': True, 'altname': []}, \
    '2MASS_KS': {'file': 'ks_2mass.txt', 'description': '2MASS Ks-band', 'zeropoint': 666.7, 'method': 'vega', 'rsr': True, 'altname': ['2MASS_K']}, \
#    '2MASS_K': {'file': 'ks_2mass.txt', 'description': '2MASS Ks-band', 'zeropoint': 666.7, 'method': 'vega'}, \
#    '2MASS_Ks': {'file': 'ks_2mass.txt', 'description': '2MASS Ks-band', 'zeropoint': 666.7, 'method': 'vega'}, \
    'BESSEL_U': {'file': 'BESSEL_U.txt', 'description': 'Bessel U-band', 'zeropoint': 0., 'method': 'vega', 'rsr': False, 'altname': ['U']}, \
    'BESSEL_B': {'file': 'Bessel_B.txt', 'description': 'Bessel B-band', 'zeropoint': 0., 'method': 'vega', 'rsr': False, 'altname': ['B']}, \
    'BESSEL_V': {'file': 'Bessel_V.txt', 'description': 'Bessel V-band', 'zeropoint': 0., 'method': 'vega', 'rsr': False, 'altname': ['V']}, \
    'BESSEL_R': {'file': 'Bessel_R.txt', 'description': 'Bessel R-band', 'zeropoint': 0., 'method': 'vega', 'rsr': False, 'altname': ['R']}, \
    'BESSEL_I': {'file': 'Bessel_I.txt', 'description': 'Bessel I-band', 'zeropoint': 0., 'method': 'vega', 'rsr': False, 'altname': ['I']}, \
    'COUSINS_I': {'file': 'i_cousins.txt', 'description': 'Cousins I-band', 'zeropoint': 2405.3, 'method': 'vega', 'rsr': False, 'altname': ['IC']}, \
    'DECAM_U': {'file': 'DECam_u.txt', 'description': 'DECam u-band', 'zeropoint': 1568.5, 'method': 'vega', 'rsr': False, 'altname': ['DECCAM_U','DEC_U']}, \
    'DECAM_G': {'file': 'DECam_g.txt', 'description': 'DECam g-band', 'zeropoint': 3909.11, 'method': 'vega', 'rsr': False, 'altname': ['DECCAM_G','DEC_G']}, \
    'DECAM_R': {'file': 'DECam_r.txt', 'description': 'DECam r-band', 'zeropoint': 3151.44, 'method': 'vega', 'rsr': False, 'altname': ['DECCAM_R','DEC_R']}, \
    'DECAM_I': {'file': 'DECam_i.txt', 'description': 'DECam i-band', 'zeropoint': 2584.6, 'method': 'vega', 'rsr': False, 'altname': ['DECCAM_I','DEC_I']}, \
    'DECAM_Z': {'file': 'DECam_z.txt', 'description': 'DECam z-band', 'zeropoint': 2273.09, 'method': 'vega', 'rsr': False, 'altname': ['DECCAM_Z','DEC_Z']}, \
    'DECAM_Y': {'file': 'DECam_y.txt', 'description': 'DECam y-band', 'zeropoint': 2205.95, 'method': 'vega', 'rsr': False, 'altname': ['DECCAM_Y','DEC_Y']}, \
    'DECAM_VR': {'file': 'DECam_vr.txt', 'description': 'DECam z-band', 'zeropoint': 4000., 'method': 'vega', 'rsr': False, 'altname': ['DECCAM_VR','DEC_VR']}, \
    'DES_U': {'file': 'DES_u.txt', 'description': 'DES u-band (filter + atm)', 'zeropoint': 1568.5, 'method': 'vega', 'rsr': False, 'altname': []}, \
    'DES_G': {'file': 'DES_g.txt', 'description': 'DES g-band (filter + atm)', 'zeropoint': 3909.11, 'method': 'vega', 'rsr': False, 'altname': []}, \
    'DES_R': {'file': 'DES_r.txt', 'description': 'DES r-band (filter + atm)', 'zeropoint': 3151.44, 'method': 'vega', 'rsr': False, 'altname': []}, \
    'DES_I': {'file': 'DES_i.txt', 'description': 'DES i-band (filter + atm)', 'zeropoint': 2584.6, 'method': 'vega', 'rsr': False, 'altname': []}, \
    'DES_Z': {'file': 'DES_z.txt', 'description': 'DES z-band (filter + atm)', 'zeropoint': 2273.09, 'method': 'vega', 'rsr': False, 'altname': []}, \
    'DES_Y': {'file': 'DES_y.txt', 'description': 'DES y-band (filter + atm)', 'zeropoint': 2205.95, 'method': 'vega', 'rsr': False, 'altname': []}, \
    'EUCLID_VIS': {'file': 'Euclid_VIS.txt', 'description': 'Euclid VIS-band', 'zeropoint': 2666.99, 'method': 'ab', 'rsr': False, 'altname': ['EVIS','E_VIS','EUC_VIS']}, \
    'EUCLID_Y': {'file': 'Euclid_Y.txt', 'description': 'Euclid Y-band', 'zeropoint': 1898.78, 'method': 'ab', 'rsr': False, 'altname': ['EY','E_Y','EUC_Y']}, \
    'EUCLID_J': {'file': 'Euclid_J.txt', 'description': 'Euclid H-band', 'zeropoint': 1353.80, 'method': 'ab', 'rsr': False, 'altname': ['EJ','E_J','EUC_J']}, \
    'EUCLID_H': {'file': 'Euclid_H.txt', 'description': 'Euclid K-band', 'zeropoint': 921.81, 'method': 'ab', 'rsr': False, 'altname': ['EH','E_H','EUC_H']}, \
    'FOURSTAR_J': {'file': 'fourstar-j.txt', 'description': 'FOURSTAR J-band', 'zeropoint': 1581.2, 'method': 'vega', 'rsr': False, 'altname': ['4star j']}, \
    'FOURSTAR_J1': {'file': 'fourstar-j1.txt', 'description': 'FOURSTAR J1-band', 'zeropoint': 1978.7, 'method': 'vega', 'rsr': False, 'altname': ['4star j1']}, \
    'FOURSTAR_J2': {'file': 'fourstar-j2.txt', 'description': 'FOURSTAR J2-band', 'zeropoint': 1774.5, 'method': 'vega', 'rsr': False, 'altname': ['4star j2']}, \
    'FOURSTAR_J3': {'file': 'fourstar-j3.txt', 'description': 'FOURSTAR J3-band', 'zeropoint': 1488.8, 'method': 'vega', 'rsr': False, 'altname': ['4star j3']}, \
    'FOURSTAR_H': {'file': 'fourstar-h.txt', 'description': 'FOURSTAR H-band', 'zeropoint': 1054.9, 'method': 'vega', 'rsr': False, 'altname': ['4star h']}, \
    'FOURSTAR_H_SHORT': {'file': 'fourstar-hshort.txt', 'description': 'FOURSTAR H short', 'zeropoint': 1119.1, 'method': 'vega', 'rsr': False, 'altname': ['4star h short','4star h-short','4star hs','fourstar hs','fourstar h1']}, \
    'FOURSTAR_H_LONG': {'file': 'fourstar-hlong.txt', 'description': 'FOURSTAR H long', 'zeropoint': 980.7, 'method': 'vega', 'rsr': False, 'altname': ['4star h long','4star h-long','4star hl','fourstar hl','fourstar h2']}, \
    'FOURSTAR_KS': {'file': 'fourstar-ks.txt', 'description': 'FOURSTAR Ks-band', 'zeropoint': 675.7, 'method': 'vega', 'rsr': False, 'altname': ['4star k','4star ks','fourstar k']}, \
    'FOURSTAR_1.18': {'file': 'fourstar-118.txt', 'description': 'FOURSTAR 1.18 micron narrow band', 'zeropoint': 675.7, 'method': 'vega', 'rsr': False, 'altname': ['4star 1.18','4star 118','fourstar 118']}, \
    'FOURSTAR_2.09': {'file': 'fourstar-209.txt', 'description': 'FOURSTAR 2.09 micron narrow band', 'zeropoint': 675.7, 'method': 'vega', 'rsr': False, 'altname': ['4star 2.09','4star 209','fourstar 209']}, \
    'GAIA_G': {'file': 'GAIA_G.txt', 'description': 'GAIA G-band', 'zeropoint': 3534.7, 'method': 'vega', 'rsr': False, 'altname': ['gaia']}, \
    'GAIA_B': {'file': 'GAIA_Bp.txt', 'description': 'GAIA Bp-band', 'zeropoint': 3296.2, 'method': 'vega', 'rsr': False, 'altname': ['gaia-bp']}, \
    'GAIA_R': {'file': 'GAIA_Rp.txt', 'description': 'GAIA Rp-band', 'zeropoint': 2620.3, 'method': 'vega', 'rsr': False, 'altname': ['gaia-rp']}, \
    'HAWK_Y': {'file': 'hawk-y.txt', 'description': 'HAWK Y-band', 'zeropoint': 2092.9, 'method': 'vega', 'rsr': False, 'altname': []}, \
    'HAWK_J': {'file': 'hawk-j.txt', 'description': 'HAWK J-band', 'zeropoint': 1543.5, 'method': 'vega', 'rsr': False, 'altname': []}, \
    'HAWK_H': {'file': 'hawk-h.txt', 'description': 'HAWK H-band', 'zeropoint': 1053.6, 'method': 'vega', 'rsr': False, 'altname': []}, \
    'HAWK_H2': {'file': 'hawk-h2.txt', 'description': 'HAWK H2-band', 'zeropoint': 688.8, 'method': 'vega', 'rsr': False, 'altname': []}, \
    'HAWK_CH4': {'file': 'hawk-ch4.txt', 'description': 'HAWK CH4-band', 'zeropoint': 1093.4, 'method': 'vega', 'rsr': False, 'altname': []}, \
    'HAWK_KS': {'file': 'hawk-ks.txt', 'description': 'HAWK Ks-band', 'zeropoint': 675.3, 'method': 'vega', 'rsr': False, 'altname': ['hawk k']}, \
    'HAWK_BRG': {'file': 'hawk-brg.txt', 'description': 'HAWK Brackett Gamma', 'zeropoint': 638.9, 'method': 'vega', 'rsr': False, 'altname': []}, \
    'HAWK_NB1060': {'file': 'hawk-nb1060.txt', 'description': 'HAWK Narrow Band 1060', 'zeropoint': 2003.27, 'method': 'vega', 'rsr': False, 'altname': ['hawk 1060']}, \
    'HAWK_NB1190': {'file': 'hawk-nb1190.txt', 'description': 'HAWK Narrow Band 1190', 'zeropoint': 1697.50, 'method': 'vega', 'rsr': False, 'altname': ['hawk 1190']}, \
    'HAWK_NB2090': {'file': 'hawk-nb2090.txt', 'description': 'HAWK Narrow Band 2090', 'zeropoint': 706.68, 'method': 'vega', 'rsr': False, 'altname': ['hawk 2090']}, \
    'IRAC_CH1': {'file': 'irac1.txt', 'description': 'IRAC Channel 1 (3.6 micron)', 'zeropoint': 280.9, 'method': 'vega', 'rsr': True, 'altname': ['irac 1','irac 3.6','[3.6]']}, \
    'IRAC_CH2': {'file': 'irac2.txt', 'description': 'IRAC Channel 2 (4.5 micron)', 'zeropoint': 179.7, 'method': 'vega', 'rsr': True, 'altname': ['irac 2','irac 4.5','[4.5]']}, \
    'IRAC_CH3': {'file': 'irac3.txt', 'description': 'IRAC Channel 3 (5.8 micron)', 'zeropoint': 115.0, 'method': 'vega', 'rsr': True, 'altname': ['irac 3','irac 5.8','[5.8]']}, \
    'IRAC_CH4': {'file': 'irac4.txt', 'description': 'IRAC Channel 4 (8.0 micron)', 'zeropoint': 64.13, 'method': 'vega', 'rsr': True, 'altname': ['irac 4','irac 8.0','[8.0]']}, \
    'KEPLER': {'file': 'Kepler.txt', 'description': 'Kepler bandpass', 'zeropoint': 3033.1, 'method': 'vega', 'rsr': False, 'altname': ['kep','kepler k','kp']}, \
    'MKO_J_ATM': {'file': 'j_atm_mko.txt', 'description': 'MKO J-band + atmosphere', 'zeropoint': 1562.3, 'method': 'vega', 'rsr': False, 'altname': []}, \
    'MKO_H_ATM': {'file': 'h_atm_mko.txt', 'description': 'MKO H-band + atmosphere', 'zeropoint': 1045.9, 'method': 'vega', 'rsr': False, 'altname': []}, \
    'MKO_K_ATM': {'file': 'k_atm_mko.txt', 'description': 'MKO K-band + atmosphere', 'zeropoint': 647.7, 'method': 'vega', 'rsr': False, 'altname': []}, \
# this needs to be added!
#    'MKO_Y': {'file': 'mko_j.txt', 'description': 'MKO Y-band + atmosphere', 'zeropoint': 1562.3, 'method': 'vega', 'rsr': False, 'altname': ['Y']}, \
    'MKO_J': {'file': 'mko_j.txt', 'description': 'MKO J-band + atmosphere', 'zeropoint': 1562.3, 'method': 'vega', 'rsr': False, 'altname': []}, \
    'MKO_H': {'file': 'mko_h.txt', 'description': 'MKO H-band + atmosphere', 'zeropoint': 1045.9, 'method': 'vega', 'rsr': False, 'altname': []}, \
    'MKO_K': {'file': 'mko_ks.txt', 'description': 'MKO K-band', 'zeropoint': 647.7, 'method': 'vega', 'rsr': False, 'altname': []}, \
    'MKO_KP': {'file': 'mko_kp.txt', 'description': 'MKO Kp-band', 'zeropoint': 693.7, 'method': 'vega', 'rsr': False, 'altname': ['mko k prime']}, \
    'MKO_LP': {'file': 'mko_lp.txt', 'description': 'MKO Lp-band', 'zeropoint': 248.3, 'method': 'vega', 'rsr': False, 'altname': ['mko l','mko l prime']}, \
    'MKO_MP': {'file': 'mko_mp.txt', 'description': 'MKO Mp-band', 'zeropoint': 164.7, 'method': 'vega', 'rsr': False, 'altname': ['mko m','mko m prime']}, \
    'NICMOS_F090M': {'file': 'nic1_f090m.txt', 'description': 'NICMOS F090M', 'zeropoint': 2255.0, 'method': 'vega', 'rsr': False, 'altname': ['F090M']}, \
    'NICMOS_F095N': {'file': 'nic1_f095n.txt', 'description': 'NICMOS F095N', 'zeropoint': 2044.6, 'method': 'vega', 'rsr': False, 'altname': ['F095N']}, \
    'NICMOS_F097N': {'file': 'nic1_f097n.txt', 'description': 'NICMOS F097N', 'zeropoint': 2275.4, 'method': 'vega', 'rsr': False, 'altname': ['F097N']}, \
    'NICMOS_F108N': {'file': 'nic1_f108n.txt', 'description': 'NICMOS F108N', 'zeropoint': 1937.3, 'method': 'vega', 'rsr': False, 'altname': ['F108N']}, \
    'NICMOS_F110M': {'file': 'nic1_f110m.txt', 'description': 'NICMOS F110M', 'zeropoint': 1871.8, 'method': 'vega', 'rsr': False, 'altname': ['F110M']}, \
    'NICMOS_F110W': {'file': 'nic1_f110w.txt', 'description': 'NICMOS F110W', 'zeropoint': 1768.5, 'method': 'vega', 'rsr': False, 'altname': ['']}, \
    'NICMOS_F113N': {'file': 'nic1_f113n.txt', 'description': 'NICMOS F113N', 'zeropoint': 1821.0, 'method': 'vega', 'rsr': False, 'altname': ['F113N']}, \
    'NICMOS_F140W': {'file': 'nic1_f140w.txt', 'description': 'NICMOS F140W', 'zeropoint': 1277.1, 'method': 'vega', 'rsr': False, 'altname': ['']}, \
    'NICMOS_F145M': {'file': 'nic1_f145m.txt', 'description': 'NICMOS F145M', 'zeropoint': 1242.0, 'method': 'vega', 'rsr': False, 'altname': ['F145M']}, \
    'NICMOS_F160W': {'file': 'nic1_f160w.txt', 'description': 'NICMOS F160W', 'zeropoint': 1071.7, 'method': 'vega', 'rsr': False, 'altname': ['']}, \
    'NICMOS_F164N': {'file': 'nic1_f164n.txt', 'description': 'NICMOS F164N', 'zeropoint': 1003.0, 'method': 'vega', 'rsr': False, 'altname': ['']}, \
    'NICMOS_F165M': {'file': 'nic1_f165m.txt', 'description': 'NICMOS F165M', 'zeropoint': 1023.6, 'method': 'vega', 'rsr': False, 'altname': ['F165M']}, \
    'NICMOS_F166N': {'file': 'nic1_f166n.txt', 'description': 'NICMOS F166N', 'zeropoint': 1047.7, 'method': 'vega', 'rsr': False, 'altname': ['F166N']}, \
    'NICMOS_F170M': {'file': 'nic1_f170m.txt', 'description': 'NICMOS F170M', 'zeropoint': 979.1, 'method': 'vega', 'rsr': False, 'altname': ['F170M']}, \
    'NICMOS_F187N': {'file': 'nic1_f187n.txt', 'description': 'NICMOS F187N', 'zeropoint': 803.7, 'method': 'vega', 'rsr': False, 'altname': ['F187N']}, \
    'NICMOS_F190N': {'file': 'nic1_f190n.txt', 'description': 'NICMOS F190N', 'zeropoint': 836.5, 'method': 'vega', 'rsr': False, 'altname': ['F190N']}, \
    'NIRC2_J': {'file': 'nirc2-j.txt', 'description': 'NIRC2 J-band', 'zeropoint': 1562.7, 'method': 'vega', 'rsr': False, 'altname': []}, \
    'NIRC2_H': {'file': 'nirc2-h.txt', 'description': 'NIRC2 H-band', 'zeropoint': 1075.5, 'method': 'vega', 'rsr': False, 'altname': []}, \
    'NIRC2_HCONT': {'file': 'nirc2-hcont.txt', 'description': 'NIRC2 H-continuum band', 'zeropoint': 1044.5, 'method': 'vega', 'rsr': False, 'altname': []}, \
    'NIRC2_K': {'file': 'nirc2-k.txt', 'description': 'NIRC2 K-band', 'zeropoint': 648.9, 'method': 'vega', 'rsr': False, 'altname': []}, \
    'NIRC2_KP': {'file': 'nirc2-kp.txt', 'description': 'NIRC2 Kp-band', 'zeropoint': 689.3, 'method': 'vega', 'rsr': False, 'altname': ['nirc2 k prime']}, \
    'NIRC2_KS': {'file': 'nirc2-ks.txt', 'description': 'NIRC2 Ks-band', 'zeropoint': 676.2, 'method': 'vega', 'rsr': False, 'altname': ['nirc2 k short']}, \
    'NIRC2_KCONT': {'file': 'nirc2-kcont.txt', 'description': 'NIRC2 K continuum-band', 'zeropoint': 605.9, 'method': 'vega', 'rsr': False, 'altname': ['nirc2 k continuum']}, \
    'NIRC2_FE2': {'file': 'nirc2-fe2.txt', 'description': 'NIRC2 Fe II', 'zeropoint': 1019.7, 'method': 'vega', 'rsr': False, 'altname': []}, \
    'NIRC2_LP': {'file': 'nirc2-lp.txt', 'description': 'NIRC2 LP', 'zeropoint': 248.0, 'method': 'vega', 'rsr': False, 'altname': ['nirc2 l prime','nirc2 l']}, \
    'NIRC2_M': {'file': 'nirc2-ms.txt', 'description': 'NIRC2 M', 'zeropoint': 165.8, 'method': 'vega', 'rsr': False, 'altname': []}, \
    'NIRCAM_F070W': {'file': 'jwst-nircam-F070W.txt', 'description': 'JWST NIRCAM F070W (wide 0.70 micron)', 'zeropoint': 0., 'method': 'vega', 'rsr': True, 'altname': []}, \
    'NIRCAM_F090W': {'file': 'jwst-nircam-F090W.txt', 'description': 'JWST NIRCAM F090W (wide 0.90 micron)', 'zeropoint': 0., 'method': 'vega', 'rsr': True, 'altname': []}, \
    'NIRCAM_F115W': {'file': 'jwst-nircam-F115W.txt', 'description': 'JWST NIRCAM F115W (wide 1.15 micron)', 'zeropoint': 0., 'method': 'vega', 'rsr': True, 'altname': []}, \
    'NIRCAM_F140M': {'file': 'jwst-nircam-F140M.txt', 'description': 'JWST NIRCAM F140M (medium 1.40 micron)', 'zeropoint': 0., 'method': 'vega', 'rsr': True, 'altname': []}, \
    'NIRCAM_F150W': {'file': 'jwst-nircam-F150W.txt', 'description': 'JWST NIRCAM F150W (wide 1.50 micron)', 'zeropoint': 0., 'method': 'vega', 'rsr': True, 'altname': []}, \
    'NIRCAM_F150W2': {'file': 'jwst-nircam-F150W2.txt', 'description': 'JWST NIRCAM F150W2 (wide 1.50 micron)', 'zeropoint': 0., 'method': 'vega', 'rsr': True, 'altname': []}, \
    'NIRCAM_F162M': {'file': 'jwst-nircam-F162M.txt', 'description': 'JWST NIRCAM F162M (medium 1.62 micron)', 'zeropoint': 0., 'method': 'vega', 'rsr': True, 'altname': []}, \
    'NIRCAM_F164N': {'file': 'jwst-nircam-F164N.txt', 'description': 'JWST NIRCAM F164N (narrow 1.64 micron)', 'zeropoint': 0., 'method': 'vega', 'rsr': True, 'altname': []}, \
    'NIRCAM_F182M': {'file': 'jwst-nircam-F182M.txt', 'description': 'JWST NIRCAM F182M (medium 1.82 micron)', 'zeropoint': 0., 'method': 'vega', 'rsr': True, 'altname': []}, \
    'NIRCAM_F187N': {'file': 'jwst-nircam-F187N.txt', 'description': 'JWST NIRCAM F187N (narrow 1.87 micron)', 'zeropoint': 0., 'method': 'vega', 'rsr': True, 'altname': []}, \
    'NIRCAM_F200W': {'file': 'jwst-nircam-F200W.txt', 'description': 'JWST NIRCAM F200W (wide 2.00 micron)', 'zeropoint': 0., 'method': 'vega', 'rsr': True, 'altname': []}, \
    'NIRCAM_F210W': {'file': 'jwst-nircam-F210M.txt', 'description': 'JWST NIRCAM F210M (medium 2.10 micron)', 'zeropoint': 0., 'method': 'vega', 'rsr': True, 'altname': []}, \
    'NIRCAM_F212N': {'file': 'jwst-nircam-F212N.txt', 'description': 'JWST NIRCAM F212N (narrow 2.12 micron)', 'zeropoint': 0., 'method': 'vega', 'rsr': True, 'altname': []}, \
    'NIRCAM_F250M': {'file': 'jwst-nircam-F250M.txt', 'description': 'JWST NIRCAM F250M (medium 2.50 micron)', 'zeropoint': 0., 'method': 'vega', 'rsr': True, 'altname': []}, \
    'NIRCAM_F277W': {'file': 'jwst-nircam-F277W.txt', 'description': 'JWST NIRCAM F277W (wide 2.77 micron)', 'zeropoint': 0., 'method': 'vega', 'rsr': True, 'altname': []}, \
    'NIRCAM_F300M': {'file': 'jwst-nircam-F300M.txt', 'description': 'JWST NIRCAM F300M (medium 3.00 micron)', 'zeropoint': 0., 'method': 'vega', 'rsr': True, 'altname': []}, \
    'NIRCAM_F322W2': {'file': 'jwst-nircam-F322W2.txt', 'description': 'JWST NIRCAM F322W2 (wide 3.22 micron)', 'zeropoint': 0., 'method': 'vega', 'rsr': True, 'altname': []}, \
    'NIRCAM_F323N': {'file': 'jwst-nircam-F323N.txt', 'description': 'JWST NIRCAM F323N (narrow 3.23 micron)', 'zeropoint': 0., 'method': 'vega', 'rsr': True, 'altname': []}, \
    'NIRCAM_F335M': {'file': 'jwst-nircam-F335M.txt', 'description': 'JWST NIRCAM F335M (medium 3.35 micron)', 'zeropoint': 0., 'method': 'vega', 'rsr': True, 'altname': []}, \
    'NIRCAM_F356W': {'file': 'jwst-nircam-F356W.txt', 'description': 'JWST NIRCAM F356W (wide 3.56 micron)', 'zeropoint': 0., 'method': 'vega', 'rsr': True, 'altname': []}, \
    'NIRCAM_F360M': {'file': 'jwst-nircam-F360M.txt', 'description': 'JWST NIRCAM F360M (medium 3.60 micron)', 'zeropoint': 0., 'method': 'vega', 'rsr': True, 'altname': []}, \
    'NIRCAM_F405N': {'file': 'jwst-nircam-F405N.txt', 'description': 'JWST NIRCAM F405N (narrow 4.05 micron)', 'zeropoint': 0., 'method': 'vega', 'rsr': True, 'altname': []}, \
    'NIRCAM_F410M': {'file': 'jwst-nircam-F410M.txt', 'description': 'JWST NIRCAM F410M (medium 4.10 micron)', 'zeropoint': 0., 'method': 'vega', 'rsr': True, 'altname': []}, \
    'NIRCAM_F430M': {'file': 'jwst-nircam-F430M.txt', 'description': 'JWST NIRCAM F430M (medium 4.30 micron)', 'zeropoint': 0., 'method': 'vega', 'rsr': True, 'altname': []}, \
    'NIRCAM_F444W': {'file': 'jwst-nircam-F444W.txt', 'description': 'JWST NIRCAM F444W (wide 4.44 micron)', 'zeropoint': 0., 'method': 'vega', 'rsr': True, 'altname': []}, \
    'NIRCAM_F460M': {'file': 'jwst-nircam-F460M.txt', 'description': 'JWST NIRCAM F460M (medium 4.60 micron)', 'zeropoint': 0., 'method': 'vega', 'rsr': True, 'altname': []}, \
    'NIRCAM_F466N': {'file': 'jwst-nircam-F466N.txt', 'description': 'JWST NIRCAM F466N (narrow 4.66 micron)', 'zeropoint': 0., 'method': 'vega', 'rsr': True, 'altname': []}, \
    'NIRCAM_F470N': {'file': 'jwst-nircam-F470N.txt', 'description': 'JWST NIRCAM F470N (narrow 4.70 micron)', 'zeropoint': 0., 'method': 'vega', 'rsr': True, 'altname': []}, \
    'NIRCAM_F480M': {'file': 'jwst-nircam-F480M.txt', 'description': 'JWST NIRCAM F480M (medium 4.80 micron)', 'zeropoint': 0., 'method': 'vega', 'rsr': True, 'altname': []}, \
    'PANSTARRS_G': {'file': 'panstarrs-g.txt', 'description': 'PANSTARRS g-band', 'zeropoint': 3909.11, 'method': 'ab', 'rsr': False, 'altname': []}, \
    'PANSTARRS_R': {'file': 'panstarrs-r.txt', 'description': 'PANSTARRS r-band', 'zeropoint': 3151.44, 'method': 'ab', 'rsr': False, 'altname': []}, \
    'PANSTARRS_W': {'file': 'panstarrs-w.txt', 'description': 'PANSTARRS w-band', 'zeropoint': 3024.76, 'method': 'ab', 'rsr': False, 'altname': []}, \
    'PANSTARRS_I': {'file': 'panstarrs-i.txt', 'description': 'PANSTARRS i-band', 'zeropoint': 2584.6, 'method': 'ab', 'rsr': False, 'altname': []}, \
    'PANSTARRS_Z': {'file': 'panstarrs-z.txt', 'description': 'PANSTARRS z-band', 'zeropoint': 2273.09, 'method': 'ab', 'rsr': False, 'altname': []}, \
    'PANSTARRS_Y': {'file': 'panstarrs-y.txt', 'description': 'PANSTARRS y-band', 'zeropoint': 2205.95, 'method': 'ab', 'rsr': False, 'altname': []}, \
    'SDSS_U': {'file': 'sdss-u.txt', 'description': 'SDSS u-band', 'zeropoint': 1568.5, 'method': 'ab', 'rsr': False, 'altname': ['u']}, \
    'SDSS_G': {'file': 'sdss-g.txt', 'description': 'SDSS g-band', 'zeropoint': 3965.9, 'method': 'ab', 'rsr': False, 'altname': ['g']}, \
    'SDSS_R': {'file': 'sdss-r.txt', 'description': 'SDSS r-band', 'zeropoint': 3162.0, 'method': 'ab', 'rsr': False, 'altname': ['r']}, \
    'SDSS_I': {'file': 'sdss-i.txt', 'description': 'SDSS i-band', 'zeropoint': 2602.0, 'method': 'ab', 'rsr': False, 'altname': ['i']}, \
    'SDSS_Z': {'file': 'sdss-z.txt', 'description': 'SDSS z-band', 'zeropoint': 2244.7, 'method': 'ab', 'rsr': False, 'altname': ['z']}, \
    'SKYMAPPER_U': {'file': 'skymapper-u.txt', 'description': 'SkyMapper u-band', 'zeropoint': 1320.1, 'method': 'ab', 'rsr': False, 'altname': ['skymapper u'], 'citation': '2011PASP..123..789B'}, \
    'SKYMAPPER_V': {'file': 'skymapper-v.txt', 'description': 'SkyMapper v-band', 'zeropoint': 2771.8, 'method': 'ab', 'rsr': False, 'altname': ['skymapper v'], 'citation': '2011PASP..123..789B'}, \
    'SKYMAPPER_G': {'file': 'skymapper-g.txt', 'description': 'SkyMapper g-band', 'zeropoint': 3728.2, 'method': 'ab', 'rsr': False, 'altname': ['skymapper g'], 'citation': '2011PASP..123..789B'}, \
    'SKYMAPPER_R': {'file': 'skymapper-r.txt', 'description': 'SkyMapper r-band', 'zeropoint': 3186.0, 'method': 'ab', 'rsr': False, 'altname': ['skymapper r'], 'citation': '2011PASP..123..789B'}, \
    'SKYMAPPER_I': {'file': 'skymapper-i.txt', 'description': 'SkyMapper i-band', 'zeropoint': 2495.7, 'method': 'ab', 'rsr': False, 'altname': ['skymapper i'], 'citation': '2011PASP..123..789B'}, \
    'SKYMAPPER_Z': {'file': 'skymapper-z.txt', 'description': 'SkyMapper z-band', 'zeropoint': 2227.6, 'method': 'ab', 'rsr': False, 'altname': ['skymapper z'], 'citation': '2011PASP..123..789B'}, \
    'SPECULOOS': {'file': 'SPECULOOS_iz.txt', 'description': 'SPECULOOS iz bandpass', 'zeropoint': 2317.40, 'method': 'vega', 'rsr': False, 'altname': ['speculoos_iz','iz']}, \
    'TESS': {'file': 'TESS.txt', 'description': 'TESS bandpass', 'zeropoint': 0., 'method': 'vega', 'rsr': False, 'altname': []}, \
    'UKIDSS_Z': {'file': 'ukidss-z.txt', 'description': 'UKIDSS Z-band', 'zeropoint': 2261.4, 'method': 'vega', 'rsr': False, 'altname': []}, \
    'UKIDSS_Y': {'file': 'ukidss-y.txt', 'description': 'UKIDSS Y-band', 'zeropoint': 2057.2, 'method': 'vega', 'rsr': False, 'altname': []}, \
    'UKIDSS_J': {'file': 'ukidss-j.txt', 'description': 'UKIDSS J-band', 'zeropoint': 1556.8, 'method': 'vega', 'rsr': False, 'altname': []}, \
    'UKIDSS_H': {'file': 'ukidss-h.txt', 'description': 'UKIDSS H-band', 'zeropoint': 1038.3, 'method': 'vega', 'rsr': False, 'altname': []}, \
    'UKIDSS_K': {'file': 'ukidss-k.txt', 'description': 'UKIDSS K-band', 'zeropoint': 644.1, 'method': 'vega', 'rsr': False, 'altname': []}, \
    'VISTA_Z': {'file': 'vista_z.txt', 'description': 'VISTA Z-band', 'zeropoint': 2263.81, 'method': 'vega', 'rsr': False, 'altname': []}, \
    'VISTA_Y': {'file': 'vista_y.txt', 'description': 'VISTA Y-band', 'zeropoint': 2087.32, 'method': 'vega', 'rsr': False, 'altname': []}, \
    'VISTA_J': {'file': 'vista_j.txt', 'description': 'VISTA J-band', 'zeropoint': 1554.03, 'method': 'vega', 'rsr': False, 'altname': []}, \
    'VISTA_H': {'file': 'vista_h.txt', 'description': 'VISTA H-band', 'zeropoint': 1030.40, 'method': 'vega', 'rsr': False, 'altname': []}, \
    'VISTA_KS': {'file': 'vista_ks.txt', 'description': 'VISTA Ks-band', 'zeropoint': 674.83, 'method': 'vega', 'rsr': False, 'altname': []}, \
    'WFC3_F098M': {'file': 'HST-WFC3_IR_F098M.txt', 'description': 'WFC3 F098M', 'zeropoint': 2154.5, 'method': 'vega', 'rsr': False, 'altname': ['F098M']}, \
    'WFC3_F105W': {'file': 'HST-WFC3_IR_F105W.txt', 'description': 'WFC3 F105W', 'zeropoint': 1975.2, 'method': 'vega', 'rsr': False, 'altname': ['wfc3 y','F105W']}, \
    'WFC3_F110W': {'file': 'HST-WFC3_IR_F110W.txt', 'description': 'WFC3 F110W', 'zeropoint': 1738.4, 'method': 'vega', 'rsr': False, 'altname': ['wfc3 yj','F110W']}, \
    'WFC3_F125W': {'file': 'HST-WFC3_IR_F125W.txt', 'description': 'WFC3 F125W', 'zeropoint': 1564.3, 'method': 'vega', 'rsr': False, 'altname': ['wfc3 j','F125W']}, \
    'WFC3_F126N': {'file': 'HST-WFC3_IR_F126N.txt', 'description': 'WFC3 F126N', 'zeropoint': 1552.5, 'method': 'vega', 'rsr': False, 'altname': ['F126N']}, \
    'WFC3_F127M': {'file': 'HST-WFC3_IR_F127M.txt', 'description': 'WFC3 F127M', 'zeropoint': 1496.5, 'method': 'vega', 'rsr': False, 'altname': ['F127M']}, \
    'WFC3_F128N': {'file': 'HST-WFC3_IR_F128N.txt', 'description': 'WFC3 F128N', 'zeropoint': 1392.6, 'method': 'vega', 'rsr': False, 'altname': ['F128N']}, \
    'WFC3_F130N': {'file': 'HST-WFC3_IR_F130N.txt', 'description': 'WFC3 F130N', 'zeropoint': 1475.9, 'method': 'vega', 'rsr': False, 'altname': ['F130N']}, \
    'WFC3_F132N': {'file': 'HST-WFC3_IR_F132N.txt', 'description': 'WFC3 F132N', 'zeropoint': 1466.6, 'method': 'vega', 'rsr': False, 'altname': ['F132N']}, \
    'WFC3_F139M': {'file': 'HST-WFC3_IR_F139M.txt', 'description': 'WFC3 F139M', 'zeropoint': 1342.8, 'method': 'vega', 'rsr': False, 'altname': ['F139M']}, \
    'WFC3_F140W': {'file': 'HST-WFC3_IR_F140W.txt', 'description': 'WFC3 F140W', 'zeropoint': 1324.8, 'method': 'vega', 'rsr': False, 'altname': ['F140W']}, \
    'WFC3_F153M': {'file': 'HST-WFC3_IR_F153M.txt', 'description': 'WFC3 F153M', 'zeropoint': 1142.0, 'method': 'vega', 'rsr': False, 'altname': ['F153M']}, \
    'WFC3_F160W': {'file': 'HST-WFC3_IR_F160W.txt', 'description': 'WFC3 F160W', 'zeropoint': 1138.1, 'method': 'vega', 'rsr': False, 'altname': ['wfc3 h','F160W']}, \
    'WFC3_F164N': {'file': 'HST-WFC3_IR_F164N.txt', 'description': 'WFC3 F164N', 'zeropoint': 1005.5, 'method': 'vega', 'rsr': False, 'altname': ['F164N']}, \
    'WFC3_F167N': {'file': 'HST-WFC3_IR_F167N.txt', 'description': 'WFC3 F167N', 'zeropoint': 1030.0, 'method': 'vega', 'rsr': False, 'altname': ['F167N']}, \
#    'WFC3_F127M': {'file': 'wfc3_F127M.txt', 'description': 'WFC3 F127M', 'zeropoint': 2261.3, 'method': 'vega', 'rsr': False, 'altname': []}, \
#    'WFC3_F139M': {'file': 'wfc3_F139M.txt', 'description': 'WFC3 F139M', 'zeropoint': 2261.3, 'method': 'vega', 'rsr': False, 'altname': []}, \
#    'WFC3_F164N': {'file': 'wfc3_F164N.txt', 'description': 'WFC3 F164N', 'zeropoint': 2261.3, 'method': 'vega', 'rsr': False, 'altname': []}, \
#    'WFC3_F167N': {'file': 'wfc3_F167N.txt', 'description': 'WFC3 F167N', 'zeropoint': 2261.3, 'method': 'vega', 'rsr': False, 'altname': []}, \
    'WFCAM_Z': {'file': 'wfcam-z.txt', 'description': 'UKIRT WFCAM Z', 'zeropoint': 2261.3, 'method': 'vega', 'rsr': False, 'altname': []}, \
    'WFCAM_Y': {'file': 'wfcam-y.txt', 'description': 'UKIRT WFCAM Y', 'zeropoint': 2040.9, 'method': 'vega', 'rsr': False, 'altname': []}, \
    'WFCAM_J': {'file': 'wfcam-j.txt', 'description': 'UKIRT WFCAM J', 'zeropoint': 1548.7, 'method': 'vega', 'rsr': False, 'altname': []}, \
    'WFCAM_H': {'file': 'wfcam-h.txt', 'description': 'UKIRT WFCAM H', 'zeropoint': 1027.1, 'method': 'vega', 'rsr': False, 'altname': []}, \
    'WFCAM_H2': {'file': 'wfcam-h2.txt', 'description': 'UKIRT WFCAM H2', 'zeropoint': 677.1, 'method': 'vega', 'rsr': False, 'altname': []}, \
    'WFCAM_BRG': {'file': 'wfcam-brg.txt', 'description': 'UKIRT WFCAM Brackett Gamma', 'zeropoint': 645.5, 'method': 'vega', 'rsr': False, 'altname': ['wfcam brackett gamma']}, \
    'WFCAM_K': {'file': 'wfcam-k.txt', 'description': 'UKIRT WFCAM K', 'zeropoint': 630.0, 'method': 'vega', 'rsr': False, 'altname': []}, \
    'WIRC_J': {'file': 'wirc_jcont.txt', 'description': 'WIRC J-cont', 'zeropoint': 0., 'method': 'vega', 'rsr': False, 'altname': []}, \
    'WIRC_H': {'file': 'wirc_hcont.txt', 'description': 'WIRC H-cont', 'zeropoint': 0., 'method': 'vega', 'rsr': False, 'altname': []}, \
    'WIRC_K': {'file': 'wirc_kcont.txt', 'description': 'WIRC K-cont', 'zeropoint': 0., 'method': 'vega', 'rsr': False, 'altname': []}, \
    'WIRC_CO': {'file': 'wirc_co.txt', 'description': 'WIRC CO', 'zeropoint': 0., 'method': 'vega', 'rsr': False, 'altname': []}, \
    'WIRC_CH4S': {'file': 'wirc_ch4s.txt', 'description': 'WIRC CH4S', 'zeropoint': 0., 'method': 'vega', 'rsr': False, 'altname': []}, \
    'WIRC_CH4L': {'file': 'wirc_ch4l.txt', 'description': 'WIRC CH4L', 'zeropoint': 0., 'method': 'vega', 'rsr': False, 'altname': []}, \
    'WIRC_FE2': {'file': 'wirc_feii.txt', 'description': 'WIRC Fe II', 'zeropoint': 0., 'method': 'vega', 'rsr': False, 'altname': []}, \
    'WIRC_BRGAMMA': {'file': 'wirc_brgamma.txt', 'description': 'WIRC H I Brackett Gamma', 'zeropoint': 0., 'method': 'vega', 'rsr': False, 'altname': ['wirc brackett gamma']}, \
    'WIRC_PABETA': {'file': 'wirc_pabeta.txt', 'description': 'WIRC H I Paschen Beta', 'zeropoint': 0., 'method': 'vega', 'rsr': False, 'altname': ['wirc paschen beta']}, \
    'WIRCAM_Y': {'file': 'wircam-cfht-y.txt', 'description': 'CFHT WIRCAM Y', 'zeropoint': 2073.32, 'method': 'vega', 'rsr': False, 'altname': []}, \
    'WIRCAM_J': {'file': 'wircam-cfht-j.txt', 'description': 'CFHT WIRCAM J', 'zeropoint': 1551.01, 'method': 'vega', 'rsr': False, 'altname': []}, \
    'WIRCAM_H': {'file': 'wircam-cfht-h.txt', 'description': 'CFHT WIRCAM H', 'zeropoint': 1044.35, 'method': 'vega', 'rsr': False, 'altname': []}, \
    'WIRCAM_KS': {'file': 'wircam-cfht-ks.txt', 'description': 'CFHT WIRCAM Ks', 'zeropoint': 674.62, 'method': 'vega', 'rsr': False, 'altname': []}, \
    'WIRCAM_KCONT': {'file': 'wircam-cfht-kcont.txt', 'description': 'CFHT WIRCAM K-cont', 'zeropoint': 636.17, 'method': 'vega', 'rsr': False, 'altname': []}, \
    'WIRCAM_CH4_OFF': {'file': 'wircam-cfht-ch4s.txt', 'description': 'CFHT WIRCAM CH4-off', 'zeropoint': 987.39, 'method': 'vega', 'rsr': False, 'altname': []}, \
    'WIRCAM_CH4_ON': {'file': 'wircam-cfht-ch4l.txt', 'description': 'CFHT WIRCAM CH4-on', 'zeropoint': 1076.31, 'method': 'vega', 'rsr': False, 'altname': []}, \
    'WISE_W1': {'file': 'wise_w1.txt', 'description': 'WISE W1 (3.5 micron)', 'zeropoint': 309.54, 'method': 'vega', 'rsr': True, 'altname': ['W1']}, \
    'WISE_W2': {'file': 'wise_w2.txt', 'description': 'WISE W2 (4.6 micron)', 'zeropoint': 171.79, 'method': 'vega', 'rsr': True, 'altname': ['W2']}, \
    'WISE_W3': {'file': 'wise_w3.txt', 'description': 'WISE W3 (13 micron)', 'zeropoint': 31.67, 'method': 'vega', 'rsr': True, 'altname': ['W3']}, \
    'WISE_W4': {'file': 'wise_w4.txt', 'description': 'WISE W4 (22 micron)', 'zeropoint': 8.363, 'method': 'vega', 'rsr': True, 'altname': ['W4']} \
    }
VEGAFILE = 'vega_kurucz.txt'
