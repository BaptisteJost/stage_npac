import pylab as plt
import healpy as hp
import numpy as np
import camb
from camb import model, initialpower
import lib_project as lib
import plot_project as plotpro
from astropy import units as u

c1='#1b9e77'
c2='#d95f02'
c3='#7570b3'
c4='#e7298a'
c5='#66a61e'
c6='#e6ab02'
c7='#a6761d'
c8='#666666'

import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = 13
mpl.rcParams['ytick.labelsize'] = 13


# xerr=(bins[1:]-bins[:-1])*.5
def plotit(ax,notext=False):
    l,t,te,e,b,d=np.loadtxt('COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt',unpack=1)
    # ax.plot(l,b,c='black',lw=2)

    lt,tt,et,bt,tet=np.loadtxt('july17_tensCls.dat',unpack=1)
    print("bt r = 0.01",bt[:3001]/0.001*0.01)
    print('lt=',lt)

    print('len(bt[:3001])',len(bt[:3001]))
    print('len(lt)',len(lt))

    # ax.plot(lt,bt/0.001*0.06,c='#a8006d',lw=3)
    # ax.plot(lt,bt/0.001*0.01,c='#a8006d',lw=3)
    # ax.plot(lt,bt/0.001*0.001,c='#a8006d',lw=3)
    # ax.plot(l,bt[:2507]/0.001*0.06+b,c='k',lw=2,ls='dashed',alpha=.5)

    # ax.plot(l,bt[:2507]/0.001*0.01+b,c='k',lw=2,ls='dashed',alpha=0.5)#,label='lensing BB r = 0.01')
    # ax.plot(l,bt[:2507]/0.001*0.001+b,c='k',lw=2,ls='dashed',alpha=0.5)#,label='lensing BB r = 0.001')

    ax.fill_between([1,2777],[0.003171707018341639,0.0001686920621994168],[0.7809093938324934,0.038409158645132106],color='k',alpha=.3)

    l=np.arange(3001)

    custom_lines = []
    custom_text = []

    lbins=np.array([ 500, 900,1300,1700])
    mbins=np.array([ 700.0,1100.0,1500.0,1900.0])
    hbins=np.array([  900, 1300, 1700, 2100])
    cls=np.array([0.0530,0.0740,0.0450,0.3520])
    sclm=np.array([0.0260,0.0540,0.1090,0.2390])
    sclp=np.array([0.0290,0.0570,0.1160,0.2600])
    ax.errorbar(mbins,cls,yerr=[sclm,sclp],xerr=[mbins-lbins,hbins-mbins],c='#ff3801',fmt=' ',label='POLARBEAR 2017',marker='o',alpha=1)


    lbins=np.array([ 60, 93,126,158,190,221,254,286])
    mbins=np.array([ 73.6,107.4,141.0,173.8,205.5,237.3,270.3,303.1])
    hbins=np.array([ 86,121,156,192,227,262,297,332])
    cls=np.array([1.38e-03,6.40e-03,8.44e-03,1.30e-02,2.02e-02,2.09e-02,3.60e-02,3.38e-02])
    sclm=np.array([1.05e-04,4.22e-03,5.71e-03,8.94e-03,1.56e-02,1.48e-02,2.79e-02,2.26e-02])
    sclp=np.array([2.89e-03,8.98e-03,1.14e-02,1.73e-02,2.53e-02,2.73e-02,4.48e-02,4.55e-02])
    ax.errorbar(mbins,cls,yerr=[cls-sclm,sclp-cls],xerr=[mbins-lbins,hbins-mbins],c='#ff7f00',fmt=' ',label='BICEP2/Keck+Planck/WMAP 2018',marker='o',alpha=1)
    ax.errorbar([46.5],[2.63e-03],xerr=[[46.5-37],[54-46.5]],c='#ff7f00',fmt=' ',marker='v',alpha=1,markersize=10)


    lbins=np.array([52,152 ,302 ,702 ,1102,1502,1902])
    mbins=np.array([102,227,502,902,1302,1702,2102])
    hbins=np.array([151 ,301 ,701 ,1101,1501,1901,2301])
    cls=np.array([ 0.004,0.030,0.094,0.118,0.097,0.119,0.064])
    sclm=np.array([0.014,0.010,0.010,0.016,0.026,0.042,0.060])
    sclp=sclm
    ax.errorbar(mbins,cls,yerr=[sclm,sclp],xerr=[mbins-lbins,hbins-mbins],c='#ffab00',fmt=' ',label='SPTpol 2019',marker='o',alpha=1)


    return custom_lines, custom_text
    #ax.scatter(bl,bcl,c=cred,label='PB2a 2021')

from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

fig, axMain = plt.subplots(1,2,gridspec_kw={'width_ratios': [1, 1.27]},figsize=(22,8))



# axMain = pl.subplot(111)

custom_lines, custom_text=plotit(axMain[0])

axMain[0].set_xscale('log')
axMain[0].set_yscale('log')
axMain[0].set_xlim((2, 2500))
axMain[0].set_ylim((6e-6, 1e1))

axMain[1].set_xscale('log')
axMain[1].set_yscale('log')
axMain[1].set_xlim((2, 2500))
axMain[1].set_ylim((6e-6, 1e1))

axMain[0].legend(fontsize=15,ncol=1,edgecolor='white') #,bbox_to_anchor=(1.02, 1.0), loc=2, borderaxespad=0.)

axMain[0].set_xlabel('Multipole moment $\ell$',fontsize=25)
axMain[0].text(3,2.6e-3,r'$f_{sky}=1\%\ @\ 100GHz$',rotation=-9,fontsize=15)
axMain[0].text(3,6.3e-1,r'$f_{sky}=90\%\ @\ 100GHz$',rotation=-9,fontsize=15)

axMain[0].text(2.1,5e-4,r'$r=0.06$',fontsize=13,color='#a8006d')
axMain[0].text(2.1,8e-5,r'$r=0.01$',fontsize=13,color='#a8006d')
axMain[0].text(2.1,2.2e-5,r'$r=0.001$',fontsize=13,color='#a8006d')
plt.grid(ls='dotted')
# pl.ylim(-.25e-7,2e-7)
axMain[0].set_ylabel('$\ell(\ell+1)$ $C_\ell^{BB}$ $/2\pi$ [$\mu K^2$]',fontsize=25)





l_max = 3000
raw_cl = False
ratio_var = 0.01
pars, results, powers = lib.get_basics(l_max, raw_cl, ratio=ratio_var)
pars0, results0, powers0 = lib.get_basics(l_max, raw_cl, ratio=0)
total_r0 = powers0['total']


for name in powers: print(name)



spectrum = 'unlensed_total' #'unlensed_total'
unchanged_cl = powers[spectrum]

lensed_cl = powers['total']
print('lensed_cl shape', lensed_cl.shape[0])
angle_array = np.linspace(0.,1., 100)
angle_array = angle_array * u.deg
print('angle_array',angle_array)

spectrum_plotted = 'BB'
plotting_spectrum_dict= {'TT':0 , 'EE':1, 'BB':2, 'TE':3, 'EB':4,'TB':5}

spectra_dict = lib.get_spectra_dict(unchanged_cl, angle_array, include_unchanged = False)

plotpro.one_spectrum(spectra_dict, spectrum_plotted,axMain[1])


ls = np.arange(lensed_cl.shape[0])


lt1,tt1,et1,bt1,tet1=np.loadtxt('july17_tensCls.dat',unpack=1)
l1,t1,te1,e1,b1,d1=np.loadtxt('COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt',unpack=1)

lensing_b_component = powers['lensed_scalar'][:,plotting_spectrum_dict[spectrum_plotted]]
axMain[0].plot(ls,unchanged_cl[:,plotting_spectrum_dict[spectrum_plotted]]/0.01*0.06 + lensing_b_component, 'k--', color = 'black', linewidth = 2.0, alpha=.5)
axMain[0].plot(ls,unchanged_cl[:,plotting_spectrum_dict[spectrum_plotted]]/0.01*0.01 + lensing_b_component, 'k--', color = 'black', linewidth = 2.0, alpha=.5)
axMain[0].plot(ls,unchanged_cl[:,plotting_spectrum_dict[spectrum_plotted]]/0.01*0.001 + lensing_b_component, 'k--', color = 'black', linewidth = 2.0, alpha=.5)



axMain[0].plot(ls,total_r0[:,plotting_spectrum_dict[spectrum_plotted]],  color= 'black', linewidth= 2)
axMain[0].plot(ls,unchanged_cl[:,plotting_spectrum_dict[spectrum_plotted]]/0.01 * 0.06, color = 'purple', linewidth = 3.0)
axMain[0].plot(ls,unchanged_cl[:,plotting_spectrum_dict[spectrum_plotted]]/0.01 * 0.01, color = 'purple', linewidth = 3.0)
axMain[0].plot(ls,unchanged_cl[:,plotting_spectrum_dict[spectrum_plotted]]/0.01 * 0.001, color = 'purple', linewidth = 3.0)



# axMain[1].plot(l1,b1,c='k',lw=2,label='lensing, r=0')
# axMain[1].plot(l1,bt1[:2507]/0.001*0.06+b1,c='k',lw=2,ls='dashed',alpha=0.5,label='lensing BB r = 0.06')
# axMain[1].plot(l1,bt1[:2507]/0.001*0.01+b1,c='k',lw=2,ls='--',alpha=0.5,label='lensing BB r = 0.01')
# axMain[1].plot(l1,bt1[:2507]/0.001*0.001+b1,c='k',lw=2,ls=':',alpha=0.5,label='lensing BB r = 0.001')
# axMain[1].plot(lt1,bt1/0.001*0.01,label='pirmordial BB, r={}'.format(ratio_var), color = 'purple', linewidth = 3.0)

axMain[1].plot(ls,lensed_cl[:,plotting_spectrum_dict[spectrum_plotted]], 'k--', label='total BB', color = 'black', linewidth = 2.0,alpha=0.5) #'{}'.format(key)
# axMain[1].plot(ls,lensed_cl[:,plotting_spectrum_dict[spectrum_plotted]], 'k--', label='lensing BB', color = 'black', linewidth = 2.0,alpha=0.05) #'{}'.format(key)


axMain[1].plot(ls,total_r0[:,plotting_spectrum_dict[spectrum_plotted]], label = 'lensing, r=0', color= 'black', linewidth= 2)



axMain[1].plot(ls,unchanged_cl[:,plotting_spectrum_dict[spectrum_plotted]],label='primordial BB, r={}'.format(ratio_var), color = 'purple', linewidth = 3.0) #'{}'.format(key)

 #'{}'.format(key)

print("unchanged_cl r = 0.01",unchanged_cl[:,plotting_spectrum_dict[spectrum_plotted]])
print("len unchanged_cl", len(unchanged_cl[:,plotting_spectrum_dict[spectrum_plotted]]))
# axMain[1].ylim(1e-5,1e1)
axMain[0].grid(b = True, linestyle=':')

axMain[1].grid(b = True, linestyle=':')
axMain[1].set_xlabel(r'Multipole moment $\ell$', fontsize= 25)
# plt.ylabel(r'$ \ell (\ell +1) \;   C_{\ell}^{BB} \;\;  / 2 \pi \;\; [\mu K^{2}]$',fontsize= 20)

# leg = axMain[1].legend(fontsize= 14, framealpha=0.8)
# leg.get_frame().set_linewidth(0.0)
axMain[1].legend(fontsize=15,ncol=1,edgecolor='white')
"""
plt.axes().set_aspect(0.477)
# plt.savefig('/mnt/c/Users/Baptiste/Documents/APC_2019/project/stage_npac/BB_birefringence_articel.png')
# plt.show()



plt.show()
"""
plt.subplots_adjust(wspace=0.08)
plt.savefig("bb_combined_&_birefringence.png",dpi=200)#, bbox_inches='tight')
plt.show()
