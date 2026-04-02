################################################################################
###### This is an example script to generate HDF5-format ICs for GIZMO
######  The specific example below is obviously arbitrary, but could be generalized
######  to whatever IC you need. 
################################################################################
################################################################################

## load libraries we will use 
import numpy as np
import h5py as h5py
from gizmopy import *
import matplotlib
import matplotlib.pylab as pylab
matplotlib.use('Agg') ## this calls matplotlib without a X-windows GUI

def plottime(sdir='./',snum_min=0,snum_max=2999,x0=0.5,renorm=1.):
    n0=snum_max-snum_min+1
    t=np.zeros(n0)
    vy=np.zeros(n0); vz=np.zeros(n0); by=np.zeros(n0); bz=np.zeros(n0)
    vx=np.zeros(n0); bx=np.zeros(n0)
    for i in range(snum_max+1):
        snum = i
        t[i] = load_from_snapshot('Time',-1,sdir,snum)
        ids=load_from_snapshot('ParticleIDs',0,sdir,snum) 
        xyz=load_from_snapshot('Coordinates',0,sdir,snum).transpose(); 
        x=xyz[0]; s=np.argsort(x); x=x[s]; ids=ids[s]
        if(i==0):
            id0 = ids[np.argmin(np.abs(x-x0))]
        vxyz=load_from_snapshot('Velocities',0,sdir,snum).transpose(); 
        bxyz=load_from_snapshot('MagneticField',0,sdir,snum).transpose(); 
        for j in [0,1,2]:
            vxyz[j,:]=vxyz[j,s]
            bxyz[j,:]=bxyz[j,s]
        vx[i] = np.interp(x0,x,vxyz[0])               
        vy[i] = np.interp(x0,x,vxyz[1])               
        vz[i] = np.interp(x0,x,vxyz[2])               
        bx[i] = np.interp(x0,x,bxyz[0])               
        by[i] = np.interp(x0,x,bxyz[1])               
        bz[i] = np.interp(x0,x,bxyz[2])   
        
        if(2==2):
            q = np.where(ids == id0)[0]
            vy[i] = (vxyz[1])[q][0]
            vz[i] = (vxyz[2])[q][0]
            by[i] = (bxyz[1])[q][0]
            bz[i] = (bxyz[2])[q][0]

        print(snum,t[i],vy[i],np.mean(vxyz[0]))

    dB_B0 = renorm * 1.e-3
    B0 = 0.1/np.sqrt(4.*np.pi)
    k = 2.*np.pi * 5.
    eta_Hall = 0.01

    ## this one doesn't seem to help, even though i think from my previous scaling it should?
    if(2==0):
        B0 = 0.2 * 0.1/np.sqrt(4.*np.pi)
        k = 2.*np.pi * 1.
        eta_Hall = 0.01

    ## this one -does- seem to help. so reducing k*vA, or something, helps?
    if(2==0):
        B0 = 0.1/np.sqrt(4.*np.pi)
        k = 2.*np.pi * 1.
        eta_Hall = 0.05

    rho0 = 1.
    vA = B0/np.sqrt(rho0)
    sigma = eta_Hall * k*k/2.
    omega = np.sqrt(vA*vA*k*k + sigma*sigma)
    dB = dB_B0 * B0
    dv = -dB_B0 * omega/k
    bx = bx - B0
    T_slow = 80.
    lH = eta_Hall/vA
    klH = k*lH
    klH2 = klH*klH
    omega_H = vA*vA/eta_Hall
    omega_slow = (-klH2/2. + np.sqrt((klH2/2.)**2 + klH2)) * omega_H
    T_slow = 2.*np.pi/omega_slow
    t0plot = omega_H/(2.*np.pi)

    pylab.close('all'); figure=pylab.figure(1,figsize=(6.,4.5)); f00=14.; matplotlib.rcParams.update({'font.size':f00}); legloc='best'; 
    pylab.plot(t*t0plot,vx/dv,color='blue',linestyle='--',label=r'$v_{x}/\delta v$',linewidth=1.)
    pylab.plot(t*t0plot,vy/dv,color='blue',linestyle='-',label=r'$v_{y}/\delta v$',linewidth=1.)
    pylab.plot(t*t0plot,vz/dv,color='blue',linestyle=':',label=r'$v_{z}/\delta v$',linewidth=1.)
    pylab.plot(t*t0plot,bx/dB,color='red',linestyle='--',label=r'$B_{x}/\delta B$',linewidth=2.)
    pylab.plot(t*t0plot,by/dB,color='red',linestyle='-',label=r'$B_{y}/\delta B$',linewidth=2.)
    pylab.plot(t*t0plot,bz/dB,color='red',linestyle=':',label=r'$B_{z}/\delta B$',linewidth=2.)

    #pylab.plot(t*t0plot,np.sqrt(by*by+bz*bz)/dB,color='orange',linestyle='--',label=r'$|B_{yz}|/\delta B$',linewidth=2.)
    #pylab.plot(t*t0plot,np.abs(np.sqrt(vy*vy+vz*vz)/dv),color='lime',linestyle='--',label=r'$|v_{yz}|/\delta v$',linewidth=2.)

    
    
    tdamp_numerical = 500. # numerical inherent dissipation
    tdamp_physical = 50. # for omega_0 = 0.2*omega_H, roughly
    tdamp = 1./(1./tdamp_physical + 1./tdamp_numerical) # omega_O = 0.2*omega_H, roughly
    tdamp = 1./(0.1/tdamp_physical + 1./tdamp_numerical) # omega_O = 0.02*omega_H, roughly
    renorm = 1.0; poff=0.1;

    tdamp*=2.2; renorm=0.85; poff=0.05; omega_slow*=1.05
   
    pylab.plot(t*t0plot,renorm*0.55*np.sin(omega_slow*t+poff+0.05)*np.exp(-t/tdamp),color='magenta',linestyle=':',label=r'Wave at $\omega_{\rm slow}$')
    pylab.plot(t*t0plot,-renorm*0.55*np.cos(omega_slow*t+poff+0.15)*np.exp(-t/tdamp),color='magenta',linestyle='--')

    pylab.xlabel(r'Time: $\omega_{H} t/2\pi$')
    pylab.ylabel(r'Value')
    pylab.ylim(-1.,1.)
    
    #By_analytic = dB * np.cos(omega*t - k*x0) * np.cos(sigma*t)
    #Bz_analytic = dB * np.cos(omega*t - k*x0) * np.sin(sigma*t)
    #pylab.plot(t,By_analytic/dB,color='magenta',linestyle=':')
    #pylab.plot(t,Bz_analytic/dB,color='magenta',linestyle=':')


    pylab.legend(loc='best',fontsize=f00*0.8,handletextpad=0.5,borderpad=0.5,labelcolor='linecolor',
        handlelength=3.,frameon=False,columnspacing=0.2,labelspacing=0.2,numpoints=1)
    pylab.subplots_adjust(left=0.1,bottom=0.1,right=1-0.01,top=1-0.01,hspace=0.01,wspace=0.01);
    pylab.savefig(sdir+'/Nina_HallTest.pdf',transparent=True,bbox_inches='tight',pad_inches=0)

# the main routine. this specific example builds an N-dimensional box of gas plus 
#   a collisionless particle species, with a specified mass ratio. the initial 
#   gas particles are distributed in a uniform lattice; the initial collisionless 
#   particles laid down randomly according to a uniform probability distribution 
#   with a specified random velocity dispersion
#
def make_IC():
    '''
    This is an example subroutine provided to demonstrate how to make HDF5-format
    ICs for GIZMO. The specific example here is arbitrary, but can be generalized
    to whatever IC you need
    '''

    DIMS=1; # number of dimensions 
    N_1D=160; # 1D particle number (so total particle number is N_1D^DIMS)
    fname='box_1d_r160.hdf5'; # output filename 

    Lbox = 1.0 # box side length
    rho_desired = 1.0 # box average initial gas density
    P_desired = 1.0 # initial gas pressure
    gamma_eos = 5./3.;
    x0=np.arange(-0.5,0.5,1./N_1D); x0+=0.5*(0.5-x0[-1]);
    # now extend that to a full lattice in DIMS dimensions
    if(DIMS==3):
        xv_g, yv_g, zv_g = np.meshgrid(x0,x0,x0, sparse=False, indexing='xy')
    if(DIMS==2):
        xv_g, yv_g = np.meshgrid(x0,x0, sparse=False, indexing='xy'); zv_g = 0.0*xv_g
    if(DIMS==1):
        xv_g=x0; yv_g = 0.0*xv_g; zv_g = 0.0*xv_g; 
    # the gas particle number is the lattice size: this should be the gas particle number
    Ngas = xv_g.size
    # flatten the vectors (since our ICs should be in vector, not matrix format): just want a
    #  simple list of the x,y,z positions here. Here we multiply the desired box size in
    xv_g=xv_g.flatten()*Lbox; yv_g=yv_g.flatten()*Lbox; zv_g=zv_g.flatten()*Lbox; 
    # set the initial velocity in x/y/z directions (here zero)
    vx_g=0.*xv_g; vy_g=0.*xv_g; vz_g=0.*xv_g;
    # set the initial magnetic field in x/y/z directions (here zero). 
    #  these can be overridden (if constant field values are desired) by BiniX/Y/Z in the parameterfile
    bx_g=0.*xv_g; by_g=0.*xv_g; bz_g=0.*xv_g;
    # set the particle masses. Here we set it to be a list the same length, with all the same mass
    #   since their space-density is uniform this gives a uniform density, at the desired value
    mv_g=rho_desired/((1.*Ngas)/(Lbox*Lbox*Lbox)) + 0.*xv_g
    # set the initial internal energy per unit mass. recall gizmo uses this as the initial 'temperature' variable
    #  this can be overridden with the InitGasTemp variable (which takes an actual temperature)
    uv_g=P_desired/((gamma_eos-1.)*rho_desired) + 0.*xv_g
    # set the gas IDs: here a simple integer list
    id_g=np.arange(1,Ngas+1)


    # now we get ready to actually write this out
    #  first - open the hdf5 ics file, with the desired filename
    file = h5py.File(fname,'w') 

    # set particle number of each type into the 'npart' vector
    #  NOTE: this MUST MATCH the actual particle numbers assigned to each type, i.e.
    #   npart = np.array([number_of_PartType0_particles,number_of_PartType1_particles,number_of_PartType2_particles,
    #                     number_of_PartType3_particles,number_of_PartType4_particles,number_of_PartType5_particles])
    #   or else the code simply cannot read the IC file correctly!
    #
    npart = np.array([Ngas,0,0,0,0,0]) # we have gas and particles we will set for type 3 here, zero for all others

    # now we make the Header - the formatting here is peculiar, for historical (GADGET-compatibility) reasons
    h = file.create_group("Header");
    # here we set all the basic numbers that go into the header
    # (most of these will be written over anyways if it's an IC file; the only thing we actually *need* to be 'correct' is "npart")
    h.attrs['NumPart_ThisFile'] = npart; # npart set as above - this in general should be the same as NumPart_Total, it only differs 
                                         #  if we make a multi-part IC file. with this simple script, we aren't equipped to do that.
    h.attrs['NumPart_Total'] = npart; # npart set as above
    h.attrs['NumPart_Total_HighWord'] = 0*npart; # this will be set automatically in-code (for GIZMO, at least)
    h.attrs['MassTable'] = np.zeros(6); # these can be set if all particles will have constant masses for the entire run. however since 
                                        # we set masses explicitly by-particle this should be zero. that is more flexible anyways, as it 
                                        # allows for physics which can change particle masses 
    h.attrs['Time'] = 0.0;  # initial time
    h.attrs['NumFilesPerSnapshot'] = 1; # number of files for multi-part snapshots
    h.attrs['Flag_DoublePrecision'] = 0; # flag indicating whether ICs are in single/double precision
    ## no other parameters will be read from the header if we are parsing an HDF5 file for reading in
    
    # Now, the actual data!
    #   These blocks should all be written in the order of their particle type (0,1,2,3,4,5)
    #   If there are no particles of a given type, nothing is needed (no block at all)
    #   PartType0 is 'special' as gas. All other PartTypes take the same, more limited set of information in their ICs
    
    # start with particle type zero. first (assuming we have any gas particles) create the group 
    p = file.create_group("PartType0")
    # now combine the xyz positions into a matrix with the correct format
    q=np.zeros((Ngas,3)); q[:,0]=xv_g; q[:,1]=yv_g; q[:,2]=zv_g;
    # write it to the 'Coordinates' block
    p.create_dataset("Coordinates",data=q)
    # similarly, combine the xyz velocities into a matrix with the correct format
    q=np.zeros((Ngas,3)); q[:,0]=vx_g; q[:,1]=vy_g; q[:,2]=vz_g;
    # write it to the 'Velocities' block
    p.create_dataset("Velocities",data=q)
    # write particle ids to the ParticleIDs block
    p.create_dataset("ParticleIDs",data=id_g)
    # write particle masses to the Masses block
    p.create_dataset("Masses",data=mv_g)
    # write internal energies to the InternalEnergy block
    p.create_dataset("InternalEnergy",data=uv_g)
    # combine the xyz magnetic fields into a matrix with the correct format
    q=np.zeros((Ngas,3)); q[:,0]=bx_g; q[:,1]=by_g; q[:,2]=bz_g;
    # write magnetic fields to the MagneticField block. note that this is unnecessary if the code is compiled with 
    #   MAGNETIC off. however, it is not a problem to have the field there, even if MAGNETIC is off, so you can 
    #   always include it with some dummy values and then use the IC for either case
    p.create_dataset("MagneticField",data=q)

    # close the HDF5 file, which saves these outputs
    file.close()
    # all done!



def make_IC_3d():
    DIMS=3; # number of dimensions 
    N_long=160; # 1D particle number (so total particle number is N_1D^DIMS)
    N_short=10;
    fname='box_3d_r160_x10x10.hdf5'; # output filename 
    Lbox_long = 1.0 # box side length
    rho_desired = 1.0 # box average initial gas density
    P_desired = 1.0 # initial gas pressure
    gamma_eos = 5./3.;
    Lbox_short = Lbox_long * (1.*N_short)/(1.*N_long)
    x0=np.arange(-0.5,0.5,1./N_long); x0+=0.5*(0.5-x0[-1]);
    xv_g, yv_g, zv_g = np.meshgrid(x0,x0,x0, sparse=False, indexing='xy')
    xv_g=xv_g.flatten()*Lbox_long; yv_g=yv_g.flatten()*Lbox_long; zv_g=zv_g.flatten()*Lbox_long; 
    ok = (np.abs(yv_g) < Lbox_short/2.) & (np.abs(zv_g) < Lbox_short/2.)
    xv_g = xv_g[ok]
    yv_g = yv_g[ok]
    zv_g = zv_g[ok]
    #pylab.close('all'); pylab.xlim(-Lbox_short/2.,Lbox_short/2.); pylab.ylim(-Lbox_short/2.,Lbox_short/2.); pylab.plot(xv_g,yv_g,marker='.',color='blue',linestyle='')
    Ngas = xv_g.size
    vx_g=0.*xv_g; vy_g=0.*xv_g; vz_g=0.*xv_g;
    bx_g=0.*xv_g; by_g=0.*xv_g; bz_g=0.*xv_g;
    mv_g=rho_desired/((1.*Ngas)/(Lbox_long*Lbox_short*Lbox_short)) + 0.*xv_g
    uv_g=P_desired/((gamma_eos-1.)*rho_desired) + 0.*xv_g
    id_g=np.arange(1,Ngas+1)
    file = h5py.File(fname,'w') 
    npart = np.array([Ngas,0,0,0,0,0]) # we have gas and particles we will set for type 3 here, zero for all others
    h = file.create_group("Header");
    h.attrs['NumPart_ThisFile'] = npart; # npart set as above - this in general should be the same as NumPart_Total, it only differs 
    h.attrs['NumPart_Total'] = npart; # npart set as above
    h.attrs['NumPart_Total_HighWord'] = 0*npart; # this will be set automatically in-code (for GIZMO, at least)
    h.attrs['MassTable'] = np.zeros(6); # these can be set if all particles will have constant masses for the entire run. however since 
    h.attrs['Time'] = 0.0;  # initial time
    h.attrs['NumFilesPerSnapshot'] = 1; # number of files for multi-part snapshots
    h.attrs['Flag_DoublePrecision'] = 0; # flag indicating whether ICs are in single/double precision
    p = file.create_group("PartType0")
    q=np.zeros((Ngas,3)); q[:,0]=xv_g; q[:,1]=yv_g; q[:,2]=zv_g;
    p.create_dataset("Coordinates",data=q)
    q=np.zeros((Ngas,3)); q[:,0]=vx_g; q[:,1]=vy_g; q[:,2]=vz_g;
    p.create_dataset("Velocities",data=q)
    p.create_dataset("ParticleIDs",data=id_g)
    p.create_dataset("Masses",data=mv_g)
    p.create_dataset("InternalEnergy",data=uv_g)
    q=np.zeros((Ngas,3)); q[:,0]=bx_g; q[:,1]=by_g; q[:,2]=bz_g;
    p.create_dataset("MagneticField",data=q)
    file.close()


