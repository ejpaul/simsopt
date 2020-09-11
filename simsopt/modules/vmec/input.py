# coding: utf-8

import os
import sys
import logging

import numpy as np
import scipy
import scipy.linalg
from scipy.io import netcdf
import f90nml
from scanf import scanf
import numpy as np
from simsopt.modules.vmec.surface_utils import point_in_polygon, init_modes, \
    min_max_indices_2d, proximity_slice, self_contact, self_intersect, \
    global_curvature_surface, proximity_surface, proximity_derivatives_func, \
    cosine_IFT, sine_IFT, surface_intersect

__author__ = "Elizabeth Paul, Bharat Medasani"
__copyright__ = ""
__version__ = "0.0.1"
__maintainer__ = "Bharat Medasani"
__email__ = "mbkumar@gmail.com"
__status__ = "Preliminary"
__date__ = "Jul 03, 2020"


class VmecInput:
  
    #def __init__(self, ntheta=100, nzeta=100, **kwargs):
    #    self.__dict__.update(kwargs)
    #    self.update_grids(ntheta, nzeta)
    #    self.min_curvature_radius = 0.3

    #@classmethod
    #def from_file(cls, filename):
    def __init__(self, input_filename, ntheta=100, nzeta=100):
        """
        Reads VMEC input parameters from a VMEC input file. 

        Args:
            input_filename (str): The filename to read from.
            ntheta (int): Number of poloidal gridpoints
            nzeta (int): Number of toroidal gridpoints (per period)

        Returns:
            VMECInput.
        """
        logger = logging.getLogger(__name__)
        self.input_filename = input_filename
        self.directory = os.getcwd()

        # Read items from Fortran namelist
        nml = f90nml.read(input_filename)
        nml = nml.get("indata")
        self.nfp = nml.get("nfp")
        mpol_input = nml.get("mpol")
        if (mpol_input < 2):
            logger.error('Error! mpol must be > 1.')
            sys.exit(1)
        self.namelist = nml
        
        self.raxis = nml.get("raxis")
        self.zaxis = nml.get("zaxis")
        self.curtor = nml.get("curtor")
        self.ac_aux_f = nml.get("ac_aux_f")
        self.ac_aux_s = nml.get("ac_aux_s")
        self.pcurr_type = nml.get("pcurr_type")
        self.am_aux_f = nml.get("am_aux_f")
        self.am_aux_s = nml.get("am_aux_s")
        self.pmass_type = nml.get("pmass_type")
        
        self.update_modes(nml.get("mpol"), nml.get("ntor"))

        self.update_grids(ntheta, nzeta)
        self.min_curvature_radius = 0.1
        self.exp_weight = 0.01
        self.animec = False

    def update_namelist(self):
        """
        Updates Fortran namelist with class attributes (except for rbc/zbs)
        
        """
        self.namelist['nfp'] = self.nfp
        self.namelist['raxis'] = self.raxis
        self.namelist['zaxis'] = self.zaxis
        self.namelist['curtor'] = self.curtor
        self.namelist['ac_aux_f'] = self.ac_aux_f
        self.namelist['ac_aux_s'] = self.ac_aux_s
        self.namelist['pcurr_type'] = self.pcurr_type
        self.namelist['am_aux_f'] = self.am_aux_f
        self.namelist['am_aux_s'] = self.am_aux_s
        self.namelist['pmass_type'] = self.pmass_type
        self.namelist['mpol'] = self.mpol
        self.namelist['ntor'] = self.ntor
        if self.animec:
            self.namelist['ah_aux_f'] = self.ah_aux_f
            self.namelist['ah_aux_s'] = self.ah_aux_s
            self.namelist['photp_type'] = self.photp_type
            self.namelist['bcrit'] = self.bcrit

    def update_modes(self, mpol, ntor):
        """
        Updates poloidal and toroidal mode numbers (xm and xn) given maximum
            mode numbers (mpol and ntor)
            
        Args:
            mpol (int): maximum poloidal mode number + 1
            ntor (int): maximum toroidal mode number
        """
        self.mpol = mpol
        self.ntor = ntor
        [self.mnmax, self.xm, self.xn] = init_modes(
                self.mpol - 1, self.ntor)
        [self.rbc,self.zbs] = self.read_boundary_input()
        
    def init_grid(self,ntheta,nzeta):
        thetas = np.linspace(0, 2 * np.pi, ntheta + 1)
        zetas = np.linspace(0,2 * np.pi / self.nfp, nzeta + 1)
        thetas = np.delete(thetas, -1)
        zetas = np.delete(zetas, -1)
        [thetas_2d, zetas_2d] = np.meshgrid(thetas, zetas)
        dtheta = thetas[1] - thetas[0]
        dzeta = zetas[1] - zetas[0]
        
        return thetas_2d,zetas_2d, dtheta, dzeta
        
    def update_grids(self,ntheta,nzeta):
        """
        Updates poloidal and toroidal grids (thetas and zetas) given number of
            grid points
            
        Args:
            ntheta (int): number of poloidal grid points
            nzeta (int): number of toroidal grid points (per period)
        """
        self.ntheta = ntheta
        self.nzeta = nzeta
        self.thetas = np.linspace(0, 2 * np.pi, ntheta + 1)
        self.zetas = np.linspace(0,2 * np.pi / self.nfp, nzeta + 1)
        self.thetas = np.delete(self.thetas, -1)
        self.zetas = np.delete(self.zetas, -1)
        [self.thetas_2d, self.zetas_2d] = np.meshgrid(self.thetas, self.zetas)
        self.dtheta = self.thetas[1] - self.thetas[0]
        self.dzeta = self.zetas[1] - self.zetas[0]
        
        self.zetas_full = np.linspace(0, 2 * np.pi, self.nfp * nzeta + 1)
        self.zetas_full = np.delete(self.zetas_full, -1)
        [self.thetas_2d_full, self.zetas_2d_full] = np.meshgrid(
                self.thetas, self.zetas_full)
        
    def read_boundary_input(self):
        """
        Read in boundary harmonics (rbc, zbs) from Fortran namelist

        Returns:
            rbc (float array): boundary harmonics for radial coordinate
                (same size as xm and xn)
            zbs (float array): boundary harmonics for height coordinate
                (same size as xm and xn)
        """
        rbc_array = np.array(self.namelist["rbc"]).T
        zbs_array = np.array(self.namelist["zbs"]).T

        rbc_array[rbc_array == None] = 0
        zbs_array[zbs_array == None] = 0
        [dim1_rbc,dim2_rbc] = np.shape(rbc_array)
        [dim1_zbs,dim2_zbs] = np.shape(zbs_array)

        [nmin_rbc, mmin_rbc, nmax_rbc, mmax_rbc] = min_max_indices_2d(
                'rbc', self.input_filename)
        nmax_rbc = max(nmax_rbc, -nmin_rbc)
        [nmin_zbs, mmin_zbs, nmax_zbs, mmax_zbs] = min_max_indices_2d(
                'zbs', self.input_filename)
        nmax_zbs = max(nmax_zbs, -nmin_zbs)

        rbc = np.zeros(self.mnmax)
        zbs = np.zeros(self.mnmax)
        for imn in range(self.mnmax):
            if (((self.xn[imn] - nmin_rbc) < dim1_rbc) and \
                    ((self.xm[imn] - mmin_rbc) < dim2_rbc) and \
                    ((self.xn[imn] - nmin_rbc) > -1) and \
                    ((self.xm[imn] - mmin_rbc) > -1)):
                rbc[imn] = rbc_array[int(self.xn[imn] - nmin_rbc), \
                                     int(self.xm[imn] - mmin_rbc)]
            if (((self.xn[imn] - nmin_zbs) < dim1_zbs) and \
                    ((self.xm[imn] - mmin_zbs) < dim2_zbs) and \
                    ((self.xn[imn] - nmin_rbc) > -1) and \
                    ((self.xm[imn] - mmin_rbc) > -1)):
                zbs[imn] = zbs_array[int(self.xn[imn] - nmin_zbs), \
                                     int(self.xm[imn]-mmin_zbs)]

        return rbc, zbs

    def area(self, theta=None, zeta=None):
        """
        Computes area of boundary surface
        
        Args:
            theta (float array): poloidal grid for area evaluation (optional)
            zeta (float array): toroidal grid for area evaluation (optional)
            
        Returns:
            area (float): area of boundary surface
        """
        if (np.shape(theta) != np.shape(zeta)):
            raise ValueError('theta and zeta must have the same shape in area.')
        norm_normal = self.jacobian(theta, zeta)
        area = np.sum(norm_normal) * self.dtheta * self.dzeta * self.nfp
        return area
      
    def volume(self,theta=None,zeta=None):
        """
        Computes volume enclosed by boundary surface
        
        Args:
            theta (float array): poloidal grid for volume evaluation (optional)
            zeta (float array): toroidal grid for volume evaluation (optional)
        
        Returns:
            volume (float): volume enclosed by boundary surface
        """
        if (np.shape(theta) != np.shape(zeta)):
            raise ValueError('theta and zeta must have the same shape in area.')
        [Nx, Ny, Nz] = self.N(theta, zeta)
        [X, Y, Z, R] = self.position(theta, zeta)
        volume = abs(np.sum(Z * Nz)) * self.dtheta * self.dzeta * self.nfp
        return volume
      
    def N(self, theta=None, zeta=None):
        """
        Computes components of normal vector multiplied by surface Jacobian
        
        Args:
            theta (float array): poloidal grid for N evaluation (optional)
            zeta (float array): toroidal grid for N evaluation (optional)
        
        Returns:
            Nx (float array): x component of unit normal multiplied by Jacobian
            Ny (float array): y component of unit normal multiplied by Jacobian
            Nz (float array): z component of unit normal multiplied by Jacobian
        """
        if (np.shape(theta) != np.shape(zeta)):
            raise ValueError('theta and zeta must have the same shape in area.')
        [dxdtheta, dxdzeta, dydtheta, dydzeta, dzdtheta, dzdzeta, dRdtheta, \
                    dRdzeta] = self.position_first_derivatives(theta,zeta)
      
        Nx = -dydzeta * dzdtheta + dydtheta * dzdzeta
        Ny = -dzdzeta * dxdtheta + dzdtheta * dxdzeta
        Nz = -dxdzeta * dydtheta + dxdtheta * dydzeta
        return Nx, Ny, Nz
      
    def position_first_derivatives(self, theta=None, zeta=None):
        """
        Computes derivatives of position vector with respect to angles
        
        Args:
            theta (float array): poloidal grid for evaluation (optional)
            zeta (float array): toroidal grid for evaluation (optional)
        
        Returns:
            dxdtheta (float array): derivative of x wrt poloidal angle
            dxdzeta (float array): derivative of x wrt toroidal angle
            dydtheta (float array): derivative of y wrt poloidal angle
            dydzeta (float array): derivative of y wrt toroidal angle
            dzdtheta (float array): derivative of z wrt poloidal angle
            dzdzeta (float array): derivative of z wrt toroidal angle
            dRdtheta (float array): derivative of R wrt poloidal angle
            dRdzeta (float array): derivative of R wrt toroidal angle

        """
        if (theta is None and zeta is None):
            theta = self.thetas_2d
            zeta = self.zetas_2d
        elif (np.array(theta).shape != np.array(zeta).shape):
            raise ValueError('Incorrect shape for theta and zeta in \
                position_first_derivatives')
        
        R = cosine_IFT(self.xm,self.xn,self.nfp,theta,zeta,self.rbc)
        dRdtheta = sine_IFT(self.xm,self.xn,self.nfp,theta,zeta,-self.xm*self.rbc)
        dzdtheta = cosine_IFT(self.xm,self.xn,self.nfp,theta,zeta,self.xm*self.zbs)
        dRdzeta = sine_IFT(self.xm,self.xn,self.nfp,theta,zeta,self.nfp*self.xn*self.rbc)
        dzdzeta = cosine_IFT(self.xm,self.xn,self.nfp,theta,zeta,-self.nfp*self.xn*self.zbs)
        
#         dRdtheta = np.zeros(np.shape(theta))
#         dzdtheta = np.zeros(np.shape(theta))
#         dRdzeta = np.zeros(np.shape(theta))
#         dzdzeta = np.zeros(np.shape(theta))
#         R = np.zeros(np.shape(theta))
#         for im in range(self.mnmax):
#             angle = self.xm[im] * theta - self.nfp * self.xn[im] * zeta
#             cos_angle = np.cos(angle)
#             sin_angle = np.sin(angle)
#             dRdtheta = dRdtheta - self.xm[im] * self.rbc[im] * sin_angle
#             dzdtheta = dzdtheta + self.xm[im] * self.zbs[im] * cos_angle
#             dRdzeta = dRdzeta + self.nfp * self.xn[im] * self.rbc[im] * sin_angle
#             dzdzeta = dzdzeta - self.nfp * self.xn[im] * self.zbs[im] * cos_angle
#             R = R + self.rbc[im] * cos_angle
        dxdtheta = dRdtheta * np.cos(zeta)
        dydtheta = dRdtheta * np.sin(zeta)
        dxdzeta = dRdzeta * np.cos(zeta) - R * np.sin(zeta)
        dydzeta = dRdzeta * np.sin(zeta) + R * np.cos(zeta)
        return dxdtheta, dxdzeta, dydtheta, dydzeta, dzdtheta, dzdzeta, \
                dRdtheta, dRdzeta

    def jacobian(self, theta=None, zeta=None):
        """
        Computes surface Jacobian (differential area element)
        
        Args:
            theta (float array): poloidal grid for evaluation (optional)
            zeta (float array): toroidal grid for evaluation (optional)
            
        Returns:
            norm_normal (float array): surface Jacobian

        """
        [Nx, Ny, Nz] = self.N(theta, zeta)
        norm_normal = np.sqrt(Nx**2 + Ny**2 + Nz**2)
        return norm_normal
      
    def normalized_jacobian(self, theta=None, zeta=None):
        """
        Computes surface Jacobian normalized by surface area
        
        Args:
            theta (float array): poloidal grid for evaluation (optional)
            zeta (float array): toroidal grid for evaluation (optional)
        
        Returns:
            normalized_norm_normal (float array): normalized surface Jacobian

        """
        norm_normal = self.jacobian(theta, zeta)
        area = self.area(theta, zeta)
        normalized_norm_normal = norm_normal * 4*np.pi * np.pi / area
        return normalized_norm_normal
      
    def normalized_jacobian_derivatives(self, xm_sensitivity, xn_sensitivity,
                                        theta=None, zeta=None):
        """
        Computes derivatives of normalized Jacobian with respect to Fourier
            harmonics of the boundary (rbc, zbs)
        
        Args:
            xm_sensitivity (int array): poloidal modes for derivative
                evaluation
            xn_sensitivity (int array): toroidal modes for derivative
                evaluation
            theta (float array): poloidal grid for evaluation (optional)
            zeta (float array): toroidal grid for evaluation (optional)
            
        Returns:
            dnormalized_jacobiandrmnc (float array): derivatives with respect to
                radius boundary harmonics
            dnormalized_jacobiandzmns (float array): derivatives with respect to
                height boundary harmonics
        
        """
        logger = logging.getLogger(__name__)
        if (theta is None and zeta is None):
            zeta = self.zetas_2d
            theta = self.thetas_2d
        if (theta.ndim != zeta.ndim):
            logger.error('Error! Incorrect dimensions for theta and zeta in '
                         'normalized_jacobian_derivatives.')
            sys.exit(0)
        if (theta.ndim == 1):
            dim1 = len(theta)
            dim2 = 1
        elif (theta.ndim == 2):
            dim1 = len(theta[:, 0])
            dim2 = len(theta[0, :])
        else:
            logger.error('Error! Incorrect dimensions for theta and zeta in '
                         'normalized_jacobian_derivatives.')
            sys.exit(0)
        [dareadrmnc, dareadzmns] = self.area_derivatives(
                    xm_sensitivity, xn_sensitivity, theta, zeta)
        [dNdrmnc, dNdzmns] = self.jacobian_derivatives(
                    xm_sensitivity, xn_sensitivity, theta, zeta)
        area = self.area(theta, zeta)
        N = self.jacobian(theta, zeta)
        
        mnmax_sensitivity = len(xm_sensitivity)
        dnormalized_jacobiandrmnc = np.zeros((mnmax_sensitivity, dim1, dim2))
        dnormalized_jacobiandzmns = np.zeros((mnmax_sensitivity, dim1, dim2))
        for imn in range(mnmax_sensitivity):
            dnormalized_jacobiandrmnc[imn, :, :] = dNdrmnc[imn]/area - \
                    N * dareadrmnc[imn] / (area * area)
            dnormalized_jacobiandzmns[imn, :, :] = dNdzmns[imn]/area - \
                    N * dareadzmns[imn] / (area * area)
        dnormalized_jacobiandrmnc *= 4 * np.pi * np.pi
        dnormalized_jacobiandzmns *= 4 * np.pi * np.pi

        return dnormalized_jacobiandrmnc, dnormalized_jacobiandzmns
        
    def position(self, theta=None, zeta=None):
        """
        Computes position vector
        
        Args:
            theta (float array): poloidal grid for evaluation (optional)
            zeta (float array): toroidal grid for evaluation (optional)
            
        Returns:
            X (float array): x coordinate on angular grid
            Y (float array): y coordinate on angular grid
            Z (float array): z coordinate on angular grid
            R (float array): height coordinate on angular grid
        
        """
        if (theta is None and zeta is None):
            theta = self.thetas_2d
            zeta = self.zetas_2d
        elif (np.array(theta).shape != np.array(zeta).shape):
            raise ValueError(
                'Error! Incorrect dimensions for theta and zeta in position.')
      
        R = cosine_IFT(self.xm,self.xn,self.nfp,theta,zeta,self.rbc)
        Z = sine_IFT(self.xm,self.xn,self.nfp,theta,zeta,self.zbs)
#         R = np.zeros(np.shape(theta))
#         Z = np.zeros(np.shape(zeta))
#         for im in range(self.mnmax):
#             angle = self.xm[im] * theta - self.nfp * self.xn[im] * zeta
#             cos_angle = np.cos(angle)
#             sin_angle = np.sin(angle)
#             R = R + self.rbc[im] * cos_angle
#             Z = Z + self.zbs[im] * sin_angle
        X = R * np.cos(zeta)
        Y = R * np.sin(zeta)
        return X, Y, Z, R
      
    def position_second_derivatives(self, theta=None, zeta=None):
        """
        Computes second derivatives of position vector with respect to the
            toroidal and poloidal angles
            
        Args:
            theta (float array): poloidal grid for evaluation (optional)
            zeta (float array): toroidal grid for evaluation (optional)
            
        Returns:
            d2xdtheta2 (float array): second derivative of x wrt poloidal angle
            d2xdzeta2 (float array): second derivative of x wrt toroidal angle
            d2xdthetadzeta (float array): second derivative of x wrt toroidal
                and poloidal angles
            d2ydtheta2 (float array): second derivative of y wrt poloidal angle
            d2ydzeta2 (float array): second derivative of y wrt toroidal angle
            d2ydthetadzeta (float array): second derivative of y wrt toroidal
                and poloidal angles
            d2Zdtheta2 (float array): second derivative of height wrt poloidal
                angle
            d2Zdzeta2 (float array): second derivative of height wrt toroidal
                angle
            d2Zdthetadzeta (float array): second derivative of height wrt
                toroidal and poloidal angles

        """
        logger = logging.getLogger(__name__)
        if (theta is None and zeta is None):
            theta = self.thetas_2d
            zeta = self.zetas_2d
        elif (np.array(theta).shape != np.array(zeta).shape):
            logger.error('Incorrect shape of theta and zeta in '
                         'position_first_derivatives')
            sys.exit(0)
            
        R = cosine_IFT(self.xm,self.xn,self.nfp,theta,zeta,self.rbc)
        dRdtheta = sine_IFT(self.xm,self.xn,self.nfp,theta,zeta,\
                            -self.xm*self.rbc)
        dzdtheta = cosine_IFT(self.xm,self.xn,self.nfp,theta,zeta,\
                              self.xm*self.zbs)
        dRdzeta = sine_IFT(self.xm,self.xn,self.nfp,theta,zeta,\
                           self.nfp*self.xn*self.rbc)
        dzdzeta = cosine_IFT(self.xm,self.xn,self.nfp,theta,zeta,\
                             -self.nfp*self.xn*self.zbs)
        d2Rdtheta2 = -cosine_IFT(self.xm,self.xn,self.nfp,theta,zeta,\
                                 self.xm*self.xm*self.rbc)
        d2Zdtheta2 = -sine_IFT(self.xm,self.xn,self.nfp,theta,zeta,\
                                 self.xm*self.xm*self.zbs)
        d2Rdzeta2 = -cosine_IFT(self.xm,self.xn,self.nfp,theta,zeta,\
                     self.nfp*self.nfp*self.xn*self.xn*self.rbc)
        d2Zdzeta2 = -sine_IFT(self.xm,self.xn,self.nfp,theta,zeta,\
                     self.nfp*self.nfp*self.xn*self.xn*self.zbs)
        d2Rdthetadzeta = cosine_IFT(self.xm,self.xn,self.nfp,theta,zeta,\
                         self.nfp*self.xm*self.xn*self.rbc)
        d2Zdthetadzeta = sine_IFT(self.xm,self.xn,self.nfp,theta,zeta,\
                         self.nfp*self.xm*self.xn*self.zbs)
        
#         d2Rdtheta2 = np.zeros(np.shape(theta))
#         d2Rdzeta2 = np.zeros(np.shape(theta))
#         d2Zdtheta2 = np.zeros(np.shape(theta))
#         d2Zdzeta2 = np.zeros(np.shape(theta))
#         d2Rdthetadzeta = np.zeros(np.shape(theta))
#         d2Zdthetadzeta = np.zeros(np.shape(theta))
#         dRdtheta = np.zeros(np.shape(theta))
#         dzdtheta = np.zeros(np.shape(theta))
#         dRdzeta = np.zeros(np.shape(theta))
#         dzdzeta = np.zeros(np.shape(theta))
#         R = np.zeros(np.shape(theta))
#         for im in range(self.mnmax):
#             angle = self.xm[im]*theta - self.nfp*self.xn[im]*zeta
#             cos_angle = np.cos(angle)
#             sin_angle = np.sin(angle)
#             R = R + this_rmnc[im]*cos_angle
#             d2Rdtheta2 -= self.xm[im] * self.xm[im] * self.rbc[im] * cos_angle
#             d2Zdtheta2 -= self.xm[im] * self.xm[im] * self.zbs[im] * sin_angle
#             d2Rdzeta2 -= self.nfp * self.nfp * self.xn[im] * self.xn[im] * \
#                     self.rbc[im] * cos_angle
#             d2Zdzeta2 -= self.nfp * self.nfp * self.xn[im] * self.xn[im] * \
#                     self.zbs[im] * sin_angle
#             d2Rdthetadzeta += self.nfp * self.xm[im] * self.xn[im] * \
#                     self.rbc[im] * cos_angle
#             d2Zdthetadzeta += self.nfp * self.xm[im] * self.xn[im] * \
#                     self.zbs[im] * sin_angle
#             dRdtheta -= self.xm[im] * self.rbc[im] * sin_angle
#             dzdtheta += self.xm[im] * self.zbs[im] * cos_angle
#             dRdzeta += self.nfp * self.xn[im] * self.rbc[im] * sin_angle
#             dzdzeta -= self.nfp * self.xn[im] * self.zbs[im] * cos_angle
        d2xdtheta2 = d2Rdtheta2 * np.cos(zeta)
        d2ydtheta2 = d2Rdtheta2 * np.sin(zeta)
        d2xdzeta2 = d2Rdzeta2 * np.cos(zeta) - 2 * dRdzeta * np.sin(zeta) - \
                R * np.cos(zeta)
        d2ydzeta2 = d2Rdzeta2 * np.sin(zeta) + 2 * dRdzeta * np.cos(zeta) - \
                R * np.sin(zeta)
        d2xdthetadzeta = d2Rdthetadzeta * np.cos(zeta) - dRdtheta * np.sin(zeta)
        d2ydthetadzeta = d2Rdthetadzeta * np.sin(zeta) + dRdtheta * np.cos(zeta)
        return d2xdtheta2, d2xdzeta2, d2xdthetadzeta, d2ydtheta2, d2ydzeta2, \
                d2ydthetadzeta, d2Zdtheta2, d2Zdzeta2, d2Zdthetadzeta
        
    def mean_curvature(self, theta=None, zeta=None):
        """
        Computes mean curvature of boundary surface
        
        Args:
            theta (float array): poloidal grid for evaluation (optional)
            zeta (float array): toroidal grid for evaluation (optional)

        Returns:
            H (float array): mean curvature on angular grid
        
        """
        [dxdtheta, dxdzeta, dydtheta, dydzeta, dZdtheta, dZdzeta, dRdtheta, \
                dRdzeta] = self.position_first_derivatives(theta, zeta)
        [d2xdtheta2, d2xdzeta2, d2xdthetadzeta, d2ydtheta2, d2ydzeta2, \
                d2ydthetadzeta, d2Zdtheta2, d2Zdzeta2, d2Zdthetadzeta] = \
                self.position_second_derivatives(theta, zeta)
        
        norm_x = dydtheta * dZdzeta - dydzeta * dZdtheta
        norm_y = dZdtheta * dxdzeta - dZdzeta * dxdtheta
        norm_z = dxdtheta * dydzeta - dxdzeta * dydtheta
        norm_normal = np.sqrt(norm_x**2 + norm_y**2 + norm_z**2)
        nx = norm_x / norm_normal
        ny = norm_y / norm_normal
        nz = norm_z / norm_normal
        E = dxdtheta * dxdtheta + dydtheta * dydtheta + dZdtheta * dZdtheta
        F = dxdtheta * dxdzeta + dydtheta * dydzeta + dZdtheta * dZdzeta
        G = dxdzeta * dxdzeta + dydzeta * dydzeta + dZdzeta * dZdzeta
        e = nx * d2xdtheta2 + ny * d2ydtheta2 + nz * d2Zdtheta2
        f = nx * d2xdthetadzeta + ny * d2ydthetadzeta + nz * d2Zdthetadzeta
        g = nx * d2xdzeta2 + ny * d2ydzeta2 + nz * d2Zdzeta2
        H = (e*G -2*f*F + g*E)/(E*G - F*F)
        return H    
      
    def print_namelist(self, input_filename=None, namelist=None):
        """
        Prints Fortran namelist to text file for passing to VMEC
        
        Args:
            input_filename (str): desired name of text file (optional)
            namelist (f90nml object): namelist to print (optional)
            
        """
        self.update_namelist()
        if (namelist is None):
            namelist = self.namelist
        if (input_filename is None):
            input_filename = self.input_filename
        f = open(input_filename, 'w')
        f.write("&INDATA\n")
        for item in namelist.keys():
            if namelist[item] is not None:
                if (not isinstance(namelist[item], 
                                   (list, tuple, np.ndarray, str))):
                    f.write(item + "=" + str(namelist[item]) + "\n")
                elif (isinstance(namelist[item], str)):
                    f.write(item + "=" + "'" + str(namelist[item]) + "'" + "\n")
                elif (item not in ['rbc', 'zbs']):
                    f.write(item + "=")
                    f.writelines(map(lambda x: str(x) + " ", namelist[item]))
                    f.write("\n")
        # Write RBC
        for imn in range(self.mnmax):
            if (self.rbc[imn] != 0):
                f.write('RBC(' + str(int(self.xn[imn])) + "," \
                     + str(int(self.xm[imn])) + ") = " \
                     + str(self.rbc[imn]) + '\n')
        # Write ZBS
        for imn in range(self.mnmax):
            if (self.zbs[imn] != 0):
                f.write('ZBS(' + str(int(self.xn[imn])) + "," \
                     + str(int(self.xm[imn])) + ") = " \
                     + str(self.zbs[imn]) + '\n')

        f.write("/\n")
        f.close()
        
    def axis_position(self):
        """
        Computes position of initial axis guess
        
        Returns:
            R0 (float array): Radius coordinate of axis on toroidal grid
            Z0 (float array): Height coordinate of axis on toroidal grid
            
        """
        R0 = np.zeros(self.nzeta)
        Z0 = np.zeros(self.nzeta)
        
        if isinstance(self.raxis, (np.ndarray, list)):
            length = len(self.raxis)
        else:
            length = 1
            self.raxis = [self.raxis]
        for im in range(length):
            angle = -self.nfp * im * self.zetas
            R0 += self.raxis[im] * np.cos(angle)

        if isinstance(self.zaxis, (np.ndarray, list)):
            length = len(self.zaxis)
        else:
            length = 1
            self.zaxis = [self.zaxis]

        for im in range(length):
            angle = -self.nfp * im * self.zetas
            Z0 += self.zaxis[im] * np.sin(angle)
     
        return R0, Z0
    
    def test_axis(self):
        """
        Given defining boundary shape, determine if axis guess lies
            inside boundary
            
        Returns:
            inSurface (bool): True if axis guess lies within boundary
            
        """
        [X, Y, Z, R] = self.position()
        [R0, Z0] = self.axis_position()
        
        inSurface = np.zeros(self.nzeta)
        for izeta in range(self.nzeta):
            inSurface[izeta] = point_in_polygon(R[izeta, :], Z[izeta, :], 
                                                R0[izeta], Z0[izeta])
        inSurface = np.all(inSurface)
        return inSurface
      
    def modify_axis(self):
        """
        Given defining boundary shape, attempts to find axis guess which lies
            within the boundary
            
        Returns:
            axisSuccess (bool): True if suitable axis was obtained
            
        """

        logger = logging.getLogger(__name__)
        [X, Y, Z, R] = self.position()
        zetas = self.zetas
          
        angle_1s = np.linspace(0, 2*np.pi, 20)
        angle_2s = np.linspace(0, 2*np.pi, 20)
        angle_1s = np.delete(angle_1s, -1)
        angle_2s = np.delete(angle_2s, -1)
        inSurface = False
        for angle_1 in angle_1s:
            for angle_2 in angle_2s:
                if (angle_1 != angle_2):
                    R0_angle1 = np.zeros(np.shape(zetas))
                    Z0_angle1 = np.zeros(np.shape(zetas))
                    R0_angle2 = np.zeros(np.shape(zetas))
                    Z0_angle2 = np.zeros(np.shape(zetas))
                    for im in range(self.mnmax):
                        angle = self.xm[im] * angle_1 - self.nfp * \
                                self.xn[im] * zetas
                        R0_angle1 = R0_angle1 + self.rbc[im] * np.cos(angle)
                        Z0_angle1 = Z0_angle1 + self.zbs[im] * np.sin(angle)
                        angle = self.xm[im] * angle_2 - self.nfp * \
                                self.xn[im] * zetas
                        R0_angle2 = R0_angle2 + self.rbc[im] * np.cos(angle)
                        Z0_angle2 = Z0_angle2 + self.zbs[im] * np.sin(angle)
                    Raxis = 0.5*(R0_angle1 + R0_angle2)
                    Zaxis = 0.5*(Z0_angle1 + Z0_angle2)
                    # Check new axis shape
                    inSurfaces = np.zeros(self.nzeta)
                    for izeta in range(self.nzeta):
                        inSurfaces[izeta] = point_in_polygon(
                                R[izeta, :], Z[izeta, :], Raxis[izeta], 
                                Zaxis[izeta])
                    inSurface = np.all(inSurfaces)
                    if (inSurface):
                        break
        if (not inSurface):
            logger.warning('Unable to find suitable axis shape in modify_axis.')
            axisSuccess = False
            return axisSuccess
        
        # Now Fourier transform 
        raxis_cc = np.zeros(self.ntor)
        zaxis_cs = np.zeros(self.ntor)
        for n in range(self.ntor):
            angle = -n * self.nfp * zetas
            raxis_cc[n] = np.sum(Raxis * np.cos(angle)) / \
                    np.sum(np.cos(angle)**2)
            if (n > 0):
                zaxis_cs[n] = np.sum(Zaxis * np.sin(angle)) / \
                        np.sum(np.sin(angle)**2)
        self.raxis = raxis_cc
        self.zaxis = zaxis_cs
        axisSuccess = True
        return axisSuccess
    
    def test_boundary(self,theta=None,zeta=None):
        import matplotlib.pyplot as plt
        if (theta is None or zeta is None):
            theta = self.thetas_2d
            zeta = self.zetas_2d
            ntheta = self.ntheta
            nzeta = self.nzeta
        else:
            ntheta = len(theta[0,:])
            nzeta = len(zeta[:,0])
        [X,Y,Z,R] = self.position(theta=theta,zeta=zeta)
        if np.any(R<0):
            return True
        # Compute izeta of intersection
        izeta = boundary_intersect(R,Z)
        if (izeta == -1):
            return False
        else:
            print('Intersection at izeta = '+str(izeta))
            return True
        # Iterate over cross-sections
#         for izeta in range(nzeta):
#             if (self_intersect(R[izeta,:],Z[izeta,:])):
#                 plt.plot(R[izeta,:],Z[izeta,:],marker='*')
#                 print(izeta)
#                 return True
            
    def radius_derivatives(self, xm_sensitivity, xn_sensitivity, theta=None, 
                           zeta=None):
        """
        Computes derivatives of radius with respect to boundary harmoncics
            (rbc,zbs)
            
        Args:
            xm_sensitivity (int array): poloidal modes for derivative
            evaluation
            xn_sensitivity (int array): toroidal modes for derivative
            evaluation
            theta (float array): poloidal grid for evaluation (optional)
            zeta (float array): toroidal grid for evaluation (optional)
        Returns:
            dRdrmnc (float array): derivative of radius with respect to rbc
            dZdzmns (float array): derivative of radius with respect to zbs
            
        """
        logger = logging.getLogger(__name__)
        if (theta is None and zeta is None):
            zeta = self.zetas_2d
            theta = self.thetas_2d
        if (theta.ndim != zeta.ndim):
            logger.error('Error! Incorrect dimensions for theta and zeta '
                         'in radius_derivatives.')
            sys.exit(0)
        if (theta.ndim == 1):
            dim1 = len(theta)
            dim2 = 1
        elif (theta.ndim == 2):
            dim1 = len(theta[:, 0])
            dim2 = len(theta[0, :])
        else:
            logger.error('Error! Incorrect dimensions for theta and zeta '
                         'in radius_derivatives.')
            sys.exit(0)
          
        mnmax_sensitivity = len(xm_sensitivity)
        
        dRdrmnc = np.zeros((mnmax_sensitivity, dim1, dim2))
        dRdzmns = np.zeros((mnmax_sensitivity, dim1, dim2))
        for imn in range(mnmax_sensitivity):
            angle = xm_sensitivity[imn] * theta - \
                    self.nfp * xn_sensitivity[imn] * zeta
            cos_angle = np.cos(angle)
            dRdrmnc[imn, :, :] = cos_angle
        return dRdrmnc, dRdzmns
      
    def volume_derivatives(self, xm_sensitivity, xn_sensitivity, theta=None, 
                           zeta=None):
        """
        Computes derivatives of volume with respect to boundary harmoncics
            (rbc,zbs)
            
        Args:
            xm_sensitivity (int array): poloidal modes for derivative
                evaluation
            xn_sensitivity (int array): toroidal modes for derivative
                evaluation
            theta (float array): poloidal grid for evaluation (optional)
            zeta (float array): toroidal grid for evaluation (optional)
        Returns:
            dRdrmnc (float array): derivative of volume with respect to rbc
            dZdzmns (float array): derivative of volume with respect to zbs
            
        """
        logger = logging.getLogger(__name__)
        if (theta is None and zeta is None):
            zeta = self.zetas_2d
            theta = self.thetas_2d
        if (theta.ndim != zeta.ndim):
            logger.error('Error! Incorrect dimensions for theta and zeta '
                         'in volume_derivatives.')
            sys.exit(0)
        if (theta.ndim == 1):
            dim1 = len(theta)
            dim2 = 1
        elif (theta.ndim == 2):
            dim1 = len(theta[:,0])
            dim2 = len(theta[0,:])
        else:
            logger.error('Error! Incorrect dimensions for theta and zeta '
                         'in volume_derivatives.')
            sys.exit(0)
        [dxdtheta, dxdzeta, dydtheta, dydzeta, dzdtheta, dzdzeta, dRdtheta, \
                dRdzeta] = self.position_first_derivatives(theta, zeta)
        [X, Y, Z, R] = self.position(theta, zeta)
        [Nx, Ny, Nz] = self.N(theta, zeta)
        
        mnmax_sensitivity = len(xm_sensitivity)
        
        d2rdthetadrmnc = np.zeros((3, mnmax_sensitivity, dim1, dim2))
        d2rdzetadrmnc = np.zeros((3, mnmax_sensitivity, dim1, dim2))
        d2rdthetadzmns = np.zeros((3, mnmax_sensitivity, dim1, dim2))
        d2rdzetadzmns = np.zeros((3, mnmax_sensitivity, dim1, dim2))
        dzdzmns = np.zeros((mnmax_sensitivity, dim1, dim2))
        for imn in range(mnmax_sensitivity):
            angle = xm_sensitivity[imn] * theta - \
                    self.nfp * xn_sensitivity[imn] * zeta
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            cos_zeta = np.cos(zeta)
            sin_zeta = np.sin(zeta)
            d2rdthetadrmnc[0,imn,:,:] = -xm_sensitivity[imn] * \
                    sin_angle * cos_zeta
            d2rdthetadrmnc[1,imn,:,:] = -xm_sensitivity[imn] * \
                    sin_angle * sin_zeta
            d2rdzetadrmnc[0,imn,:,:] = self.nfp*xn_sensitivity[imn] * \
                    sin_angle * cos_zeta - cos_angle * sin_zeta
            d2rdzetadrmnc[1,imn,:,:] = self.nfp * xn_sensitivity[imn] * \
                    sin_angle * sin_zeta + cos_angle * cos_zeta
            dzdzmns[imn,:,:] = dzdzmns[imn,:,:] = sin_angle
          
        dNzdrmnc = -d2rdzetadrmnc[0,:,:,:] * dydtheta - \
                dxdzeta * d2rdthetadrmnc[1,:,:,:] + \
                d2rdthetadrmnc[0,:,:,:] * dydzeta + \
                dxdtheta * d2rdzetadrmnc[1,:,:,:] 
        dNzdzmns = -d2rdzetadzmns[0,:,:,:] * dydtheta - \
                dxdzeta * d2rdthetadzmns[1,:,:,:] + \
                d2rdthetadzmns[0,:,:,:] * dydzeta + \
                dxdtheta * d2rdzetadzmns[1,:,:,:]
        dvolumedrmnc = -np.tensordot(dNzdrmnc, Z,axes=2) * self.dtheta * \
                self.dzeta * self.nfp
        dvolumedzmns = -(np.tensordot(dNzdzmns, Z,axes=2) + \
                np.tensordot(dzdzmns, Nz, axes=2)) * self.dtheta * \
                self.dzeta * self.nfp
        return dvolumedrmnc, dvolumedzmns
      
    def area_derivatives(self, xm_sensitivity, xn_sensitivity, theta=None, 
                         zeta=None):
        """
        Computes derivatives of area of boundary with respect to boundary
            harmoncics (rbc,zbs)
            
        Args:
            xm_sensitivity (int array): poloidal modes for derivative
                evaluation
            xn_sensitivity (int array): toroidal modes for derivative
                evaluation
            theta (float array): poloidal grid for evaluation (optional)
            zeta (float array): toroidal grid for evaluation (optional)
        Returns:
            dareadrmnc (float array): derivative of area with respect to rbc
            dareadzmns (float array): derivative of area with respect to zbs
            
        """
        [dNdrmnc, dNdzmns] = self.jacobian_derivatives(
                xm_sensitivity, xn_sensitivity, theta, zeta)
        dareadrmnc = np.sum(dNdrmnc, axis=(1, 2)) * self.dtheta * \
                self.dzeta * self.nfp
        dareadzmns = np.sum(dNdzmns, axis=(1, 2)) * self.dtheta * \
                self.dzeta * self.nfp
        return dareadrmnc, dareadzmns
        # The above 3 statements could be rewritten as 
        # return map(lambda x: np.sum(x, axis=(1, 2)) * self.dtheta * \
        #        self.dzeta * self.nfp, [dNdrmnc, dNdzmns])
        # or for readability
        # def some_fn(x):  # Choose appropriate name for some_fn
        #    return np.sum(x, axis=(1, 2)) * self.dtheta * self.dzeta * self.nfp
        # return map(some_fn, [dNdrmnc, dNdzmns])
      
    # Check that theta and zeta are of correct size
    # Shape of theta and zeta must be the same
    def jacobian_derivatives(self, xm_sensitivity, xn_sensitivity, theta=None, 
                             zeta=None):
        """
        Computes derivatives of surface Jacobian with respect to
            boundary harmoncics (rbc,zbs)
            
        Args:
            xm_sensitivity (int array): poloidal modes for derivative
                evaluation
            xn_sensitivity (int array): toroidal modes for derivative
                evaluation
            theta (float array): poloidal grid for evaluation (optional)
            zeta (float array): toroidal grid for evaluation (optional)
        Returns:
            dNdrmnc (float array): derivative of jacobian with respect to rbc
            dNdzmns (float array): derivative of jacobian with respect to zbs
            
        """
        logger = logging.getLogger(__name__)
        if (theta is None and zeta is None):
            zeta = self.zetas_2d
            theta = self.thetas_2d
        if (theta.ndim != zeta.ndim):
            logger.error('Error! Incorrect dimensions for theta and zeta '
                         'in jacobian_derivatives.')
            sys.exit(0)
        if (theta.ndim == 1):
            dim1 = len(theta)
            dim2 = 1
        elif (theta.ndim == 2):
            dim1 = len(theta[:,0])
            dim2 = len(theta[0,:])
        else:
            logger.error('Error! Incorrect dimensions for theta and zeta '
                         'in jacobian_derivatives.')
            sys.exit(0)
        [dxdtheta, dxdzeta, dydtheta, dydzeta, dzdtheta, dzdzeta, dRdtheta, \
                dRdzeta] = self.position_first_derivatives(theta, zeta)
        [Nx, Ny, Nz] = self.N(theta, zeta)
        N = np.sqrt(Nx*Nx + Ny*Ny + Nz*Nz)
        nx = Nx / N
        ny = Ny / N
        nz = Nz / N
        
        mnmax_sensitivity = len(xm_sensitivity)
        
        d2rdthetadrmnc = np.zeros((3, mnmax_sensitivity, dim1, dim2))
        d2rdzetadrmnc = np.zeros((3, mnmax_sensitivity, dim1, dim2))
        d2rdthetadzmns = np.zeros((3, mnmax_sensitivity, dim1, dim2))
        d2rdzetadzmns = np.zeros((3, mnmax_sensitivity, dim1, dim2))
        for imn in range(mnmax_sensitivity):
            angle = xm_sensitivity[imn] * theta - \
                    self.nfp * xn_sensitivity[imn] * zeta
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            cos_zeta = np.cos(zeta)
            sin_zeta = np.sin(zeta)
            d2rdthetadrmnc[0,imn,:,:] = -xm_sensitivity[imn] * \
                    sin_angle * cos_zeta
            d2rdthetadrmnc[1,imn,:,:] = -xm_sensitivity[imn] * \
                    sin_angle * sin_zeta
            d2rdzetadrmnc[0,imn,:,:] = self.nfp * xn_sensitivity[imn] * \
                    sin_angle * cos_zeta - cos_angle * sin_zeta
            d2rdzetadrmnc[1,imn,:,:] = self.nfp * xn_sensitivity[imn] * \
                    sin_angle * sin_zeta + cos_angle * cos_zeta
            d2rdthetadzmns[2,imn,:,:] = xm_sensitivity[imn] * cos_angle
            d2rdzetadzmns[2,imn,:,:] = -self.nfp * xn_sensitivity[imn] * \
                    cos_angle

        dNdrmnc = (d2rdthetadrmnc[1,:,:,:] * dzdzeta - d2rdthetadrmnc[2,:,:,:]*dydzeta)*nx \
            + (d2rdthetadrmnc[2,:,:,:]*dxdzeta - d2rdthetadrmnc[0,:,:,:]*dzdzeta)*ny \
            + (d2rdthetadrmnc[0,:,:,:]*dydzeta - d2rdthetadrmnc[1,:,:,:]*dxdzeta)*nz \
            + (dydtheta*d2rdzetadrmnc[2,:,:,:] - dzdtheta*d2rdzetadrmnc[1,:,:,:])*nx \
            + (dzdtheta*d2rdzetadrmnc[0,:,:,:] - dxdtheta*d2rdzetadrmnc[2,:,:,:])*ny \
            + (dxdtheta*d2rdzetadrmnc[1,:,:,:] - dydtheta*d2rdzetadrmnc[0,:,:,:])*nz
        dNdzmns = (d2rdthetadzmns[1,:,:,:]*dzdzeta - d2rdthetadzmns[2,:,:,:]*dydzeta)*nx \
            + (d2rdthetadzmns[2,:,:,:]*dxdzeta - d2rdthetadzmns[0,:,:,:]*dzdzeta)*ny \
            + (d2rdthetadzmns[0,:,:,:]*dydzeta - d2rdthetadzmns[1,:,:,:]*dxdzeta)*nz \
            + (dydtheta*d2rdzetadzmns[2,:,:,:] - dzdtheta*d2rdzetadzmns[1,:,:,:])*nx \
            + (dzdtheta*d2rdzetadzmns[0,:,:,:] - dxdtheta*d2rdzetadzmns[2,:,:,:])*ny \
            + (dxdtheta*d2rdzetadzmns[1,:,:,:] - dydtheta*d2rdzetadzmns[0,:,:,:])*nz
        return dNdrmnc, dNdzmns
    
    def global_curvature(self,theta=None,zeta=None):
        if (theta is None or zeta is None):
            theta = self.thetas_2d
            zeta = self.zetas_2d
            dtheta = self.dtheta
            dzeta = self.dzeta
            ntheta = self.ntheta
            nzeta = self.nzeta
        else:            
            dtheta = theta[0,1]-theta[0,0]
            dzeta = zeta[1,0]-zeta[0,0]
            ntheta = len(theta[0,:])
            nzeta = len(zeta[:,0])
        [X, Y, Z, R] = self.position(theta=theta,zeta=zeta)
        [dxdtheta, dxdzeta, dydtheta, dydzeta, dZdtheta, dZdzeta, dRdtheta, \
                dRdzeta] = self.position_first_derivatives(theta=theta,zeta=zeta)
        dldtheta = np.sqrt(dRdtheta**2 + dZdtheta**2)
        tR = dRdtheta/dldtheta
        tz = dZdtheta/dldtheta
        global_curvature_radius = global_curvature_surface(R,Z,tR,tz)
#         global_curvature_radius = np.zeros((nzeta,ntheta))
#         for izeta in range(nzeta):
#             for itheta in range(ntheta):
#                 p1 = np.array([R[izeta,itheta],Z[izeta,itheta]])
#                 this_global_curvature = np.zeros(ntheta)
#                 for ithetap in range(ntheta):
#                     p2 = np.array([R[izeta,ithetap],Z[izeta,ithetap]])
#                     t2 = np.array([tR[izeta,ithetap],tz[izeta,ithetap]])
#                     if (np.all(np.array(p1) != np.array(p2))):
#                         this_global_curvature[ithetap] = self_contact(p1,p2,t2)
#                     else:
#                         this_global_curvature[ithetap] = 1e12
#                 global_curvature_radius[izeta,itheta] = np.min(this_global_curvature)
        return global_curvature_radius
    
    def summed_proximity(self,ntheta=None,nzeta=None):
        """
        Constraint function integrated over the two angles
        """
        if (ntheta == None or nzeta == None):
            theta = self.thetas_2d
            zeta = self.zetas_2d
            dtheta = self.dtheta
            dzeta = self.dzeta
            ntheta = self.ntheta
            nzeta = self.nzeta
        else:            
            [theta, zeta, dtheta, dzeta] = self.init_grid(ntheta,nzeta)
        this_proximity = self.proximity(ntheta=ntheta,nzeta=nzeta)
        return np.sum(this_proximity)*dtheta*dzeta
  
    def summed_proximity_derivatives(self,xm_sensitivity,xn_sensitivity,\
                                     ntheta=None,nzeta=None):
        if (ntheta == None or nzeta == None):
            theta = self.thetas_2d
            zeta = self.zetas_2d
            dtheta = self.dtheta
            dzeta = self.dzeta
        else:            
            [theta, zeta, dtheta, dzeta] = self.init_grid(ntheta,nzeta)

        [dQpdrmnc, dQpdzmns] = self.proximity_derivatives(xm_sensitivity,\
                                                    xn_sensitivity,ntheta,nzeta)
        return np.sum(dQpdrmnc,axis=(0,2))*dtheta*dzeta, \
               np.sum(dQpdzmns,axis=(0,2))*dtheta*dzeta

    def proximity(self,derivatives=False,ntheta=None,nzeta=None):
        if (ntheta == None or nzeta == None):
            theta = self.thetas_2d
            zeta = self.zetas_2d
            dtheta = self.dtheta
            dzeta = self.dzeta
            ntheta = self.ntheta
            nzeta = self.nzeta
        else:            
            [theta, zeta, dtheta, dzeta] = self.init_grid(ntheta,nzeta)
        [X, Y, Z, R] = self.position(theta=theta,zeta=zeta)
        [dxdtheta, dxdzeta, dydtheta, dydzeta, dZdtheta, dZdzeta, dRdtheta, \
                dRdzeta] = self.position_first_derivatives(theta=theta,zeta=zeta)
        dldtheta = np.sqrt(dRdtheta**2 + dZdtheta**2)
        tR = dRdtheta/dldtheta
        tz = dZdtheta/dldtheta
        if derivatives:
            [Qp, dQpdp1, dQpdp2, dQpdtau2, dQpdlprime] = \
                proximity_surface(R,Z,tR,tz,dldtheta,self.min_curvature_radius,\
                                  self.exp_weight,derivatives=True)
            return dtheta*Qp, dtheta*dQpdp1, dtheta*dQpdp2, dtheta*dQpdtau2,\
                   dtheta*dQpdlprime 
        else:
            [Qp, dQpdp1, dQpdp2, dQpdtau2, dQpdlprime] = \
                proximity_surface(R,Z,tR,tz,dldtheta,self.min_curvature_radius,\
                                  self.exp_weight,derivatives=False)
            return dtheta*Qp

    def proximity_derivatives(self,xm_sensitivity,xn_sensitivity,ntheta=None,\
                              nzeta=None):
        if (ntheta == None or nzeta == None):
            theta = self.thetas_2d
            zeta = self.zetas_2d
            dtheta = self.dtheta
            dzeta = self.dzeta
        else:            
            [theta, zeta, dtheta, dzeta] = self.init_grid(ntheta,nzeta)
            
        [Qp, dQpdp1, dQpdp2, dQpdtau2, dQpdlprime] = \
            self.proximity(derivatives=True,ntheta=ntheta,nzeta=nzeta)
        [dxdtheta, dxdzeta, dydtheta, dydzeta, dZdtheta, dZdzeta, dRdtheta, \
                dRdzeta] = self.position_first_derivatives(theta=theta,zeta=zeta)
        lprime = np.sqrt(dRdtheta**2 + dZdtheta**2)
        
        [dQpdrmnc, dQpdzmns] = proximity_derivatives_func(theta,zeta,self.nfp,\
                                      xm_sensitivity,xn_sensitivity,dQpdp1,\
                                      dQpdp2,dQpdtau2,dQpdlprime,dRdtheta,\
                                      dZdtheta,lprime)
        
#         mnmax_sensitivity = len(xm_sensitivity)
#         dQpdrmnc = np.zeros((nzeta,mnmax_sensitivity,ntheta))
#         dQpdzmns = np.zeros((nzeta,mnmax_sensitivity,ntheta))
#         for izeta in range(nzeta):
#             for imn in range(mnmax_sensitivity):
#                 angle = xm_sensitivity[imn]*theta[izeta,:] - self.nfp*xn_sensitivity[imn]*zeta[izeta,:]
#                 dRdrmnc = np.cos(angle)
#                 dZdzmns = np.sin(angle)
#                 d2Rdthetadrmnc = -xm_sensitivity[imn]*np.sin(angle)
#                 d2Zdthetadzmns = xm_sensitivity[imn]*np.cos(angle)
#                 dlprimedrmnc = dRdtheta[izeta,:]*d2Rdthetadrmnc/lprime[izeta,:]
#                 dlprimedzmns = dZdtheta[izeta,:]*d2Zdthetadzmns/lprime[izeta,:]
#                 dtauRdrmnc = d2Rdthetadrmnc/lprime[izeta,:] - dRdtheta[izeta,:]*dlprimedrmnc/lprime[izeta,:]**2
#                 dtauRdzmns = - dRdtheta[izeta,:]*dlprimedzmns/lprime[izeta,:]**2
#                 dtauZdrmnc = - dZdtheta[izeta,:]*dlprimedrmnc/lprime[izeta,:]**2
#                 dtauZdzmns = d2Zdthetadzmns/lprime[izeta,:] - dZdtheta[izeta,:]*dlprimedzmns/lprime[izeta,:]**2
#                 for itheta in range(ntheta):
#                     dQpdrmnc[izeta,imn,itheta] = dQpdp1[izeta,itheta,0]*dRdrmnc[itheta] + np.dot(dQpdp2[izeta,itheta,:,0],dRdrmnc) \
#                         + np.dot(dQpdtau2[izeta,itheta,:,0],dtauRdrmnc) \
#                         + np.dot(dQpdtau2[izeta,itheta,:,1],dtauZdrmnc) \
#                         + np.dot(dQpdlprime[izeta,itheta,:],dlprimedrmnc)
#                     dQpdzmns[izeta,imn,itheta] = dQpdp1[izeta,itheta,1]*dZdzmns[itheta] + np.dot(dQpdp2[izeta,itheta,:,1],dZdzmns) \
#                         + np.dot(dQpdtau2[izeta,itheta,:,0],dtauRdzmns) \
#                         + np.dot(dQpdtau2[izeta,itheta,:,1],dtauZdzmns) \
#                         + np.dot(dQpdlprime[izeta,itheta,:],dlprimedzmns)
        return dQpdrmnc, dQpdzmns

# #TODO : test for correct sizes
# def point_in_polygon(R, Z, R0, Z0):
#     """
#     Determines if point on the axis (R0,Z0) lies within boundary defined by
#         (R,Z), a toroidal slice of the boundary
        
#     Args:
#         R (float array): radius defining toroidal slice of boundary
#         Z (float array): height defining toroidal slice of boundary
#             evaluation
#         R0 (float): radius of trial axis point
#         Z0 (float): height of trial axis point
#     Returns:
#         oddNodes (bool): True if (R0,Z0) lies in (R,Z)
        
#     """

#     ntheta = len(R)
#     oddNodes = False
#     j = ntheta-1
#     for i in range(ntheta):
#         if ((Z[i] < Z0 and Z[j] >= Z0) or (Z[j] < Z0 and Z[i] >= Z0)):
#             if (R[i] + (Z0 - Z[i]) / (Z[j] - Z[i]) * (R[j] - R[i]) < R0):
#                 oddNodes = not oddNodes
#         j = i
#     return oddNodes

# # Note that xn is not multiplied by nfp
# @jit(nopython=True)
# def init_modes(mmax,nmax):
#     mnmax = (nmax+1) + (2*nmax+1)*mmax
#     xm = np.zeros(mnmax)
#     xn = np.zeros(mnmax)

#     # m = 0 modes
#     index = 0
#     for jn in range(nmax+1):
#         xm[index] = 0
#         xn[index] = jn
#         index += 1
  
#     # m /= 0 modes
#     for jm in range(1,mmax+1):
#         for jn in range(-nmax,nmax+1):
#             xm[index] = jm
#             xn[index] = jn
#             index += 1
  
#     return mnmax, xm, xn

# # Computes minimum indices of 2d array in Fortran namelist
# @jit(nopython=True)
# def min_max_indices_2d(varName,inputFilename):
#     varName = varName.lower()
#     index_1 = []
#     index_2 = []
#     with open(inputFilename, 'r') as f:
#         inputFile = f.readlines()
#         for line in inputFile:
#             line3 = line.strip().lower()
#             find_index = line3.find(varName+'(')
#             # Line contains desired varName
#             if (find_index > -1):
#                 out = scanf(varName+"(%d,%d)",line[find_index::].lower())
#                 index_1.append(out[0])
#                 index_2.append(out[1])
#     return min(index_1), min(index_2), max(index_1), max(index_2)

# @jit(nopython=True)
# def proximity_slice(R,Z,tR,tZ,dldtheta,min_curvature_radius,exp_weight,derivatives=False):
#     assert((np.shape(R) == np.shape(Z)) and (np.shape(R) == np.shape(tR)) and (np.shape(R) == np.shape(tZ)) \
#          and (np.shape(R) == np.shape(dldtheta)))
#     ntheta = len(R)
#     Qp = np.zeros((ntheta))
#     if derivatives:
#         dQpdp1 = np.zeros((ntheta,2))
#         dQpdp2 = np.zeros((ntheta,ntheta,2))
#         dQpdtau2 = np.zeros((ntheta,ntheta,2))
#         dQpdlprime = np.zeros((ntheta,ntheta))
#     for itheta in range(ntheta):
#         p1 = [R[itheta],Z[itheta]]
#         Sc = np.zeros(ntheta)
#         if derivatives:
#             dScdp1 = np.zeros((ntheta,2))
#         for ithetap in range(ntheta):
#             p2 = [R[ithetap],Z[ithetap]]
#             tau2 = [tR[ithetap],tZ[ithetap]]
#             if (p1 != p2):
#                 # Sc(p1,p2)
#                 Sc[ithetap] = self_contact_exp(p1,p2,tau2,min_curvature_radius,exp_weight)
#                 if derivatives:
#                     [dScdp1[ithetap,:],dScdp2,dScdtau2] = \
#                         self_contact_exp_gradient(p1,p2,tau2,min_curvature_radius,exp_weight)
#                     dQpdp2[itheta,ithetap,:] = dScdp2*dldtheta[ithetap]
#                     dQpdtau2[itheta,ithetap,:] = dScdtau2*dldtheta[ithetap]
#                     dQpdlprime[itheta,ithetap] = Sc[ithetap]
#         # Integral over ithetap
#         Qp[itheta] = np.dot(Sc,dldtheta)
#         if derivatives:
#             dQpdp1[itheta,:] = np.dot(dScdp1.T,dldtheta)
#     if derivatives:
#         return Qp, dQpdp1, dQpdp2, dQpdtau2, dQpdlprime
#     else:
#         return Qp
    
# @jit(nopython=True)
# def normalDistanceRatio(p1,p2,tau2):
#     p1 = np.array(p1)
#     p2 = np.array(p2)
#     tau2 = np.array(tau2)
#     norm = scipy.linalg.norm(p1-p2)
#     return 1-((p2-p1).dot(tau2))**2/norm**2

# @jit(nopython=True)
# def normalDistanceRatio_gradient(p1,p2,tau2):
#     p1 = np.array(p1)
#     p2 = np.array(p2)
#     tau2 = np.array(tau2)
#     norm = scipy.linalg.norm(p1-p2)
#     dnormalDistanceRatiodp1 = -2*((p2-p1).dot(tau2)/norm)*(-tau2/norm + ((p2-p1).dot(tau2)/norm**3)*(p2-p1))
#     dnormalDistanceRatiodp2 = -2*((p2-p1).dot(tau2)/norm)*( tau2/norm + ((p2-p1).dot(tau2)/norm**3)*(p1-p2))
#     dnormalDistanceRatiodtau2 = -2*((p2-p1).dot(tau2)/norm)*((p2-p1)/norm)
#     return dnormalDistanceRatiodp1,dnormalDistanceRatiodp2,dnormalDistanceRatiodtau2

# @jit(nopython=True)
# def self_contact(p1,p2,tau2):
#     p1 = np.array(p1)
#     p2 = np.array(p2)
#     if (np.all(p1 == p2)):
#         return 1e12
#     norm = scipy.linalg.norm(p1-p2)
#     normal_distance_ratio = normalDistanceRatio(p1,p2,tau2)
# #     assert(normal_distance_ratio>=0) 
# #     assert(normal_distance_ratio<1)
#     if (normal_distance_ratio > 0):
#         return norm/normal_distance_ratio
#     else:
#         return 1e12
  
# # Given 2 points p1 and p2 and tangent at p2, compute self-contact function [(26) in Walker]
# @jit(nopython=True)
# def self_contact_exp(p1,p2,tau2,min_curvature_radius,exp_weight):
#     p1 = np.array(p1)
#     p2 = np.array(p2)
#     norm = scipy.linalg.norm(p1-p2)
#     normal_distance_ratio = normalDistanceRatio(p1,p2,tau2)
#     if (normal_distance_ratio > 0):
#         return np.exp(-(norm/normal_distance_ratio-min_curvature_radius)/exp_weight)
#     else:
#         return 0

# @jit(nopython=True)
# def self_contact_exp_gradient(p1,p2,tau2,min_curvature_radius,exp_weight):
#     p1 = np.array(p1)
#     p2 = np.array(p2)
#     norm = scipy.linalg.norm(p1-p2)
# #     assert(norm!=0)
#     N = normalDistanceRatio(p1,p2,tau2) # norm distance ratio squared
#     [dnormalDistanceRatiodp1,dnormalDistanceRatiodp2,dnormalDistanceRatiodtau2] \
#         = normalDistanceRatio_gradient(p1,p2,tau2)
#     if (N != 0):
#         dScdp1 = (p1-p2)/(norm*N) - norm*dnormalDistanceRatiodp1/(N*N)
#         dScdp2 = (p2-p1)/(norm*N) - norm*dnormalDistanceRatiodp2/(N*N)
#         dScdtau2 = - norm*dnormalDistanceRatiodtau2/(N*N)
#         return -self_contact_exp(p1,p2,tau2,min_curvature_radius,exp_weight)*dScdp1/exp_weight,\
#             -self_contact_exp(p1,p2,tau2,min_curvature_radius,exp_weight)*dScdp2/exp_weight,\
#             -self_contact_exp(p1,p2,tau2,min_curvature_radius,exp_weight)*dScdtau2/exp_weight
#     else:
#         return 0, 0, 0
    
# @jit(nopython=True)
# def self_intersect(x,y):
#     assert(isinstance(x,(list,np.ndarray)))
#     assert(isinstance(y,(list,np.ndarray)))
#     assert(x.ndim==1 and y.ndim==1 and len(x)==len(y))

#     # Make sure there are no repeated points
#     points = (np.array([x,y]).T).tolist()
#     assert (not any(points.count(z) > 1 for z in points))
#     # Repeat last point
#     points.append(points[0])
  
#     npoints = len(points)
#     for i in range(npoints-1):
#         p1 = points[i]
#         p2 = points[i+1]
#         # Compare with all line segments that do not contain x[i] or x[i+1]
#         for j in range(i+2,npoints-1):
#             p3 = points[j]
#             p4 = points[j+1]
#             # Ignore if any points are in common with p1
#             if (p1==p3 or p2==p3 or p1==p4 or p2==p4):
#                 continue
#             if (segment_intersect(p1,p2,p3,p4)):
#                 print(i)
#                 print(j)
#                 return True
#     return False

# # Check if segment defined by (p1, p2) intersections segment defined by (p2, p4)
# def segment_intersect(p1,p2,p3,p4):
  
#     assert(isinstance(p1,(list,np.ndarray)))
#     assert(len(p1)==2 and np.array(p1).ndim==1)
#     assert(isinstance(p2,(list,np.ndarray)))
#     assert(len(p2)==2 and np.array(p2).ndim==1)
#     assert(isinstance(p3,(list,np.ndarray)))
#     assert(len(p3)==2 and np.array(p3).ndim==1)
#     assert(isinstance(p4,(list,np.ndarray)))
#     assert(len(p4)==2 and np.array(p4).ndim==1)
  
#     l1 = np.array(p3) - np.array(p1)
#     l2 = np.array(p2) - np.array(p1)
#     l3 = np.array(p4) - np.array(p1)
  
#     # Check for intersection of bounding box 
#     b1 = [min(p1[0],p2[0]),min(p1[1],p2[1])]
#     b2 = [max(p1[0],p2[0]),max(p1[1],p2[1])]
#     b3 = [min(p3[0],p4[0]),min(p3[1],p4[1])]
#     b4 = [max(p3[0],p4[0]),max(p3[1],p4[1])]
  
#     boundingBoxIntersect = ((b1[0] <= b4[0]) and (b2[0] >= b3[0]) 
#         and (b1[1] <= b4[1]) and (b2[1] >= b3[1]))
#     return (((l1[0]*l2[1]-l1[1]*l2[0])*(l3[0]*l2[1]-l3[1]*l2[0]) < 0) and 
#         boundingBoxIntersect)

