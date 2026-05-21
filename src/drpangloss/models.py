from functools import partial

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
import zodiax as zx
from jax import jit

from ._utils import bessel_jn
from .inference import (
    fisher_matrix as _fisher_matrix,
    laplace_covariance as _laplace_covariance,
)


rad2mas = 180.0 / np.pi * 3600.0 * 1000.0  # convert rad to mas
mas2rad = np.pi / 180.0 / 3600.0 / 1000.0  # convert mas to rad
deg2rad = np.pi / 180.0  # convert deg to rad
rad2deg = 180.0 / np.pi  # convert rad to deg

i2pi = 1j * 2.0 * np.pi


class OIData(zx.Base):
    """
    Store and transform optical-interferometry observables.

    Parameters
    ----------
    data : dict or object
        Either a dictionary with explicit interferometric arrays, or an OIFITS
        object opened with ``pyoifits``.

    Notes
    -----
    The object stores baseline coordinates, observables, uncertainties, and
    optional closure-phase index triplets. It provides convenience methods for
    flattening data/model vectors and converting complex visibilities to the
    configured visibility/phase conventions.
    """

    u: jax.Array
    v: jax.Array
    wavel: jax.Array
    vis: jax.Array
    d_vis: jax.Array
    phi: jax.Array
    d_phi: jax.Array
    i_cps1: jax.Array
    i_cps2: jax.Array
    i_cps3: jax.Array
    v2_flag: bool = eqx.field(static=True)
    cp_flag: bool = eqx.field(static=True)

    def __init__(self, data):
        """
        Initialize from an OIFITS object or explicit arrays.

        Parameters
        ----------
        data : dict or object
            OIFITS data opened with ``pyoifits``, or a dictionary containing
            ``u``, ``v``, ``wavel``, ``vis``, ``d_vis``, ``phi``, ``d_phi``,
            optional closure-phase indices, and convention flags.
        """

        if not isinstance(data, dict):
            # assume data is an oifits file opened with pyoifits
            data_names = [d.name for d in data.get_dataHDUs()]
            assert "OI_VIS" in data_names or "OI_VIS2" in data_names, (
                "No visibility data found in OIFITS file"
            )
            assert "OI_T3" in data_names or "OI_PHI" in data_names, (
                "No phase data found in OIFITS file"
            )

            # get the data from the oifits file
            self.wavel = np.array(
                data[1].data["EFF_WAVE"], dtype=float
            )  # note that for AMI this is scalar but for CHARA it is an array

            # if square visibilities are available, get them, otherwise get unsquared visibilities
            if "OI_VIS2" in data_names:
                visdata = data["OI_VIS2"]
                self.vis = np.array(visdata.data["VIS2DATA"], dtype=float)
                self.d_vis = np.array(visdata.data["VIS2ERR"], dtype=float)
                vis_sta_index = visdata.data["STA_INDEX"]

                self.u, self.v = (
                    np.array(visdata.data["UCOORD"], dtype=float),
                    np.array(visdata.data["VCOORD"], dtype=float),
                )

                self.v2_flag = True

            elif "OI_VIS" in data_names:
                visdata = data["OI_VIS"]
                vis_key = (
                    "VISAMP" if "VISAMP" in visdata.data.names else "VISPHI"
                )
                d_vis_key = (
                    "VISAMPERR"
                    if "VISAMPERR" in visdata.data.names
                    else "VISERR"
                )
                self.vis = np.array(visdata.data[vis_key], dtype=float)
                self.d_vis = np.array(visdata.data[d_vis_key], dtype=float)
                self.u, self.v = (
                    np.array(visdata.data["UCOORD"], dtype=float),
                    np.array(visdata.data["VCOORD"], dtype=float),
                )
                vis_sta_index = np.array(visdata.data["STA_INDEX"], dtype=int)

                self.v2_flag = False

            # if absolute phases are available, get them, otherwise get closure phases
            if "OI_PHI" in data_names:
                phidata = data["OI_PHI"]
                self.phi = np.array(phidata.data["VISPHI"], dtype=float)
                self.d_phi = np.array(phidata.data["VISERR"], dtype=float)
                self.i_cps1, self.i_cps2, self.i_cps3 = None, None, None

                self.cp_flag = False

            elif "OI_T3" in data_names:
                phidata = data["OI_T3"]
                self.phi = np.array(phidata.data["T3PHI"], dtype=float)
                self.d_phi = np.array(phidata.data["T3PHIERR"], dtype=float)

                cp_sta_index = np.array(phidata.data["STA_INDEX"], dtype=int)
                self.i_cps1, self.i_cps2, self.i_cps3 = cp_indices(
                    vis_sta_index, cp_sta_index
                )

                self.cp_flag = True

        else:
            # assume data is a dict of the form {'u':u,'v':v,'wavel':wavel,'vis':vis,'d_vis':d_vis,
            #'phi':phi,'d_phi':d_phi,'i_cps1':i_cps1,'i_cps2':i_cps2,'i_cps3':i_cps3,'v2_flag':v2_flag,'cp_flag':cp_flag}

            self.u = np.array(data["u"], dtype=float)
            self.v = np.array(data["v"], dtype=float)
            self.wavel = np.array(data["wavel"], dtype=float)

            self.vis = np.array(data["vis"], dtype=float)
            self.d_vis = np.array(data["d_vis"], dtype=float)

            self.phi = np.array(data["phi"], dtype=float)
            self.d_phi = np.array(data["d_phi"], dtype=float)

            try:
                idx1 = data["i_cps1"]
                idx2 = data["i_cps2"]
                idx3 = data["i_cps3"]
                if idx1 is None or idx2 is None or idx3 is None:
                    raise KeyError
                self.i_cps1 = np.array(idx1, dtype=int)
                self.i_cps2 = np.array(idx2, dtype=int)
                self.i_cps3 = np.array(idx3, dtype=int)
            except KeyError:
                self.i_cps1 = None
                self.i_cps2 = None
                self.i_cps3 = None

            self.v2_flag = bool(data.get("v2_flag", True))
            self.cp_flag = bool(data.get("cp_flag", self.i_cps1 is not None))

    def __repr__(self):
        """Return a compact string summary of the loaded interferometric data."""
        phname = "CP" if self.cp_flag else "Phi"
        visname = "V2" if self.v2_flag else "Vis"
        return (
            f"OIData(u={self.u}, v={self.v}, {phname}={self.phi}, d_{phname}={self.d_phi}, "
            f"{visname}={self.vis}, d_{visname}={self.d_vis}, "
            f"i_cps1={self.i_cps1}, i_cps2={self.i_cps2}, i_cps3={self.i_cps3})"
        )

    def flatten_data(self):
        """
        Flatten closure phases and uncertainties.
        """
        return np.concatenate([self.vis, self.phi]), np.concatenate(
            [
                self.d_vis,
                self.d_phi,
            ]
        )

    def unpack_all(self):
        """
        Unpack all data to be used in some legacy model functions.
        """
        return (
            self.u / self.wavel,
            self.v / self.wavel,
            self.phi,
            self.d_phi,
            self.vis,
            self.d_vis,
            self.i_cps1,
            self.i_cps2,
            self.i_cps3,
        )

    def flatten_model(self, cvis):
        """
        Flatten model visibilities and phases.

        Parameters
        ----------
        cvis : array-like
            Complex visibilities from a model evaluation.

        Returns
        -------
        array-like
            Concatenated visibility and phase model vector in the same
            convention/order as ``flatten_data``.
        """

        return np.concatenate([self.to_vis(cvis), self.to_phases(cvis)])

    def to_vis(self, cvis):
        """
        Convert complex visibilities to visibilities or squared visibilities.
        """
        if self.v2_flag:
            return np.abs(cvis) ** 2
        else:
            return np.abs(cvis)

    def to_phases(self, cvis):
        """
        Convert complex visibilities to closure phases or absolute phases.
        """
        if self.cp_flag:
            return closure_phases(cvis, self.i_cps1, self.i_cps2, self.i_cps3)
        else:
            return np.rad2deg(np.angle(cvis))

    def model(self, model_object):
        """
        Compute the model visibilities and phases for the given model object.
        """
        cvis = model_object.model(self.u, self.v, self.wavel)
        return self.flatten_model(cvis)


class BinaryModelAngular(zx.Base):
    """
    Represent a binary companion using angular separation and position angle.

    Parameters
    ----------
    sep : float or array-like
        On-sky separation in milliarcseconds.
    pa : float or array-like
        Position angle in degrees, measured East of North.
    contrast : float or array-like
        Brightness contrast ratio ``star/companion``.

    Notes
    -----
    This parameterization is often convenient for reporting astrophysical
    constraints directly in polar-like coordinates. The model evaluates complex
    visibilities on the provided interferometric baseline geometry.
    """

    sep: jax.Array
    pa: jax.Array
    contrast: jax.Array

    def __init__(self, sep, pa, contrast):
        """
        Initialize a binary model in angular coordinates.

        Parameters
        ----------
        sep : float or array-like
            Separation in milliarcseconds.
        pa : float or array-like
            Position angle in degrees.
        contrast : float or array-like
            Contrast ratio between primary and companion (``star/companion``).

        """

        self.sep = np.asarray(sep, dtype=float)
        self.pa = np.asarray(pa, dtype=float)
        self.contrast = np.asarray(contrast, dtype=float)

    def __repr__(self):
        """Return a readable representation of binary angular parameters."""
        return f"BinaryModel(sep={self.sep}, pa={self.pa}, contrast={self.contrast})"

    def unpack_all(self):
        """
        Return all model parameters in angular form.

        Returns
        -------
        tuple[array-like, array-like, array-like]
            Tuple ``(sep, pa, contrast)``.
        """
        return self.sep, self.pa, self.contrast

    def model(self, u, v, wavel):
        """
        Evaluate complex visibilities for this angular binary model.

        Parameters
        ----------
        u : array-like
            Baseline ``u`` coordinates in meters.
        v : array-like
            Baseline ``v`` coordinates in meters.
        wavel : array-like
            Effective wavelength(s) in meters.

        Returns
        -------
        array-like
            Complex visibility samples on the provided baselines.
        """
        uu, vv = u / wavel, v / wavel
        return cvis_binary_angular(uu, vv, self.sep, self.pa, self.contrast)


class BinaryModelCartesian(zx.Base):
    """
    Represent a binary companion using Cartesian sky offsets.

    Parameters
    ----------
    dra : float or array-like
        Right-ascension offset in milliarcseconds.
    ddec : float or array-like
        Declination offset in milliarcseconds.
    flux : float or array-like
        Companion-to-primary flux ratio.

    Notes
    -----
    This parameterization is useful for optimization and inference workflows
    that operate directly in Cartesian offsets.
    """

    dra: jax.Array
    ddec: jax.Array
    flux: jax.Array

    def __init__(self, dra, ddec, flux):
        """
        Initialize a binary model in Cartesian offsets.

        Parameters
        ----------
        dra : float or array-like
            Right-ascension offset in milliarcseconds.
        ddec : float or array-like
            Declination offset in milliarcseconds.
        flux : float or array-like
            Flux ratio for the companion component.

        """

        self.dra = np.asarray(dra, dtype=float)
        self.ddec = np.asarray(ddec, dtype=float)
        self.flux = np.asarray(flux, dtype=float)

    def __repr__(self):
        """Return a readable representation of binary Cartesian parameters."""
        return f"BinaryModelCartesian(dra={self.dra}, ddec={self.ddec}, flux={self.flux})"

    def unpack_all(self):
        """
        Return all model parameters in Cartesian form.

        Returns
        -------
        tuple[array-like, array-like, array-like]
            Tuple ``(dra, ddec, flux)``.
        """
        return self.dra, self.ddec, self.flux

    def model(self, u, v, wavel):
        """
        Evaluate complex visibilities for this Cartesian binary model.

        Parameters
        ----------
        u : array-like
            Baseline ``u`` coordinates in meters.
        v : array-like
            Baseline ``v`` coordinates in meters.
        wavel : array-like
            Effective wavelength(s) in meters.

        Returns
        -------
        array-like
            Complex visibility samples on the provided baselines.
        """
        uu, vv = u / wavel, v / wavel
        return cvis_binary(uu, vv, self.ddec, self.dra, self.flux)


# TODO: add azimuthal modulations
class BinaryGaussianRimModel(zx.Base):
    r"""
    Represents a chromatic 'disk' rim surrounding a binary star.
    The primary and secondary star are modelled as point sources. The primary
    has an additional spectral index slope, while the secondary is considered
    grey. The rim is modelled as a (potentially azimuthaly modulated) Gaussian
    rim with its own spectral index slope. In addition, allows one to add a
    grey overresolved flux background.

    Parameters
    ----------
    flux_p : float or array-like
        Total flux fraction of the primary.
    dra_p : float or array-like
        Right-ascension offset of the primary in milliarcseconds.
    ddec_p : float or array-like
        Declination offset of the primary in milliarcseconds. Calculated relative
        to the center of the the disk rim.
    si_p : float or array-like
        Spectral index of the primary.
    flux_s : float or array-like
        Total flux fraction of the grey secondary.
    dra_s : float or array-like
        Right-ascension offset of the secondary in milliarcseconds.
    ddec_s : float or array-like
        Declination offset of the secondary in milliarcseconds. Calculated relative
        to the center of the the disk rim.
    diam_rim: float or array-like
        Diameter of the rim in milliarcseconds.
    fwhm_rim: float or array-like
        Gaussian FWHM of the rim in milliarcseconds.
    inc_rim: float or array-like
        Apparent inclination of the rim in degrees.
    pa_rim: float or array-like
        Position angle of the rim's projected major axis in degrees, measured North to
        East (i.e. counter-clockwise in conventional astronomical image orientation).
    si_rim: float or array-like
        Spectral index of the rim.
    flux_bkg: float or array-like
        Total flux fraction of the grey overresolved background.
    az_amps: array-like
        1D array containing amplitude coefficients for rim cosine azimuthal modulations.
        The first element is seen as the amplitude for the first-order modulation,
        the second as the amplitude for the second-order modulation, etc.
    az_phis: array-like
        1D array containing offset angles of the rim's cosine azimuthal modulations,
        relative to the position angle of the rim's projected major axis, in
        degrees. The first element is seen as the offset for the first-order
        modulation, the second for the second-order modulation, etc.
    wave0: float or array-like
        Wavelength at which the total flux fractions are defined in meters. Note
        that this is not a free parameter, but is just fixed at initialization.

    Notes
    -----
    * The intensity profiles is separable into a symmetric radial profile and cosine
    azimuthal modulations, meaning the image intensity can be described in polar image
    coordinates as $I(r, \theta) = f(r) \left( 1 + \sum_{m=0}^{n}
    A_m \cos{(m(\theta - \phi_m)} \right)$, where $f(r)$ describes a Gaussian radial
    intensity profile.

    * Note that the spectral slope is defined in the wavelength-formulation of flux,
    i.e. $d$ in $F_{\lambda} \propto \lambda^{d}$.
    """

    flux_p: jax.Array
    dra_p: jax.Array
    ddec_p: jax.Array
    si_p: jax.Array
    flux_s: jax.Array
    dra_s: jax.Array
    ddec_s: jax.Array
    diam_rim: jax.Array
    fwhm_rim: jax.Array
    inc_rim: jax.Array
    pa_rim: jax.Array
    si_rim: jax.Array
    flux_bkg: jax.Array
    az_amps: jax.Array
    az_phis: jax.Array
    wave0: jax.Array = eqx.field(static=True)

    def __init__(
        self,
        flux_p,
        dra_p,
        ddec_p,
        si_p,
        flux_s,
        dra_s,
        ddec_s,
        diam_rim,
        fwhm_rim,
        inc_rim,
        pa_rim,
        si_rim,
        flux_bkg,
        az_amps,
        az_phis,
        wave0,
    ):
        """
        Initialize a cirumbinary Gaussian rim model.

        Parameters
        ----------
        flux_p : float or array-like
            Total flux fraction of the primary.
        dra_p : float or array-like
            Right-ascension offset of the primary in milliarcseconds.
        ddec_p : float or array-like
            Declination offset of the primary in milliarcseconds. Calculated relative
            to the center of the the disk rim.
        si_p : float or array-like
            Spectral index of the primary.
        flux_s : float or array-like
            Total flux fraction of the grey secondary.
        dra_s : float or array-like
            Right-ascension offset of the secondary in milliarcseconds.
        ddec_s : float or array-like
            Declination offset of the secondary in milliarcseconds. Calculated relative
            to the center of the the disk rim.
        diam_rim: float or array-like
            Diameter of the rim in milliarcseconds.
        fwhm_rim: float or array-like
            Gaussian FWHM of the rim in milliarcseconds.
        inc_rim: float or array-like
            Apparent inclination of the rim in degrees.
        pa_rim: float or array-like
            Position angle of the rim's projected major axis in degrees, measured North to
            East (i.e. counter-clockwise in conventional astronomical image orientation).
        si_rim: float or array-like
            Spectral index of the rim.
        flux_bkg: float or array-like
            Total flux fraction of the grey overresolved background.
        az_amps: array-like
            1D array containing amplitude coefficients for rim cosine azimuthal modulations.
            The first element is seen as the amplitude for the first-order modulation,
            the second as the amplitude for the second-order modulation, etc.
        az_phis: array-like
            1D array containing offset angles of the rim's cosine azimuthal modulations,
            relative to the position angle of the rim's projected major axis, in
            degrees. The first element is seen as the offset for the first-order
            modulation, the second for the second-order modulation, etc.
        wave0: float or array-like
            Wavelength at which the total flux fractions are defined in meters. Note
            that this is not a free parameter, but is just fixed at initialization.
        """
        self.flux_p = np.asarray(flux_p, dtype=float)
        self.dra_p = np.asarray(dra_p, dtype=float)
        self.ddec_p = np.asarray(ddec_p, dtype=float)
        self.si_p = np.asarray(si_p, dtype=float)
        self.flux_s = np.asarray(flux_s, dtype=float)
        self.dra_s = np.asarray(dra_s, dtype=float)
        self.ddec_s = np.asarray(ddec_s, dtype=float)
        self.diam_rim = np.asarray(diam_rim, dtype=float)
        self.fwhm_rim = np.asarray(fwhm_rim, dtype=float)
        self.inc_rim = np.asarray(inc_rim, dtype=float)
        self.pa_rim = np.asarray(pa_rim, dtype=float)
        self.si_rim = np.asarray(si_rim, dtype=float)
        self.flux_bkg = np.asarray(flux_bkg, dtype=float)
        self.az_amps = np.asarray(az_amps, dtype=float)
        self.az_phis = np.asarray(az_phis, dtype=float)
        self.wave0 = np.asarray(wave0, dtype=float)

    def __repr__(self):
        """Return a readable representation of the model parameters."""
        repr_str = (
            f"BinaryGaussianRimModel(flux_p={self.flux_p}, dra_p={self.dra_p}, "
            f"ddec_p={self.ddec_p}, si_p={self.si_p}, flux_s={self.flux_s}, "
            f"dra_s={self.dra_s}, ddec_s={self.ddec_s}, diam_rim={self.diam_rim}, "
            f"fwhm_rim={self.fwhm_rim}, inc_rim={self.inc_rim}, pa_rim={self.pa_rim}, "
            f"si_rim={self.si_rim}, flux_bkg={self.flux_bkg}, az_amps={self.az_amps},"
            f"az_phis={self.az_phis}, wave0={self.wave0}"
        )
        return repr_str

    def unpack_all(self):
        """
        Return all model parameters.

        Returns
        -------
        tuple[array-like, array-like, array-like, array-like, array-like,
              array-like, array-like, array-like, array-like, array-like,
              array-like, array-like, array-like, array-like, array_like,
              array_like]
            Tuple ``(flux_p, dra_p, ddec_p, si_p, flux_s, dra_s, ddec_s,
                     diam_rim, fwhm_rim, inc_rim, pa_rim, si_rim, flux_bkg, az_amps,
                     az_phis)``.
        """
        return (
            self.flux_p,
            self.dra_p,
            self.ddec_p,
            self.si_p,
            self.flux_s,
            self.dra_s,
            self.ddec_s,
            self.diam_rim,
            self.fwhm_rim,
            self.inc_rim,
            self.pa_rim,
            self.si_rim,
            self.flux_bkg,
            self.az_amps,
            self.az_phis,
        )

    def model(self, u, v, wavel):
        """
        Evaluate complex visibilities for this Binary with Gaussian rim model.

        Parameters
        ----------
        u : array-like
            Baseline ``u`` coordinates in meters.
        v : array-like
            Baseline ``v`` coordinates in meters.
        wavel : array-like
            Effective wavelength(s) in meters.

        Returns
        -------
        array-like
            Complex visibility samples on the provided baselines.
        """
        uu, vv = u / wavel, v / wavel

        # Complex visbilities for Gaussian rim.
        cvis_rim = cvis_gaussian_rim(
            uu,
            vv,
            0.0,
            0.0,
            self.diam_rim,
            self.fwhm_rim,
            self.inc_rim,
            self.pa_rim,
            self.az_amps,
            self.az_phis,
        )
        # Complex visibilities for binary components.
        dra_p_rad, ddec_p_rad = self.dra_p * mas2rad, self.ddec_p * mas2rad
        dra_s_rad, ddec_s_rad = self.dra_s * mas2rad, self.ddec_s * mas2rad
        cvis_p = np.exp(-i2pi * (uu * dra_p_rad + vv * ddec_p_rad))
        cvis_s = np.exp(-i2pi * (uu * dra_s_rad + vv * ddec_s_rad))

        # Calculate spectra for each component.
        flux_rim = 1 - self.flux_p - self.flux_s - self.flux_bkg
        spec_rim = flux_rim * (wavel / self.wave0) ** self.si_rim
        spec_p = self.flux_p * (wavel / self.wave0) ** self.si_p
        spec_s = np.full(spec_rim.shape, self.flux_s)
        spec_bkg = np.full(spec_rim.shape, self.flux_bkg)

        # Combine into spectral-weighted total complex visibility.
        cvis_tot = (
            spec_rim * cvis_rim + spec_p * cvis_p + spec_s * cvis_s
        ) / (spec_rim + spec_p + spec_s + spec_bkg)

        return cvis_tot


def cvis_binary_angular(u, v, sep, pa, contrast):
    # adapted from pymask
    """Compute complex visibilities for an angular-parameterized binary model.

    Parameters
    ----------
    u : array-like
        Baseline ``u`` coordinates in wavelength units.
    v : array-like
        Baseline ``v`` coordinates in wavelength units.
    sep : float or array-like
        Separation in milliarcseconds.
    pa : float or array-like
        Position angle in degrees.
    contrast : float or array-like
        Contrast ratio ``star/companion``.

    Returns
    -------
    array-like
        Complex visibility samples.
    """

    # normalize visibilities so total power is 1

    th = pa * deg2rad

    ddec = mas2rad * (sep * np.cos(th))
    dra = -1 * mas2rad * (sep * np.sin(th))

    # decompose into two "luminosity"
    l2 = 1.0 / (contrast + 1)
    l1 = 1 - l2

    # phase-factor
    phi = np.exp(-i2pi * (u * dra + v * ddec))
    cvis = l1 + l2 * phi

    return cvis


def cvis_binary(u, v, ddec, dra, planet):
    # adapted from pymask
    """Compute complex visibilities for a Cartesian-parameterized binary model.

    Parameters
    ----------
    u : array-like
        Baseline ``u`` coordinates in wavelength units.
    v : array-like
        Baseline ``v`` coordinates in wavelength units.
    ddec : float or array-like
        Declination offset in milliarcseconds.
    dra : float or array-like
        Right-ascension offset in milliarcseconds.
    planet : float or array-like
        Flux ratio of the companion.

    Returns
    -------
    array-like
        Complex visibility samples.
    """

    star = 1

    # normalize visibilities so total power is 1
    p3 = star / (star + planet)
    p2 = planet / (star + planet)

    # relative locations
    ddec = ddec * np.pi / (180.0 * 3600.0 * 1000.0)
    dra = dra * np.pi / (180.0 * 3600.0 * 1000.0)
    phi_r = np.cos(-2 * np.pi * (u * dra + v * ddec))
    phi_i = np.sin(-2 * np.pi * (u * dra + v * ddec))

    cvis = p3 + p2 * phi_r + p2 * phi_i * 1.0j

    return cvis


def cvis_gaussian_rim(u, v, dra, ddec, diam, fwhm, inc, pa, az_amps, az_phis):
    """Compute complex visibilities for a (modulated) Gaussian rim.

    Parameters
    ----------
    u : array-like
        Baseline ``u`` coordinates in wavelength units.
    v : array-like
        Baseline ``v`` coordinates in wavelength units.
    dra : float or array-like
        Right-ascension offset of the rim in milliarcseconds.
    ddec : float or array-like
        Declination offset of the rim in milliarcseconds.
    diam: float or array-like
        Diameter of the rim in milliarcseconds.
    fwhm: float or array-like
        Gaussian FWHM of the rim in milliarcseconds.
    inc: float or array-like
        Apparent inclination of the rim in degrees.
    pa: float or array-like
        Position angle of the rim's projected major axis in degrees, measured North to
        East (i.e. counter-clockwise in conventional astronomical image orientation).
    az_amps: array-like
        1D array containing amplitude coefficients for cosine azimuthal modulations.
        The first element is seen as the amplitude for the first-order modulation,
        the second as the amplitude for the second-order modulation, etc.
    az_phis: array-like
        1D array containing offset angles of the cosine azimuthal modulations,
        relative to the position angle of the rim's projected major axis, in
        degrees. The first element is seen as the offset for the first-order
        modulation, the second for the second-order modulation, etc.

    Returns
    -------
    array-like
        Complex visibility samples.
    """
    # Relevant changes of units
    inc_rad, pa_rad, dra_rad, ddec_rad = (
        inc * deg2rad,
        pa * deg2rad,
        dra * deg2rad,
        ddec * deg2rad,
    )

    # Transform spatial frequency coordinates to frame of reference where the model rim
    # is uninclined and the major axis is pointed North (postitive y-axis).
    stretch_factor = np.cos(inc_rad)

    # Apply rotation matrix and stretch factor (latter for projected minor rim axis)
    ut = u * np.cos(pa_rad) + v * np.sin(pa_rad)
    vt = stretch_factor * (-u * np.sin(pa_rad) + v * np.cos(pa_rad))

    # NOTE: we consider the radial profile out to 5 sigma from the peak, i.e. ~3.73E-6
    # of the peak flux. Unless we're dealing with even higher contrast observations, we
    # should be fine, but should be kept in mind that this is hardcoded for now.
    std_rim = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    rin, rout = (diam / 2.0) - 5.0 * std_rim, (diam / 2.0) + 5.0 * std_rim
    # Make sure lower bound is not negative here.
    rin = np.max(0.0, rin)

    # Create radial position profile in milliarcseconds.
    # NOTE: radial steps are 2% of the rim's sigma in size. This is hardcoded, but
    # should be fine for any reasonable interferomter configuration and rim
    # we expect to use/resolve in the near future.
    rpos = np.linspace(rin, rout, 500)

    # Get Gaussian intensity profile.
    rprof = 1.0 * np.exp((rpos - diam / 2.0) ** 2.0 / 2.0 * std_rim**2.0)

    # Compute complex visibilities.
    cvis = cvis_radial_profile_modulated(ut, vt, rpos, rprof, az_amps, az_phis)

    # Apply offset phase-factor.
    phi = np.exp(-i2pi * (u * dra_rad + v * ddec_rad))
    cvis *= phi

    return cvis


# TODO: implement
# TODO: JIT this one?
def cvis_radial_profile_modulated(
    u, v, rpos, intensity, az_amps, az_phis, *, nbase=100
):
    r"""Compute the complex visibility for a object whose intensity profiles is
    separable into a symmetric radial profile and cosine azimuthal modulations,
    meaning the image intensity can be described in polar image coordinates as
    $I(r, \theta) = f(r) \left( 1 + \sum_{m=0}^{n} A_m \cos{(m(\theta - \phi_m)}
    \right)$, where $f(r)$ describes the base radial intensity profile.

    Parameters
    ----------
    u : array-like
        Baseline ``u`` coordinates in wavelength units.
    v : array-like
        Baseline ``v`` coordinates in wavelength units.
    rpos : array-like
        Radial coordinate positions of the radial profile in milliarcseconds.
    intensity: array-like
        Radial intensity profile defined at the ``rpos`` positions.
    az_amps: array-like
        1D array containing amplitude coefficients for cosine azimuthal modulations.
        The first element is seen as the amplitude for the first-order modulation,
        the second as the amplitude for the second-order modulation, etc.
    az_phis: array-like
        1D array containing offset angles of the cosine azimuthal modulations,
        relative to the position angle of the rim's projected major axis, in
        degrees. The first element is seen as the offset for the first-order
        modulation, the second for the second-order modulation, etc.
    nbase: int
        Number of baseline length values to consider during the calculation. The
        required Hankel transforms will then be calculated only for ``nbase`` baseline
        lengths, and then lineary interpolated to the baseline lenghts corresponding
        to the given ``u`` and ``v``

    Notes
    -----
    The radial profile is used in trapezoidal quadrature to calculate the Hankel
    transforms. It's best to make sure that ``rpos`` resolves the radial intensity
    profile fairly well.

    This function does not account for rotation or geometric stretching (e.g. due to
    inclination). A separate transformation of $uv$ coordinates should account for this.
    The phase angles of the cosine modulations are defined relative to the
    spatial y-axis (North), turning counterclockwise to the x-axis (East). This means
    that a single 0-th order modulation with a phase angle of $0 \, \mathrm{deg}$
    results in a bright peak towards the North, and a faint peak towards the South.
    A phase angle of $90 \, \mathrm{deg}$ would result in a bright peak towards the
    East, and a faint one towards the West.
    """

    # NOTE: maybe you do not wish to calculate the Hankel transform for every single
    # single UV value here, might just take a subsample and then linearly interpolate
    # cause it'll be really expensive otherwise

    pass


@partial(jit, static_argnames=["n"])
def hankel_n(n, u, v, rpos, intensity):
    """Function to compute the $n$-th order Hankel transform of given radial intensity
    profile. For $n=0$, this returns the complex visibility of a centro-symmetric
    object with the given radial intensity profile.

    Parameters
    ----------
    n : int
        The order of the Hankel transform to calculate.
    u : array-like
        Baseline ``u`` coordinates in wavelength units.
    v : array-like
        Baseline ``v`` coordinates in wavelength units.
    rpos : array-like
        Radial coordinate positions of the radial profile in milliarcseconds.
    intensity: array-like
        Radial intensity profile defined at the ``rpos`` positions.

    Returns
    -------
    array-like
        $n$-th order Hankel transform evaluated at the specified spatial frequencies
        ``u`` and ``v``.
    """
    rpos_rad = rpos * mas2rad  # Put radial positions in radian.
    base_norm = np.hypot(
        u, v
    )  # Baseline norm in wavelength units (cycles/rad).

    # Calculate the scalar Hankel transform normalization factor (only needs to be
    # computed once).
    hankel_fnorm = np.trapezoid(intensity * rpos_rad, rpos_rad)

    # Broadcast multiply into a 2D kernel of shape (Nb, Nr) containing all possible
    # multiplied versions of baseline and radial intensity position.
    x = 2.0 * np.pi * base_norm[:, None] * rpos_rad[None, :]

    # Calculate bessel function for each element in x array.
    kernel = bessel_jn(n, x)

    # Set up array of integrands where we have to radially integrate over the second
    # axis with broadcasting (i.e. shape (Nb, Nr)).
    integrand_arr = intensity[None, :] * kernel * rpos_rad[None, :]

    # Integrate each row across radial axis, collecting the result for each into
    # vector of shape (Nb,), giving the normalized n-th Hankel transform for each
    # baseline.
    hankel_n = np.trapezoid(integrand_arr, rpos_rad, axis=1) / hankel_fnorm

    return hankel_n


# TODO: will need an alternative log-like which takes into account parameters being
# fixed, and also shared accross epochs.


def loglike(values, params, data_obj, model_class):
    """
    Abstract log-likelihood function for a given model class and data object, assuming Gaussian errors.

    Parameters
    ----------
    values : array-like
        Values of the model parameters.
    params : list
        List of parameter names.
    data_obj : OIData
        Object containing the data to be fitted.
    model_class : class
        Model class to be fitted to the data.

    Returns
    -------
    float
        Log-likelihood value.
    """

    param_dict = dict(zip(params, values))

    model_data = data_obj.model(model_class(**param_dict))
    data, errors = data_obj.flatten_data()

    return -0.5 * np.sum((data - model_data) ** 2 / errors**2)


def loglike_nosignal(values, params, data_obj, model_class):
    """
    Abstract null log-likelihood function for a given model class and data object, assuming Gaussian errors.

    Parameters
    ----------
    values : array-like
        Values of the model parameters.
    params : list
        List of parameter names.
    data_obj : OIData
        Object containing the data to be fitted.
    model_class : class
        Model class to be fitted to the data.

    Returns
    -------
    float
        Log-likelihood value.
    """

    param_dict = dict(zip(params, values))

    model_data = data_obj.model(model_class(**param_dict))
    _, errors = data_obj.flatten_data()
    data = np.concatenate(
        [
            np.ones_like(data_obj.vis),
            np.zeros_like(data_obj.phi),
        ]
    )

    return -0.5 * np.sum((data - model_data) ** 2 / errors**2)


def laplace_cov(values, params, data_obj, model_class):
    """
    Compute the full Laplace covariance matrix for all model parameters jointly.

    Computes the inverse of the Hessian of the negative log-likelihood with
    respect to all parameters in ``params`` simultaneously, returning an
    ``N x N`` covariance matrix (where ``N = len(params)``).

    .. note::
        This function returns the *full* covariance matrix over all ``N``
        parameters.  To obtain only the marginal flux uncertainty at a fixed
        position, use :func:`laplace_contrast_uncertainty` instead.

    Parameters
    ----------
    values : array-like
        Values of the model parameters.
    params : list
        List of parameter names.
    data_obj : OIData
        Object containing the data to be fitted.
    model_class : class
        Model class to be fitted to the data.

    Returns
    -------
    array-like
        ``N x N`` covariance matrix, where ``N = len(params)``.
    """

    objective = lambda vals: -loglike(vals, params, data_obj, model_class)
    return _laplace_covariance(objective, np.asarray(values, dtype=float))


def laplace_contrast_uncertainty(
    flux, dra, ddec, data_obj, model_class, params=None
):
    """
    Compute the Laplace uncertainty in flux at a fixed sky position.

    Unlike :func:`laplace_cov`, which inverts the *full* N-parameter Hessian,
    this function **fixes** ``dra`` and ``ddec`` and computes only the scalar
    curvature of the negative log-likelihood along the **flux axis alone**:

    .. math::

        \\sigma_f = \\left(\\frac{\\partial^2 (-\\log L)}{\\partial f^2}\\right)^{-1/2}

    This is a 1-D (scalar) second derivative, not a matrix inversion.  It is
    appropriate when the position is held fixed (e.g. on a detection grid) and
    only the contrast uncertainty at that grid point is needed.  For the joint
    uncertainty over all parameters, use :func:`laplace_cov` instead.

    Parameters
    ----------
    flux : float
        Flux ratio value at which the local Laplace uncertainty is evaluated.
    dra : float
        Right ascension offset in mas (held fixed).
    ddec : float
        Declination offset in mas (held fixed).
    data_obj : OIData
        Object containing the data to be fitted.
    model_class : class
        Model class to be fitted to the data.
    params : list[str] or tuple[str, str, str], optional
        Parameter names corresponding to ``(dra, ddec, flux)``. Defaults to
        ``["dra", "ddec", "flux"]``.

    Returns
    -------
    float
        Scalar uncertainty in the contrast (standard deviation along flux axis).
    """

    if params is None:
        params = ["dra", "ddec", "flux"]

    objective = lambda f: (
        -loglike([dra, ddec, f], params, data_obj, model_class)
    )
    # Compute the scalar second derivative d²(-logL)/df² via double grad.
    # Using jax.grad twice makes it explicit that we expect a scalar result.
    # jax.hessian on a scalar-to-scalar function returns a 0-d array (not a
    # matrix), so calling hessian_matrix here would be misleading.
    d2_flux = jax.grad(jax.grad(objective))(np.asarray(flux, dtype=float))
    return np.sqrt(1.0 / np.asarray(d2_flux, dtype=float))


def fisher(values, params, data_obj, model_class, ridge=0.0):
    """Approximate the local Fisher matrix at a parameter point.

    Parameters
    ----------
    values : array-like
        Parameter vector at which to evaluate the local curvature.
    params : list[str]
        Parameter names corresponding to ``values``.
    data_obj : OIData
        Observational data object.
    model_class : class
        Model class used to evaluate the likelihood.
    ridge : float, optional
        Diagonal regularization term.

    Returns
    -------
    array-like
        Fisher information matrix.
    """
    objective = lambda vals: -loglike(vals, params, data_obj, model_class)
    return _fisher_matrix(
        objective, np.asarray(values, dtype=float), ridge=ridge
    )


def chi2ppf(p, df):
    """
    Percentile function for chi-square.

    For ``df=1`` (the path used in ``nsigma``), use the closed-form identity
    based on the standard normal quantile, i.e. square ``norm.ppf((p+1)/2)``.
    This remains JAX-native,
    differentiable, and fast.

    For ``df != 1``, this falls back to numpyro's gammaincinv backend when
    available.

    Parameters
    ----------
    p : array-like
        Percentile value
    df : array-like
        Degrees of freedom

    Returns
    -------
    array-like
        Corresponding chi2 value to the percentile
    """
    p = np.asarray(p, dtype=float)
    p = np.clip(p, np.finfo(float).eps, 1.0 - np.finfo(float).eps)

    try:
        if float(onp.asarray(df)) == 1.0:
            z = jax.scipy.stats.norm.ppf((p + 1.0) / 2.0)
            return z**2
    except Exception:
        pass

    from numpyro.distributions.util import gammaincinv

    return gammaincinv(df / 2.0, p) * 2.0


def nsigma(chi2r_test, chi2r_true, ndof):
    """
    Parameters
    ----------
    chi2r_test: float
        Reduced chi-squared of test model.
    chi2r_true: float
        Reduced chi-squared of true model.
    ndof: int
        Number of degrees of freedom.

    Returns
    -------
    nsigma: float
        Detection significance.
    """

    q = jax.scipy.stats.chi2.cdf(ndof * chi2r_test / chi2r_true, ndof)
    p = 1.0 - q
    nsigma = np.sqrt(chi2ppf(p, 1.0))

    return nsigma


def closure_phases(cvis, index_cps1, index_cps2, index_cps3):
    """
    Calculate closure phases from complex visibilities.

    Parameters
    ----------
    cvis : array-like
        Complex visibilities.
    index_cps1 : array-like
        First baseline indices for each closure triangle.
    index_cps2 : array-like
        Second baseline indices for each closure triangle.
    index_cps3 : array-like
        Third baseline indices for each closure triangle.

    Returns
    -------
    array-like
        Closure phases in degrees.

    """
    visphiall = np.rad2deg(np.angle(cvis))
    visphiall = np.mod(visphiall + 180.0, 360.0) - 180.0
    visphi = np.reshape(visphiall, (len(cvis), 1))
    cp = (
        visphi[np.array(index_cps1)]
        + visphi[np.array(index_cps2)]
        - visphi[np.array(index_cps3)]
    )
    out = np.reshape(np.mod(cp + 180.0, 360.0) - 180.0, len(index_cps1))
    return out


def cp_indices(vis_sta_index, cp_sta_index):
    """Map closure-triangle station indices to baseline indices.

    Parameters
    ----------
    vis_sta_index : array-like
        Baseline station index pairs from visibility data.
    cp_sta_index : array-like
        Triangle station index triplets from closure-phase data.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Arrays ``(i_cps1, i_cps2, i_cps3)`` identifying the three baselines
        composing each closure phase.
    """
    vis_sta_index, cp_sta_index = (
        onp.array(vis_sta_index, dtype=int),
        onp.array(cp_sta_index, dtype=int),
    )
    i_cps1 = onp.zeros(len(onp.array(cp_sta_index)), dtype=int)
    i_cps2 = onp.zeros(len(onp.array(cp_sta_index)), dtype=int)
    i_cps3 = onp.zeros(len(onp.array(cp_sta_index)), dtype=int)

    for i in range(len(cp_sta_index)):
        i_cps1[i] = onp.argwhere(
            (cp_sta_index[i][0] == vis_sta_index[:, 0])
            & (cp_sta_index[i][1] == vis_sta_index[:, 1])
        )[0, 0]
        i_cps2[i] = onp.argwhere(
            (cp_sta_index[i][1] == vis_sta_index[:, 0])
            & (cp_sta_index[i][2] == vis_sta_index[:, 1])
        )[0, 0]
        i_cps3[i] = onp.argwhere(
            (cp_sta_index[i][0] == vis_sta_index[:, 0])
            & (cp_sta_index[i][2] == vis_sta_index[:, 1])
        )[0, 0]
    return (
        onp.array(i_cps1, dtype=int),
        onp.array(i_cps2, dtype=int),
        onp.array(i_cps3, dtype=int),
    )
