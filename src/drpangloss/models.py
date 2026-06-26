from collections.abc import Sequence
from functools import partial

import equinox as eqx
import jax
from astropy.io import fits
import jax.numpy as jnp
import numpy as np
import os
import zodiax as zx
from jax.scipy.signal import fftconvolve

from ._utils import (
    DEG2RAD,
    I2PI,
    MAS2RAD,
    bessel_jn,
    img_get_sky_coordinates,
    undo_elliptical_transf_coord,
    undo_elliptical_transf_spat_freq,
)
from .inference import (
    fisher_matrix as _fisher_matrix,
    laplace_covariance as _laplace_covariance,
)


# TODO: make this work if one wavelength solution HDU has multiple associated data HDUs
# (epoch 5 is an example of this). Also make an option to discard flagged data, maybe
# using a boolean mask?
class OIData(zx.Base):
    """
    Store and transform optical-interferometry observables.

    Parameters
    ----------
    data : dict or str | PathLike[str] or Sequence[str | PathLike[str]]
        Either a dictionary with explicit interferometric arrays, or a path or Sequence
        of paths to OIFITS files.
    filter_flagged : bool
        Whether or not to filter flagged data from the reading process when reading
        from OIFITS files.

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

    def __init__(self, data, *, filter_flagged=True):
        """
        Initialize from an OIFITS object or explicit arrays.

        Parameters
        ----------
        data : dict or str | PathLike[str] or Sequence[str | PathLike[str]]
            A path or Sequence of paths to OIFITS data files, or a dictionary containing
            ``u``, ``v``, ``wavel``, ``vis``, ``d_vis``, ``phi``, ``d_phi``,
            optional closure-phase indices, and convention flags.
        filter_flagged : bool
            Whether or not to filter flagged data from the reading process when reading
            from OIFITS files.
        """

        if not isinstance(data, dict):
            # Start off with empty 1D JAX arrays to concatenate to. We assume 1 element
            # per probed spatial frequency point. OIFITS data arrays are shaped (Nb, Nw)
            # , with Nb the number of baselines and Nw the number of wavelengths -> we
            # row-flatten these to 1D (Nb * Nw,) arrays. I.e. the first Nw elements
            # correspond to the different wavelength channels of the first baseline.
            # This will require some tiling / repeating of the wavelength solution
            # and baseline coordinates as we read them in.
            (
                self.vis,
                self.d_vis,
                self.phi,
                self.d_phi,
                self.u,
                self.v,
                self.wavel,
                self.i_cps1,
                self.i_cps2,
                self.i_cps3,
            ) = (jnp.array([], dtype=float),) * 10

            # Select case for a single OIFITS file versus sequence of files. In the
            # latter case all data is concatenated to what is already contained in the
            # object.
            if isinstance(data, (str, os.PathLike)):
                with fits.open(data, mode="readonly") as hdul:
                    self._read_oifits_file(hdul)
            elif isinstance(data, Sequence):
                for oifits_file in [
                    d for d in data if isinstance(d, (str, os.PathLike))
                ]:
                    with fits.open(oifits_file, mode="readonly") as hdul:
                        self._read_oifits_file(hdul)
            else:
                ValueError(
                    "The passed object is neither a dictionary nor "
                    "a filepath or sequence of filepaths."
                )
        else:
            # assume data is a dict of the form {'u':u,'v':v,'wavel':wavel,'vis':vis,'d_vis':d_vis,
            #'phi':phi,'d_phi':d_phi,'i_cps1':i_cps1,'i_cps2':i_cps2,'i_cps3':i_cps3,'v2_flag':v2_flag,'cp_flag':cp_flag}

            self.u = jnp.array(data["u"], dtype=float)
            self.v = jnp.array(data["v"], dtype=float)
            self.wavel = jnp.array(data["wavel"], dtype=float)

            self.vis = jnp.array(data["vis"], dtype=float)
            self.d_vis = jnp.array(data["d_vis"], dtype=float)

            self.phi = jnp.array(data["phi"], dtype=float)
            self.d_phi = jnp.array(data["d_phi"], dtype=float)

            try:
                idx1 = data["i_cps1"]
                idx2 = data["i_cps2"]
                idx3 = data["i_cps3"]
                if idx1 is None or idx2 is None or idx3 is None:
                    raise KeyError
                self.i_cps1 = jnp.array(idx1, dtype=int)
                self.i_cps2 = jnp.array(idx2, dtype=int)
                self.i_cps3 = jnp.array(idx3, dtype=int)
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

    def _read_oifits_file(self, hdul):
        """Read in a sinlge OIFITS file ``astropy`` HUDList and merge with already
        stored data."""
        # loop over wavelength solution HDUs and find corresponding
        # visibilities/phases based on INSNAME header keyword.
        hdu_wave_list = [hdu for hdu in hdul if hdu.name == "OI_WAVELENGTH"]
        for hdu_wave in hdu_wave_list:
            # get wavelength solution as 1D array.
            wavel_sol = jnp.array(hdu_wave.data["EFF_WAVE"], dtype=float)
            insname = hdu_wave.header["INSNAME"]
            # get number of previously loaded visibility measurements
            nvis_old = self.vis.size

            # look up all corresponding HDUs matched based on INSNAME keyword
            hdu_vis2_list = [
                hdu
                for hdu in hdul
                if (hdu.name == "OI_VIS2" and hdu.header["INSNAME"] == insname)
            ]
            hdu_vis_list = [
                hdu
                for hdu in hdul
                if (hdu.name == "OI_VIS" and hdu.header["INSNAME"] == insname)
            ]
            hdu_t3_list = [
                hdu
                for hdu in hdul
                if (hdu.name == "OI_T3" and hdu.header["INSNAME"] == insname)
            ]

            # If visibility flag has not yet been set, set it now.
            if not hasattr(self, "v2_flag"):
                if len(hdu_vis2_list) != 0:
                    self.v2_flag = True
                elif len(hdu_vis_list) != 0:
                    self.v2_flag = False

            # if square visibilities are available, get them, otherwise get unsquared visibilities
            if self.v2_flag is True and len(hdu_vis2_list) != 0:
                # Add data to 1D array attributes.
                hdu_vis = hdu_vis2_list[0]
                vis_arr = jnp.array(hdu_vis.data["VIS2DATA"], dtype=float)
                d_vis_arr = jnp.array(hdu_vis.data["VIS2ERR"], dtype=float)
                self.vis = jnp.concatenate(
                    (
                        self.vis,
                        vis_arr.flatten(),
                    )
                )
                self.d_vis = jnp.concatenate(
                    (
                        self.d_vis,
                        d_vis_arr.flatten(),
                    )
                )
                self.wavel = jnp.concatenate(
                    (
                        self.wavel,
                        jnp.tile(wavel_sol, vis_arr.shape[0]),
                    )
                )
                u = jnp.array(hdu_vis.data["UCOORD"], dtype=float)
                v = jnp.array(hdu_vis.data["VCOORD"], dtype=float)
                self.u = jnp.concatenate(
                    (
                        self.u,
                        jnp.repeat(u, wavel_sol.size),
                    )
                )
                self.v = jnp.concatenate(
                    (
                        self.v,
                        jnp.repeat(v, wavel_sol.size),
                    )
                )

                # Stance indices for matching with closure phase data.
                vis_sta_index = hdu_vis.data["STA_INDEX"]
            elif self.v2_flag is False and len(hdu_vis_list) != 0:
                # Add data to 1D array attributes.
                hdu_vis = hdu_vis_list[0]
                vis_key = (
                    "VISAMP" if "VISAMP" in hdu_vis.data.names else "VISPHI"
                )
                d_vis_key = (
                    "VISAMPERR"
                    if "VISAMPERR" in hdu_vis.data.names
                    else "VISERR"
                )
                vis_arr = jnp.array(hdu_vis.data[vis_key], dtype=float)
                d_vis_arr = jnp.array(hdu_vis.data[d_vis_key], dtype=float)
                self.vis = jnp.concatenate((self.vis, vis_arr.flatten()))
                self.d_vis = jnp.concatenate(
                    (
                        self.d_vis,
                        d_vis_arr.flatten(),
                    )
                )
                self.wavel = jnp.concatenate(
                    (
                        self.wavel,
                        jnp.tile(wavel_sol, vis_arr.shape[0]),
                    )
                )
                u = jnp.array(hdu_vis.data["UCOORD"], dtype=float)
                v = jnp.array(hdu_vis.data["VCOORD"], dtype=float)
                self.u = jnp.concatenate(
                    (
                        self.u,
                        jnp.repeat(u, wavel_sol.size),
                    )
                )
                self.v = jnp.concatenate(
                    (
                        self.v,
                        jnp.repeat(v, wavel_sol.size),
                    )
                )

                # Stance indices for matching with closure phase data.
                vis_sta_index = hdu_vis.data["STA_INDEX"]
            else:
                raise ValueError(
                    "No corresponding OI_VIS2 or OI_VIS table found"
                    f"for OI_WAVELENGTH table with INSNAME: {insname}."
                )

            # If phase flag has not yet been set, set it now.
            if not hasattr(self, "cp_flag"):
                if (
                    len(hdu_vis_list) != 0
                    and hdu_vis_list[0].header.get("PHITYP") == "absolute"
                ):
                    self.cp_flag = False
                elif len(hdu_t3_list) != 0:
                    self.cp_flag = True

            # if absolute phases are available, get them, otherwise get closure phasess
            if self.cp_flag is False and (
                len(hdu_vis_list) != 0
                and hdu_vis_list[0].header.get("PHITYP") == "absolute"
            ):
                hdu_phi = hdu_vis_list[0]
                phi_arr = jnp.array(hdu_phi.data["VISPHI"], dtype=float)
                d_phi_arr = jnp.array(hdu_phi.data["VISPHIERR"], dtype=float)
                self.phi = jnp.concatenate((self.phi, phi_arr.flatten()))
                self.d_phi = jnp.concatenate(
                    (
                        self.d_phi,
                        d_phi_arr.flatten(),
                    )
                )
                self.i_cps1, self.i_cps2, self.i_cps3 = None, None, None
            elif self.cp_flag is True and len(hdu_t3_list) != 0:
                hdu_phi = hdu_t3_list[0]
                phi_arr = jnp.array(hdu_phi.data["T3PHI"], dtype=float)
                d_phi_arr = jnp.array(hdu_phi.data["T3PHIERR"], dtype=float)
                self.phi = jnp.concatenate((self.phi, phi_arr.flatten()))
                self.d_phi = jnp.concatenate(
                    (
                        self.d_phi,
                        d_phi_arr.flatten(),
                    )
                )
                cp_sta_index = jnp.array(hdu_phi.data["STA_INDEX"], dtype=int)

                # get indices of the baselines of the corresponding visibility
                # measurements (i.e. index of the corresponding value in self.vis).
                # We separately account for number of wavelength channels and
                # previously loaded measurements below.
                i_cps1, i_cps2, i_cps3 = cp_indices(
                    vis_sta_index, cp_sta_index
                )

                # visibility indices for 1st wavelength channels
                i_cps1 *= wavel_sol.size
                i_cps2 *= wavel_sol.size
                i_cps3 *= wavel_sol.size

                # squeeze in channel offsets with broadcasting and account for
                # previously loaded visibility measurments
                i_wave_offsets = jnp.arange(0, wavel_sol.size)[
                    None, :
                ]  # squeeze in channel offsets (shape (1, Nw))
                i_cps1 = (i_cps1[:, None] + i_wave_offsets).ravel() + nvis_old
                i_cps2 = (i_cps2[:, None] + i_wave_offsets).ravel() + nvis_old
                i_cps3 = (i_cps3[:, None] + i_wave_offsets).ravel() + nvis_old

                self.i_cps1 = jnp.concatenate((self.i_cps1, i_cps1))
                self.i_cps2 = jnp.concatenate((self.i_cps2, i_cps2))
                self.i_cps3 = jnp.concatenate((self.i_cps3, i_cps3))

                # make sure the indices are integer
                self.i_cps1 = jnp.astype(self.i_cps1, int)
                self.i_cps2 = jnp.astype(self.i_cps2, int)
                self.i_cps3 = jnp.astype(self.i_cps3, int)
            else:
                raise ValueError(
                    "No corresponding absolute phases in OI_VIS table or closure"
                    "phases in OI_T3 table found for OI_WAVELENGTH table with "
                    f"INSNAME: {insname}."
                )

    def flatten_data(self):
        """
        Flatten closure phases and uncertainties.
        """
        return jnp.concatenate([self.vis, self.phi]), jnp.concatenate(
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

        return jnp.concatenate([self.to_vis(cvis), self.to_phases(cvis)])

    def to_vis(self, cvis):
        """
        Convert complex visibilities to visibilities or squared visibilities.
        """
        if self.v2_flag:
            return jnp.abs(cvis) ** 2
        else:
            return jnp.abs(cvis)

    def to_phases(self, cvis):
        """
        Convert complex visibilities to closure phases or absolute phases.
        """
        if self.cp_flag:
            return closure_phases(cvis, self.i_cps1, self.i_cps2, self.i_cps3)
        else:
            return jnp.rad2deg(jnp.angle(cvis))

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

        self.sep = jnp.asarray(sep, dtype=float)
        self.pa = jnp.asarray(pa, dtype=float)
        self.contrast = jnp.asarray(contrast, dtype=float)

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

        self.dra = jnp.asarray(dra, dtype=float)
        self.ddec = jnp.asarray(ddec, dtype=float)
        self.flux = jnp.asarray(flux, dtype=float)

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


class BinaryGaussianRimModel(zx.Base):
    r"""
    Represents a chromatic 'disk' rim surrounding a binary star.
    The primary and secondary star are modelled as point sources. The primary
    has an additional spectral index slope, while the secondary is considered
    grey. The rim is modelled as a (potentially azimuthaly modulated) infinitely
    thin rim with its own spectral index slope, convolved with an isotropic 2D Gaussian.
    In addition, allows one to add a grey overresolved flux background.

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
    si_s : float or array-like
        Spectral index of the secondary.
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
    si_bkg: float or array-like
        Spectral index of the overresolved background.
    az_amps: array-like
        1D array containing amplitude coefficients for rim cosine azimuthal modulations.
        The first element is seen as the amplitude for the first-order modulation,
        the second as the amplitude for the second-order modulation, etc.
    az_pas: array-like
        1D array containing position angles of the rim's cosine azimuthal modulations,
        in degrees. The first element is seen as the angle for the first-order
        modulation, the second for the second-order modulation, etc.
    wave0: array-like
        Wavelength at which the total flux fractions are defined in meters. Note
        that this is not a free parameter, but is just fixed at initialization.

    Notes
    -----
    * The intensity profiles is separable into a symmetric radial profile and cosine
    azimuthal modulations, meaning the image intensity can be described in polar image
    coordinates as $I(r, \theta) = f(r) \left( 1 + \sum_{m=0}^{n}
    A_m \cos{(m(\theta - \pa_m)} \right)$, where $f(r)$ describes a Gaussian radial
    intensity profile.

    * Note that the spectral slope is defined in the wavelength-formulation of flux,
    i.e. $d$ in $F_{\lambda} \propto \lambda^{d}$.
    """

    flux_p: jax.Array
    ud_p: jax.Array
    dra_p: jax.Array
    ddec_p: jax.Array
    si_p: jax.Array
    flux_s: jax.Array
    dra_s: jax.Array
    ddec_s: jax.Array
    si_s: jax.Array
    diam_rim: jax.Array
    fwhm_rim: jax.Array
    inc_rim: jax.Array
    pa_rim: jax.Array
    si_rim: jax.Array
    flux_bkg: jax.Array
    si_bkg: jax.Array
    az_amps: jax.Array
    az_pas: jax.Array
    wave0: jax.Array

    def __init__(
        self,
        flux_p,
        ud_p,
        dra_p,
        ddec_p,
        si_p,
        flux_s,
        dra_s,
        ddec_s,
        si_s,
        diam_rim,
        fwhm_rim,
        inc_rim,
        pa_rim,
        si_rim,
        flux_bkg,
        si_bkg,
        az_amps,
        az_pas,
        wave0,
    ):
        """
        Initialize a cirumbinary Gaussian rim model.

        Parameters
        ----------
        flux_p : float or array-like
            Total flux fraction of the primary.
        ud_p : float or array-like
            Diameter of primary uniform disk in milliarceseconds.
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
        si_s : float or array-like
            Spectral index of the secondary.
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
        si_bkg: float or array-like
            Spectral index of the overresolved background.
        az_amps: array-like
            1D array containing amplitude coefficients for rim cosine azimuthal modulations.
            The first element is seen as the amplitude for the first-order modulation,
            the second as the amplitude for the second-order modulation, etc.
        az_pas: array-like
            1D array containing position angles of the rim's cosine azimuthal modulations,
            in degrees. The first element is seen as the angle for the first-order
            modulation, the second for the second-order modulation, etc.
        wave0: array-like
            Wavelength at which the total flux fractions are defined in meters. Note
            that this should not be considere a free parameter, but is just fixed at
            initialization.
        """
        self.flux_p = jnp.asarray(flux_p, dtype=float)
        self.ud_p = jnp.asarray(ud_p, dtype=float)
        self.dra_p = jnp.asarray(dra_p, dtype=float)
        self.ddec_p = jnp.asarray(ddec_p, dtype=float)
        self.si_p = jnp.asarray(si_p, dtype=float)
        self.flux_s = jnp.asarray(flux_s, dtype=float)
        self.dra_s = jnp.asarray(dra_s, dtype=float)
        self.ddec_s = jnp.asarray(ddec_s, dtype=float)
        self.si_s = jnp.asarray(si_s, dtype=float)
        self.diam_rim = jnp.asarray(diam_rim, dtype=float)
        self.fwhm_rim = jnp.asarray(fwhm_rim, dtype=float)
        self.inc_rim = jnp.asarray(inc_rim, dtype=float)
        self.pa_rim = jnp.asarray(pa_rim, dtype=float)
        self.si_rim = jnp.asarray(si_rim, dtype=float)
        self.flux_bkg = jnp.asarray(flux_bkg, dtype=float)
        self.si_bkg = jnp.asarray(si_bkg, dtype=float)
        self.az_amps = jnp.asarray(az_amps, dtype=float)
        self.az_pas = jnp.asarray(az_pas, dtype=float)
        self.wave0 = jnp.asarray(wave0, dtype=float)

    def __repr__(self):
        """Return a readable representation of the model parameters."""
        repr_str = (
            f"BinaryGaussianRimModel(flux_p={self.flux_p}, dra_p={self.dra_p}, "
            f"ud_p = {self.ud_p}, "
            f"ddec_p={self.ddec_p}, si_p={self.si_p}, flux_s={self.flux_s}, "
            f"dra_s={self.dra_s}, ddec_s={self.ddec_s}, si_s={self.si_s}, diam_rim={self.diam_rim}, "
            f"fwhm_rim={self.fwhm_rim}, inc_rim={self.inc_rim}, pa_rim={self.pa_rim}, "
            f"si_rim={self.si_rim}, flux_bkg={self.flux_bkg}, si_bkg={self.si_bkg}, az_amps={self.az_amps},"
            f"az_pas={self.az_pas}, wave0={self.wave0}"
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
              array_like, array-like, array-like, array-like]
            Tuple ``(flux_p, ud_p, dra_p, ddec_p, si_p, flux_s, dra_s, ddec_s, si_s,
                     diam_rim, fwhm_rim, inc_rim, pa_rim, si_rim, flux_bkg, si_bkg, az_amps,
                     az_pas)``.
        """
        return (
            self.flux_p,
            self.self.ud_p,
            self.dra_p,
            self.ddec_p,
            self.si_p,
            self.flux_s,
            self.dra_s,
            self.ddec_s,
            self.si_s,
            self.diam_rim,
            self.fwhm_rim,
            self.inc_rim,
            self.pa_rim,
            self.si_rim,
            self.flux_bkg,
            self.self.si_bkg,
            self.az_amps,
            self.az_pas,
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

        # Get azimuthal order offfset angles compared to rim position angle.
        az_phis = self.az_pas - self.pa_rim

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
            az_phis,
        )
        # Complex visibilities for binary components.
        cvis_p = cvis_uniform_disk(uu, vv, self.ud_p, self.dra_p, self.ddec_p)
        dra_s_rad, ddec_s_rad = self.dra_s * MAS2RAD, self.ddec_s * MAS2RAD
        cvis_s = jnp.exp(-I2PI * (uu * dra_s_rad + vv * ddec_s_rad))

        # Calculate spectra for each component.
        flux_rim = 1 - self.flux_p - self.flux_s - self.flux_bkg
        spec_rim = flux_rim * (wavel / self.wave0) ** self.si_rim
        spec_p = self.flux_p * (wavel / self.wave0) ** self.si_p
        spec_s = self.flux_s * (wavel / self.wave0) ** self.si_s
        spec_bkg = self.flux_bkg * (wavel / self.wave0) ** self.si_bkg

        # Combine into spectral-weighted total complex visibility.
        cvis_tot = (
            spec_rim * cvis_rim + spec_p * cvis_p + spec_s * cvis_s
        ) / (spec_rim + spec_p + spec_s + spec_bkg)

        return cvis_tot

    @partial(jax.jit, static_argnames="npix")
    def get_img(self, npix, ps):
        """Get an image of the model rim intensity. The returned image is normalized
        so the sum of the intensity value over all pixels is one. The rim center is
        placed in the middle of the image.

        Parameters
        ----------
        npix : int
            Number of image pixels along the field-of-view.
        ps : array-like or float
            Pixelscale of the image pixels in milliarcsecond.

        Returns
        -------
        array-like
           A 2D array containing the image intensities, normalized so the sum over
           pixels equals one.

        Notes
        -----
        * Note that the image is calculated so the $(x,y) = (0,0)$ point (i.e. the center
        of the field-of-view) lies at the geometric center of the image. For an odd
        amount of pixels, this corresponds to the center of the centermost pixel (as
        plotted using e.g. `plt.imshow()`). For an even amount of pixels, this
        corresponds to the vertex between the four centermost pixels.

        * Since this image is calculated numerically, it is a good idea to have
        a small pixelscale relative to the expected rim  (e.g. a factor of 10).
        The user should take into account the expected FWHM of the rim and the expected
        azimuthal modulations to avoid artefacts.
        """
        # Add radially symmetric component (order m=0) to beginning of the order arrays.
        az_amps = jnp.concatenate([jnp.array([1.0]), self.az_amps])
        az_phis = jnp.concatenate(
            [
                jnp.array([0.0]),
                self.az_pas - self.pa_rim,
            ]
        )
        az_orders = jnp.arange(az_amps.size)  # Array of the order indices.

        # Change of units.
        az_phis_rad = az_phis * DEG2RAD

        # Initialize empty image & get 1D coordinates of shape (npix,)
        img_zeros = jnp.zeros((npix, npix))
        xflat, yflat = img_get_sky_coordinates(img_zeros, ps=ps)

        # Get 2D meshgrid of coordinates & transform to frame where elliptical ring is
        # derotated and unstretched (i.e. elliptical coordinates).
        xmesh, ymesh = jnp.meshgrid(xflat, yflat)
        rmesh = jnp.hypot(xmesh, ymesh)

        stretch = jnp.maximum(
            jnp.cos(self.inc_rim * DEG2RAD), 1e-8
        )  # Doesn't explode at i=90 deg.
        xmesh_ell, ymesh_ell = undo_elliptical_transf_coord(
            xmesh,
            ymesh,
            pa=self.pa_rim,
            stretch=stretch,
        )
        rmesh_ell = jnp.hypot(xmesh_ell, ymesh_ell)  # Elliptical coord radius.
        thetamesh_ell = jnp.arctan2(
            xmesh_ell, ymesh_ell
        )  # Elliptical coord position angle.

        # Create a boolen mask to select pixels closest to the actual thin ellipse rim
        # (i.e. to within one pixelscale of the thin rim).
        mask = jnp.abs(rmesh_ell - self.diam_rim / 2) <= (ps)
        img = jnp.where(mask, 1.0, img_zeros)

        # Apply azimuthal modulation terms.
        f = lambda az_amp, az_phi_rad, az_order: (
            az_amp * jnp.cos(az_order * (thetamesh_ell - az_phi_rad))
        )

        # Apply azimuthal modulations.
        az_factors = jax.vmap(f)(az_amps, az_phis_rad, az_orders)
        img = img * jnp.sum(az_factors, axis=0)

        # Apply isotropic Gaussian convolution
        sigma = self.fwhm_rim / (2 * jnp.sqrt(2 * jnp.log(2)))
        img_gauss = (
            1 / (2 * jnp.pi * sigma**2) * jnp.exp(-(rmesh**2) / (2 * sigma**2))
        )
        img = fftconvolve(img, img_gauss, mode="same")

        return img / jnp.sum(img)


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

    th = pa * DEG2RAD

    ddec = MAS2RAD * (sep * jnp.cos(th))
    dra = -1 * MAS2RAD * (sep * jnp.sin(th))

    # decompose into two "luminosity"
    l2 = 1.0 / (contrast + 1)
    l1 = 1 - l2

    # phase-factor
    phi = jnp.exp(-I2PI * (u * dra + v * ddec))
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
    ddec = ddec * jnp.pi / (180.0 * 3600.0 * 1000.0)
    dra = dra * jnp.pi / (180.0 * 3600.0 * 1000.0)
    phi_r = jnp.cos(-2 * jnp.pi * (u * dra + v * ddec))
    phi_i = jnp.sin(-2 * jnp.pi * (u * dra + v * ddec))

    cvis = p3 + p2 * phi_r + p2 * phi_i * 1.0j

    return cvis


def cvis_gaussian(u, v, fwhm, dra, ddec):
    """Compute the complex visibility of a centered isotropic 2D Gaussian.

    Parameters
    ----------
    u : array-like
        Baseline ``u`` coordinates in wavelength units (cycles / rad).
    v : array-like
        Baseline ``v`` coordinates in wavelength units (cycles / rad).
    fwhm : float or array-like
        Full-width-half-maximum of the Gaussian in milliarcseconds.
    dra : float or array-like
        Right-ascension offset of the rim in milliarcseconds.
    ddec : float or array-like
        Declination offset of the rim in milliarcseconds.

    Returns
    -------
    array-like
        Complex visibility samples.

    Notes
    -----
    This function does not account for rotation and geometric stretching (e.g. due to
    inclination). A separate transformation of $uv$ coordinates should account for this.
    """
    # Change of units.
    fwhm_rad, dra_rad, ddec_rad = fwhm * MAS2RAD, dra * MAS2RAD, ddec * MAS2RAD

    # Baseline norm.
    base_norm = jnp.hypot(u, v)  # In wavelength units (cycles/rad).

    # Calculate complex visibility (forced to be complex).
    cvis = (
        jnp.exp(-(jnp.pi**2) * fwhm_rad**2 * base_norm**2 / (4 * jnp.log(2)))
        + 0j
    )

    # Apply offset phase-factor.
    phi = jnp.exp(-I2PI * (u * dra_rad + v * ddec_rad))
    cvis *= phi

    return cvis


def cvis_uniform_disk(u, v, ud, dra, ddec):
    """Compute complex visibilities for uniform disk model.

    Parameters
    ----------
    u : array-like
        Baseline ``u`` coordinates in wavelength units (cycles / rad).
    v : array-like
        Baseline ``v`` coordinates in wavelength units (cycles / rad).
    ud : float or array-like
        Diameter of the uniform disk in milliarcseconds.
    dra : float or array-like
        Right-ascension offset of the rim in milliarcseconds.
    ddec : float or array-like
        Declination offset of the rim in milliarcseconds.

    Returns
    -------
    array-like
        Complex visibility samples.

    Notes
    -----
    This function does not account for rotation and geometric stretching (e.g. due to
    inclination). A separate transformation of $uv$ coordinates should account for this.
    """
    # Change of units.
    ud_rad, dra_rad, ddec_rad = ud * MAS2RAD, dra * MAS2RAD, ddec * MAS2RAD

    # Baseline norm.
    base_norm = jnp.hypot(u, v)  # In wavelength units (cycles/rad).

    # Kernel for Bessel function.
    kernel = jnp.pi * base_norm * ud_rad

    # Calculate complex visibility (forced to be complex). Where statement for
    # avoiding divergence around x=0.
    cvis = jnp.where(
        kernel == 0, 1.0 + 0j, (2 * bessel_jn(1, kernel)[1]) / kernel + 0j
    )

    # Apply offset phase-factor.
    phi = jnp.exp(-I2PI * (u * dra_rad + v * ddec_rad))
    cvis *= phi

    return cvis


def cvis_gaussian_rim(u, v, dra, ddec, diam, fwhm, inc, pa, az_amps, az_phis):
    """Compute complex visibilities for a (modulated) rim, consisting of a radial Dirac
    delta ring (infinitely thin) subsequently convolved with an isotropic 2D Gaussian.

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
    inc_rad, dra_rad, ddec_rad = (
        inc * DEG2RAD,
        dra * MAS2RAD,
        ddec * MAS2RAD,
    )

    # Transform spatial frequency coordinates to frame of reference where the model rim
    # is uninclined and the major axis is pointed North (postitive y-axis).
    stretch_factor = jnp.maximum(
        jnp.cos(inc_rad), 1e-8
    )  # Doesn't explode at i=90 deg.
    ut, vt = undo_elliptical_transf_spat_freq(u, v, pa, stretch_factor)

    # Compute complex visibilities for Dirac delta (infinitely thin) modulated ring.
    cvis = cvis_radial_dirac_delta_modulated(
        ut, vt, diam / 2.0, az_amps, az_phis
    )

    # Add effect of convolution in image-plane with an isotropic Gaussian of the given
    # FWHM in the original image frame of reference.
    cvis *= cvis_gaussian(u, v, fwhm, 0.0, 0.0)
    # Apply offset phase-factor.

    phi = jnp.exp(-I2PI * (u * dra_rad + v * ddec_rad))
    cvis *= phi

    return cvis


def cvis_radial_dirac_delta_modulated(u, v, r0, az_amps, az_phis):
    r"""Compute the complex visibility for an azimuthally modulated radial dirac
    delta ring. The image intensity can be described in polar image coordinates as
    $I(r, \theta) \propto \delta(r-r_0) \left( 1 + \sum_{m=0}^{n} A_m
    \cos{(m(\theta - \phi_m)} \right)$, where $r_0$ defines the ring's radial position,
    $A_m$ the amplitude and $\phi_m$ the position phase angle (defined counter-clockwise
    , North to East) for the m-th order modulation.

    This is a special analytical case of the [cvis_radial_profile_modulated][] function,
    which uses quadrature to calculate the complex visibility for any provided radial
    profile. Analytical visibility calculation using this function is much faster if the
    image can be described as a convolution of a infinitesimally thin ring with another
    component whose 2D Fourier transform is known analytically (e.g. a 2D Gaussian).

    Parameters
    ----------
    u : array-like
        Baseline ``u`` coordinates in wavelength units (cycles / rad).
    v : array-like
        Baseline ``v`` coordinates in wavelength units (cycles / rad).
    r0 : float or array-like
        Scalar with the radial position of the ring in milliarcseconds.
    az_amps : array-like
        1D array containing amplitude coefficients for cosine azimuthal modulations.
        The first element is seen as the amplitude for the first-order modulation,
        the second as the amplitude for the second-order modulation, etc.
    az_phis : array-like
        1D array containing offset angles of the cosine azimuthal modulations,
        relative to the position angle of the rim's projected major axis, in
        degrees. The first element is seen as the offset for the first-order
        modulation, the second for the second-order modulation, etc.

    Returns
    -------
    array-like
        Complex visibility samples.

    Notes
    -----
    This function does not account for rotation and geometric stretching (e.g. due to
    inclination). A separate transformation of $uv$ coordinates should account for this.
    The phase angles of the cosine modulations are defined relative to the
    spatial y-axis (North), turning counterclockwise to the x-axis (East). This means
    that a single 1st order modulation with a phase angle of $0 \, \mathrm{deg}$
    results in a bright peak towards the North, and a faint peak towards the South.
    A phase angle of $90 \, \mathrm{deg}$ would result in a bright peak towards the
    East, and a faint one towards the West.
    """
    # Add radially symmetric component (order m=0) to beginning of the order arrays.
    az_amps = jnp.concatenate([jnp.array([1.0]), az_amps])
    az_phis = jnp.concatenate([jnp.array([0.0]), az_phis])
    az_orders = jnp.arange(az_amps.size)  # Array of the order indices.

    # Change of units.
    r0_rad = r0 * MAS2RAD
    az_phis_rad = az_phis * DEG2RAD

    # Get length of baseline and baseline projection angle (i.e. counterclockwise angle in
    # uv-plane, turning from top, i.e. positive v, to left, i.e. positive u).
    base_norm = jnp.hypot(u, v)  # In wavelength units (cycles/rad).
    base_proj_ang_rad = jnp.arctan2(u, v)  # Baseline projection angle in rad.

    # Maximum order to consider.
    az_order_max = jnp.size(az_orders) - 1
    # Array to evaluate Bessel functions at (shape = (Nb,)).
    xbes = 2.0 * jnp.pi * base_norm * r0_rad
    # Evaluate Bessel function up to max order (shape = (Naz, Nb)).
    bessel_vals = bessel_jn(az_order_max, xbes)

    # Get the complex visibility term associated with a single azimuthal term.
    def _get_azmod_cvis_term(az_amp, az_phi_rad, az_order):
        azmod_cvis_term = (
            az_amp
            * jnp.exp(-0.5j * jnp.pi * az_order)
            * jnp.cos(az_order * (base_proj_ang_rad - az_phi_rad))
            * bessel_vals[az_order, :]
        )
        return azmod_cvis_term

    # vmap over the different azimuthal order evaluations.
    azmod_cvis_terms = jax.vmap(
        _get_azmod_cvis_term, in_axes=(0, 0, 0), out_axes=0
    )(az_amps, az_phis_rad, az_orders)
    # Sum over all orders.
    cvis = jnp.sum(azmod_cvis_terms, axis=0)

    return cvis


# TODO: make this work with modified Hankel profile transform (where multiple
# orders are returned at the same time).
def cvis_radial_profile_modulated(u, v, rpos, intensity, az_amps, az_phis):
    r"""Compute the complex visibility for a object whose intensity profiles is
    separable into a symmetric radial profile and cosine azimuthal modulations,
    meaning the image intensity can be described in polar image coordinates as
    $I(r, \theta) = f(r) \left( 1 + \sum_{m=0}^{n} A_m \cos{(m(\theta - \phi_m)}
    \right)$, where $f(r)$ describes the base radial intensity profile, $A_m$ the
    amplitude and $\phi_m$ the position phase angle (defined counter-clockwise,
    North to East) for the m-th order modulation.

    Parameters
    ----------
    u : array-like
        Baseline ``u`` coordinates in wavelength units (cycles / rad).
    v : array-like
        Baseline ``v`` coordinates in wavelength units (cycles / rad).
    rpos : array-like
        Radial coordinate positions of the radial profile in milliarcseconds.
    intensity : array-like
        Radial intensity profile defined at the ``rpos`` positions.
    az_amps : array-like
        1D array containing amplitude coefficients for cosine azimuthal modulations.
        The first element is seen as the amplitude for the first-order modulation,
        the second as the amplitude for the second-order modulation, etc.
    az_phis : array-like
        1D array containing offset angles of the cosine azimuthal modulations,
        relative to the position angle of the rim's projected major axis, in
        degrees. The first element is seen as the offset for the first-order
        modulation, the second for the second-order modulation, etc.

    Returns
    -------
    array-like
        Complex visibility samples.

    Notes
    -----
    The radial profile is used in trapezoidal quadrature to calculate the Hankel
    transforms. It's best to make sure that ``rpos`` resolves the radial intensity
    profile and your longest expected baseline fairly well.

    This function does not account for rotation and geometric stretching (e.g. due to
    inclination). A separate transformation of $uv$ coordinates should account for this.
    The phase angles of the cosine modulations are defined relative to the
    spatial y-axis (North), turning counterclockwise to the x-axis (East). This means
    that a single 1st order modulation with a phase angle of $0 \, \mathrm{deg}$
    results in a bright peak towards the North, and a faint peak towards the South.
    A phase angle of $90 \, \mathrm{deg}$ would result in a bright peak towards the
    East, and a faint one towards the West.
    """
    # Add radially symmetric component to beginning of the azimuthal order arrays.
    az_amps = jnp.concatenate([jnp.array([1.0]), az_amps])
    az_phis = jnp.concatenate([jnp.array([0.0]), az_phis])
    az_orders = jnp.arange(az_amps.size)  # Array of the order indices.

    # Change of units.
    az_phis_rad = az_phis * DEG2RAD

    # Get length of baseline and baseline projection angle (i.e. counterclockwise angle in
    # uv-plane, turning from top, i.e. positive v, to left, i.e. positive u).
    base_norm = jnp.hypot(u, v)  # In wavelength units (cycles/rad).
    base_proj_ang_rad = jnp.arctan2(u, v)  # Baseline projection angle in rad.

    # Get the complex visibility term associated with a single azimuthal term.
    def _get_azmod_cvis_term(az_amp, az_phi_rad, az_order):
        azmod_cvis_term = (
            az_amp
            * jnp.exp(-0.5j * jnp.pi * az_order)
            * jnp.cos(az_order * (base_proj_ang_rad - az_phi_rad))
            * hankel_n(az_order, base_norm, rpos, intensity)
        )
        return azmod_cvis_term

    # vmap over the different azimuthal order evaluations.
    azmod_cvis_terms = jax.vmap(
        _get_azmod_cvis_term, in_axes=(0, 0, 0), out_axes=0
    )(az_amps, az_phis_rad, az_orders)
    # Sum over all orders.
    cvis = jnp.sum(azmod_cvis_terms, axis=0)

    return cvis


# TODO: bessel_jn(n, x) actually returns results for all orders up to the requested one
# in array of shape (n + 1, shape(x)) due to recursion relation -> make this return
# the Hankel tranforms up to the nth order instead! They can be calculated in one go!
@partial(jax.jit, static_argnames=["n"])
def hankel_n(n, base_norm, rpos, intensity):
    r"""Function to compute the $n$-th order Hankel transform of given radial intensity
    profile. For $n=0$, this returns the normalized complex visibility of a
    centro-symmetric object with the given radial intensity profile. The Hankel
    transform is defined as $H_n = \int f(r) J_n(2\pi \rho r) r \, dr / \int f(r)r \,
    dr$, with $J_n$ the $n$-th Bessel function of the first kind and $\rho$ the length
    of the baseline in wavelength units.

    Parameters
    ----------
    n : int
        The order of the Hankel transform to calculate.
    base_norm: array-like
        Length of the baselines in wavelength units (cycles / rad). Passed as a 1D
        array (different baseline lengths are broadcasted over in the computation).
    rpos : array-like
        Radial coordinate positions of the radial profile in milliarcseconds. Passed
        as a 1D array.
    intensity: array-like
        Radial intensity profile defined at the ``rpos`` positions. Passed as a 1D
        array.

    Returns
    -------
    array-like
        $n$-th order normalized Hankel transform evaluated at the specified spatial
        frequencies ``u`` and ``v``. Returned as a 1D array for the given spatial
        frequencies.
    """
    rpos_rad = rpos * MAS2RAD  # Put radial positions in radian.

    # Calculate the scalar Hankel transform normalization factor (only needs to be
    # computed once).
    hankel_fnorm = jnp.trapezoid(intensity * rpos_rad, rpos_rad)

    # Broadcast multiply into a 2D kernel of shape (Nb, Nr) containing all possible
    # multiplied versions of baseline and radial intensity position.
    x = 2.0 * jnp.pi * base_norm[:, None] * rpos_rad[None, :]

    # Calculate bessel function for each element in x array.
    kernel = bessel_jn(n, x)

    # Set up array of integrands where we have to radially integrate over the second
    # axis with broadcasting (i.e. shape (Nb, Nr)).
    integrand_arr = intensity[None, :] * kernel * rpos_rad[None, :]

    # Integrate each row across radial axis, collecting the result for each into
    # vector of shape (Nb,), giving the normalized n-th Hankel transform for each
    # baseline.
    hankel_n = jnp.trapezoid(integrand_arr, rpos_rad, axis=1) / hankel_fnorm

    return hankel_n


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

    return -0.5 * jnp.sum((data - model_data) ** 2 / errors**2)


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
    data = jnp.concatenate(
        [
            jnp.ones_like(data_obj.vis),
            jnp.zeros_like(data_obj.phi),
        ]
    )

    return -0.5 * jnp.sum((data - model_data) ** 2 / errors**2)


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
    return _laplace_covariance(objective, jnp.asarray(values, dtype=float))


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
    d2_flux = jax.grad(jax.grad(objective))(jnp.asarray(flux, dtype=float))
    return jnp.sqrt(1.0 / jnp.asarray(d2_flux, dtype=float))


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
        objective, jnp.asarray(values, dtype=float), ridge=ridge
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
    p = jnp.asarray(p, dtype=float)
    p = jnp.clip(p, jnp.finfo(float).eps, 1.0 - jnp.finfo(float).eps)

    try:
        if float(np.asarray(df)) == 1.0:
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
    nsigma = jnp.sqrt(chi2ppf(p, 1.0))

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
    visphiall = jnp.rad2deg(jnp.angle(cvis))
    visphiall = jnp.mod(visphiall + 180.0, 360.0) - 180.0
    visphi = jnp.reshape(visphiall, (len(cvis), 1))
    cp = (
        visphi[jnp.array(index_cps1)]
        + visphi[jnp.array(index_cps2)]
        - visphi[jnp.array(index_cps3)]
    )
    out = jnp.reshape(jnp.mod(cp + 180.0, 360.0) - 180.0, len(index_cps1))
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
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        Arrays ``(i_cps1, i_cps2, i_cps3)`` identifying the three baselines
        composing each closure phase.
    """
    vis_sta_index, cp_sta_index = (
        np.array(vis_sta_index, dtype=int),
        np.array(cp_sta_index, dtype=int),
    )
    i_cps1 = np.zeros(len(np.array(cp_sta_index)), dtype=int)
    i_cps2 = np.zeros(len(np.array(cp_sta_index)), dtype=int)
    i_cps3 = np.zeros(len(np.array(cp_sta_index)), dtype=int)

    for i in range(len(cp_sta_index)):
        i_cps1[i] = np.argwhere(
            (cp_sta_index[i][0] == vis_sta_index[:, 0])
            & (cp_sta_index[i][1] == vis_sta_index[:, 1])
        )[0, 0]
        i_cps2[i] = np.argwhere(
            (cp_sta_index[i][1] == vis_sta_index[:, 0])
            & (cp_sta_index[i][2] == vis_sta_index[:, 1])
        )[0, 0]
        i_cps3[i] = np.argwhere(
            (cp_sta_index[i][0] == vis_sta_index[:, 0])
            & (cp_sta_index[i][2] == vis_sta_index[:, 1])
        )[0, 0]
    # Return indices as JAX arrays.
    return (
        jnp.array(i_cps1, dtype=int),
        jnp.array(i_cps2, dtype=int),
        jnp.array(i_cps3, dtype=int),
    )
