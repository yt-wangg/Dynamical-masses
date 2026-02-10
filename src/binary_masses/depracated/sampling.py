"""
Binary star sampling utilities

This module provides tools for generating synthetic binary star populations
and computing orbital properties.
"""

import numpy as np
import astropy.units as u
import copy
from numpy import cos, sin
from astropy import constants as const


def random_sample_from_power_law(x0, x1, gamma, n_random):
    """
    Sample from a power law distribution.

    Parameters
    ----------
    x0, x1 : float
        Lower and upper bounds
    gamma : float
        Power law index
    n_random : int
        Number of samples

    Returns
    -------
    array : samples from the power law distribution
    """
    return (np.random.uniform(size=n_random) * (x1**(gamma+1) - x0**(gamma+1)) + x0**(gamma+1))**(1./(gamma+1.))


def random_sample_from_linear_distribution(k, n):
    """
    Sample from a linear distribution p(x) ∝ (1 + k*x).

    Parameters
    ----------
    k : float
        Slope parameter
    n : int
        Number of samples

    Returns
    -------
    array : samples from the linear distribution
    """
    U = np.random.uniform(size=n)

    if abs(k) < 1e-5:
        return U
    elif k < -2:
        return (-np.sqrt(-2.*k) + np.sqrt(-2.*k + 2*k*U)) / k
    elif k > 2.:
        return 1. - (-np.sqrt(2.*k) + np.sqrt(2.*k - 2*k*U)) / (-k)
    else:
        return (-(1 - 0.5 * k) + np.sqrt((1 - 0.5 * k)**2 + 2 * k * U)) / k


def sample_true_anomaly(eccentricity, n_samples, precision=0.5 * np.pi / 180., equal_spaced_mean_anomaly=False):
    """
    Sample true anomaly from orbital eccentricity.

    Parameters
    ----------
    eccentricity : float or array
        Orbital eccentricity
    n_samples : int
        Number of samples
    precision : float
        Numerical precision for solving Kepler's equation
    equal_spaced_mean_anomaly : bool
        If True, use equally spaced mean anomaly

    Returns
    -------
    array : true anomaly values in radians
    """
    if equal_spaced_mean_anomaly:
        mean_anomaly = 2 * np.pi * np.linspace(0., 1., n_samples, endpoint=False)
    else:
        mean_anomaly = 2 * np.pi * np.random.uniform(size=n_samples)

    # Initialize eccentric anomaly with mean anomaly
    ecc_anomaly = copy.copy(mean_anomaly)

    # Solve Kepler's equation iteratively
    for i in range(1, n_samples):
        if isinstance(eccentricity, (int, float, np.floating)):
            e = eccentricity
        else:
            e = eccentricity[i]

        correction = (mean_anomaly[i] - ecc_anomaly[i] + e * sin(ecc_anomaly[i])) / (1. - e * cos(ecc_anomaly[i]))

        while abs(correction) > precision:
            ecc_anomaly[i] = ecc_anomaly[i] + correction
            correction = (mean_anomaly[i] - ecc_anomaly[i] + e * sin(ecc_anomaly[i])) / (1. - e * cos(ecc_anomaly[i]))

    # Convert to true anomaly
    cos_f = (cos(ecc_anomaly) - e) / (1 - e * cos(ecc_anomaly))
    sin_f = (np.sqrt(1-e**2) * sin(ecc_anomaly)) / (1 - e * cos(ecc_anomaly))
    true_anomaly = np.mod(np.arctan2(sin_f, cos_f), 2*np.pi)

    return true_anomaly


def period_from_semi_major_axis(semi_major_axis=1*u.au, total_mass=2*u.Msun):
    """
    Calculate orbital period from semi-major axis using Kepler's third law.

    Parameters
    ----------
    semi_major_axis : astropy.Quantity
        Semi-major axis
    total_mass : astropy.Quantity
        Total mass of the binary system

    Returns
    -------
    astropy.Quantity : orbital period in years
    """
    return (2. * np.pi * np.sqrt(semi_major_axis**3 / const.G / total_mass)).to(u.year)


def semi_major_axis_from_period(period=1.*u.year, total_mass=2*u.Msun):
    """
    Calculate semi-major axis from orbital period using Kepler's third law.

    Parameters
    ----------
    period : astropy.Quantity
        Orbital period
    total_mass : astropy.Quantity
        Total mass of the binary system

    Returns
    -------
    astropy.Quantity : semi-major axis in AU
    """
    return ((period/2./np.pi)**(2./3) * (const.G * total_mass)**(1./3)).to(u.au)


class _TwoBodySystem:
    """Base class for two-body orbital calculations."""

    def compute_orbital_properties(self):
        """Compute orbital properties for the system."""
        self.period = (2. * np.pi * np.sqrt(self.semi_major_axis**3 / (const.G * self.total_mass))).to(u.yr)
        self.semi_latus_rectum = self.semi_major_axis * (1. - self.eccentricity**2)
        self.radius_at_anomaly = self.semi_latus_rectum / (1. + self.eccentricity * cos(self.true_anomaly))
        self.velocity_at_anomaly = (np.sqrt(const.G * self.total_mass / self.semi_latus_rectum)).to(u.km/u.s)

        # Position calculations
        self.x_position = self.radius_at_anomaly * (
            cos(self.longitude_ascending_node) * cos(self.argument_periapsis + self.true_anomaly) -
            cos(self.inclination) * sin(self.longitude_ascending_node) * sin(self.argument_periapsis + self.true_anomaly)
        )
        self.y_position = self.radius_at_anomaly * (
            sin(self.longitude_ascending_node) * cos(self.argument_periapsis + self.true_anomaly) +
            cos(self.inclination) * cos(self.longitude_ascending_node) * sin(self.argument_periapsis + self.true_anomaly)
        )
        self.z_position = self.radius_at_anomaly * sin(self.inclination) * sin(self.argument_periapsis + self.true_anomaly)

        # Velocity calculations
        self.x_velocity = -self.velocity_at_anomaly * (
            cos(self.longitude_ascending_node) * (sin(self.argument_periapsis+self.true_anomaly) + self.eccentricity * sin(self.argument_periapsis)) +
            cos(self.inclination) * sin(self.longitude_ascending_node) * (cos(self.argument_periapsis + self.true_anomaly) + self.eccentricity * cos(self.argument_periapsis))
        )

        self.y_velocity = -self.velocity_at_anomaly * (
            sin(self.longitude_ascending_node) * (sin(self.argument_periapsis+self.true_anomaly) + self.eccentricity * sin(self.argument_periapsis)) -
            cos(self.inclination) * cos(self.longitude_ascending_node) * (cos(self.argument_periapsis + self.true_anomaly) + self.eccentricity * cos(self.argument_periapsis))
        )

        self.z_velocity = self.velocity_at_anomaly * sin(self.inclination) * (cos(self.argument_periapsis+self.true_anomaly) + self.eccentricity*cos(self.argument_periapsis))

        # Projected properties
        self.cos_phase_angle = (self.x_position*self.x_velocity + self.y_position*self.y_velocity) / np.sqrt(self.x_position**2 + self.y_position**2) / np.sqrt(self.x_velocity**2 + self.y_velocity**2)
        self.radial_velocity_angle = (np.arccos(self.cos_phase_angle) * 180/np.pi) * u.deg

        self.projected_separation = np.sqrt(self.x_position**2 + self.y_position**2)
        self.angular_separation = (self.projected_separation / self.distance).to(u.arcsec, equivalencies=u.dimensionless_angles())

        self.projected_velocity = np.sqrt(self.x_velocity**2 + self.y_velocity**2)
        self.proper_motion_difference = (self.projected_velocity / self.distance).to(u.mas/u.yr, equivalencies=u.dimensionless_angles())


class BinarySystem(_TwoBodySystem):
    """
    A single binary star system with specified orbital parameters.

    This class represents a binary system with given masses, orbital parameters,
    and distance, and computes the observable properties.
    """

    def __init__(self,
                 primary_mass=1.*u.Msun,
                 secondary_mass=1.*u.Msun,
                 semi_major_axis=1.*u.AU,
                 eccentricity=0.,
                 distance=1.*u.kpc,
                 n_phases=100,
                 precision=0.5 * np.pi / 180.,
                 face_on=True):
        """
        Initialize a binary system.

        Parameters
        ----------
        primary_mass, secondary_mass : astropy.Quantity
            Masses of the primary and secondary stars
        semi_major_axis : astropy.Quantity
            Orbital semi-major axis
        eccentricity : float
            Orbital eccentricity (0-1)
        distance : astropy.Quantity
            Distance to the system
        n_phases : int
            Number of orbital phases to sample
        precision : float
            Numerical precision for orbital calculations
        face_on : bool
            If True, assume face-on orientation
        """
        self.primary_mass = primary_mass
        self.secondary_mass = secondary_mass
        self.total_mass = self.primary_mass + self.secondary_mass
        self.semi_major_axis = semi_major_axis
        self.eccentricity = eccentricity
        self.distance = distance
        self.n_phases = n_phases

        self.true_anomaly = sample_true_anomaly(
            self.eccentricity, n_phases, precision, equal_spaced_mean_anomaly=True
        )

        if face_on:
            self.longitude_ascending_node = np.zeros(n_phases)
            self.argument_periapsis = np.zeros(n_phases)
            self.inclination = np.zeros(n_phases)
        else:
            self.longitude_ascending_node = 2. * np.pi * np.random.uniform(size=n_phases)
            self.argument_periapsis = 2. * np.pi * np.random.uniform(size=n_phases)
            V = np.random.uniform(size=n_phases)
            self.inclination = np.arccos(2 * V - 1)

        self.compute_orbital_properties()


class BinaryPopulation(_TwoBodySystem):
    """
    A population of binary star systems.

    This class generates and computes properties for a population of binary systems
    with specified parameter distributions.
    """

    def __init__(self, n_binaries,
                 semi_major_axis=None, semi_major_axis_params=None,
                 eccentricity=None, eccentricity_alpha=None, eccentricity_k=None,
                 total_mass=None,
                 distance=None,
                 precision=0.5 * np.pi / 180.,
                 face_on=False):
        """
        Initialize a binary population.

        Parameters
        ----------
        n_binaries : int
            Number of binary systems
        semi_major_axis : array or Quantity
            Fixed semi-major axis for all systems
        semi_major_axis_params : tuple
            Parameters for power-law distribution: (a0, a1, gamma)
        eccentricity : array or float
            Fixed eccentricity for all systems
        eccentricity_alpha : float or array
            Parameter for eccentricity power-law distribution
        eccentricity_k : float or array
            Parameter for eccentricity linear distribution
        total_mass : array or Quantity
            Total mass for the systems
        distance : array or Quantity
            Distance to the systems
        precision : float
            Numerical precision for orbital calculations
        face_on : bool
            If True, assume face-on orientation for all systems
        """
        self.n_binaries = n_binaries

        # Generate semi-major axis distribution
        if semi_major_axis_params is not None:
            if len(semi_major_axis_params) != 3:
                raise ValueError('len(semi_major_axis_params) != 3')
            a0, a1, gamma = semi_major_axis_params
            self.semi_major_axis = random_sample_from_power_law(a0, a1, gamma, n_binaries) * u.au
        elif semi_major_axis is None:
            self.semi_major_axis = np.ones(n_binaries) * u.au
        elif isinstance(semi_major_axis, u.quantity.Quantity):
            self.semi_major_axis = np.ones(n_binaries) * semi_major_axis
        elif isinstance(semi_major_axis, (list, np.ndarray)):
            if len(semi_major_axis) == n_binaries:
                self.semi_major_axis = semi_major_axis * u.au
            else:
                raise ValueError('len(semi_major_axis) not equal to n_binaries')

        # Generate eccentricity distribution
        if eccentricity_alpha is not None:
            if isinstance(eccentricity_alpha, (list, np.ndarray)):
                if len(eccentricity_alpha) == n_binaries:
                    self.eccentricity = random_sample_from_power_law(0., 1., eccentricity_alpha, n_binaries)
                else:
                    raise ValueError('len(eccentricity_alpha) != n_binaries')
            elif isinstance(eccentricity_alpha, (int, float)):
                if -1 < eccentricity_alpha:
                    self.eccentricity = random_sample_from_power_law(0., 1., eccentricity_alpha, n_binaries)
                else:
                    raise ValueError('eccentricity_alpha=%f not >-1' % eccentricity_alpha)

        elif eccentricity_k is not None:
            if isinstance(eccentricity_k, (list, np.ndarray)):
                if len(eccentricity_k) == n_binaries:
                    self.eccentricity = random_sample_from_linear_distribution(eccentricity_k, n_binaries)
                else:
                    raise ValueError('len(eccentricity_k) != n_binaries')
            elif isinstance(eccentricity_k, (int, float)):
                self.eccentricity = random_sample_from_linear_distribution(eccentricity_k, n_binaries)

        elif eccentricity is None:
            self.eccentricity = np.zeros(n_binaries)
        else:
            if isinstance(eccentricity, (int, float)):
                if 0. <= eccentricity <= 1.:
                    self.eccentricity = np.ones(n_binaries) * eccentricity
                else:
                    raise ValueError('eccentricity=%f not in [0, 1]' % eccentricity)
            elif isinstance(eccentricity, (list, np.ndarray)):
                if len(eccentricity) == n_binaries:
                    self.eccentricity = eccentricity
                else:
                    raise ValueError('len(eccentricity) not equal to n_binaries')

        # Set total mass
        if total_mass is None:
            self.total_mass = 2. * u.Msun
        elif isinstance(total_mass, u.quantity.Quantity):
            self.total_mass = total_mass
        elif isinstance(total_mass, (list, np.ndarray)):
            if len(total_mass) == n_binaries:
                self.total_mass = total_mass
            else:
                raise ValueError('len(total_mass) not equal to n_binaries')

        # Set distance
        if distance is None:
            self.distance = np.ones(n_binaries) * u.kpc
        elif isinstance(distance, u.quantity.Quantity):
            self.distance = np.ones(n_binaries) * distance
        elif isinstance(distance, (list, np.ndarray)):
            if len(distance) == n_binaries:
                self.distance = distance
            else:
                raise ValueError('len(distance) != n_binaries')

        self.true_anomaly = sample_true_anomaly(self.eccentricity, n_binaries, precision)

        if face_on:
            self.longitude_ascending_node = np.zeros(n_binaries)
            self.argument_periapsis = np.zeros(n_binaries)
            self.inclination = np.zeros(n_binaries)
        else:
            self.longitude_ascending_node = 2. * np.pi * np.random.uniform(size=n_binaries)
            self.argument_periapsis = 2. * np.pi * np.random.uniform(size=n_binaries)
            V = np.random.uniform(size=n_binaries)
            self.inclination = np.arccos(2 * V - 1)

        self.compute_orbital_properties()