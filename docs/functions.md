
# galaxy.py module

The module galaxy.py provides functions for simulating populations of stars in the Milky Way. It includes classes for representing different galactic components (e.g. thin disk, thick disk, halo), as well as functions for transforming coordinates and computing densities.

Exponential density profile
===========================

The `exponential_density` function returns the density profile of a galaxy with the following equation: 
$$ density = zpart \* rpart$$ where: $zpart = e^{\\frac{-abs(z-Zsun)}{H}}$ $rpart = e^{\\frac{-(r-Rsun)}{L}}$

Arguments
---------

*   `r` (list or array): galactocentric radius
*   `z` (list or array): galactocentric height
*   `H` (list or array): scaleheight
*   `L` (list or array): scalelength

Returns
-------

*   `density` (float or array): density at the given radius and height

Examples
--------

    x = exponential_density(8700, 100, 300, 2600)
    

Spheroid density profile
========================

The `spheroid_density` function returns the density profile of a spheroidal galaxy with the following equation: $$ density = \\frac{Rsun}{((r^2+\\frac{z^2}{q^2})^\\frac{1}{2})^n} $$ where: $Rsun$ is the galactocentric radius of the sun $q$ is the flattening parameter $n$ is the power-law exponent

Arguments
---------

*   `r` (list or array): galactocentric radius
*   `z` (list or array): galactocentric height
*   `q` (float): flattening parameter
*   `n` (float): power-law exponent

Returns
-------

*   `density` (float or array): density at the given radius and height

Examples
--------

    x = spheroid_density(8700, 100, 0.64, n=2.77)

GalacticComponent
=================

A meta class for galactic components.

Properties
----------

*   `H`: The height of the disk component.
*   `L`: The length of the disk component.
*   `q`: The q parameter of the halo component.
*   `n`: The n parameter of the halo component.


Transform to cylindrical coordinates
====================================

The `transform_to_cylindrical` function converts the given spherical coordinates to cylindrical coordinates. It returns the galactocentric radius and height in cylindrical coordinates.

Arguments
---------

*   `l` (list or array): galactic longitude (in radians)
*   `b` (list or array): galactic latitude (in radians)
*   `ds` (list or array): heliocentric distance



M31Halo
=======

The `M31Halo` class is a subclass of `GalacticComponent` that represents the power-law stellar density of the halo of the Andromeda galaxy (M31), as measured by Ibata et al. 2014.

Methods
-------

#### Arguments

*   `q` (float, optional): The flattening parameter of the halo. Defaults to 1.11.
*   `gamma` (float, optional): The power-law exponent of the halo. Defaults to -3.

### `stellar_density(self, r, z)`

Computes the stellar density at a particular position in the halo.

#### Arguments

*   `r` (float or array-like): The galactocentric radius of the position(s) to compute the density for, in units of length.
*   `z` (float or array-like): The galactocentric height of the position(s) to compute the density for, in units of length.

#### Returns

*   The unitless stellar density at the specified position(s). If `r` and `z` are arrays, the returned value will also be an array with the same shape.

Examples
--------

    m31halo = M31Halo()
    density = m31halo.stellar_density(100, -100)
    

Disk
====

A subclass of `GalacticComponent` representing the disk component of a galaxy.

Methods
-------

### stellar\_density(r, z)

Compute the stellar density at a particular position.

#### Arguments

*   `r`: galacto-centric radius (astropy.quantity)
*   `z`: galacto-centric height (astropy.quantity)

#### Returns

A unit-less stellar density.

#### Examples

    d = Disk.stellar_density(100*u.pc, -100*u.pc)

Halo
====

A subclass of GalacticComponent representing the halo component of a galaxy.

Methods
-------

### stellar\_density(r, z)

Compute the stellar density at a particular position.

#### Arguments

*   `r`: galacto-centric radius (astropy.quantity)
*   `z`: galacto-centric height (astropy.quantity)

#### Returns

A unit-less stellar density.

#### Examples

    d = Disk.stellar_density(100*u.pc, -100*u.pc)