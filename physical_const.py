# physical_const.py: set of physical constants
# Copyright(C) 2018 Romain Serra
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Software Foundation, either version 3 of the License, or any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.
# If not, see < https://www.gnu.org/licenses/>.

# distances
radius_Earth = 6371.e3
alt_geo = 35786.e3
radius_geo = radius_Earth + alt_geo
dist_Earth_Moon = 384400.e37
astro_unit = 149597870.7e3

# masses
mass_Sun = 1.989e30
mass_Earth = 5.972e24 
mass_Moon = 7.342e22
mass_EM = mass_Earth + mass_Moon

# mass ratios
mu_EM = mass_Moon / mass_EM
mu_SE = mass_EM / (mass_Sun + mass_EM)

# gravity
G = 6.674e-11
Earth_constant = G * mass_Earth
Moon_constant = G * mass_Moon
EM_constant = G * mass_EM
Sun_constant = G * mass_Sun
