#*******************************************************************************#
#  stepsic - An initial condition generator for                                 #
#            STEreographically Projected cosmological Simulations               #
#    Copyright (C) 2017-2025 Gabor Racz, Balazs Pal                             #
#                                                                               #
#    This program is free software; you can redistribute it and/or modify       #
#    it under the terms of the GNU General Public License as published by       #
#    the Free Software Foundation; either version 2 of the License, or          #
#    (at your option) any later version.                                        #
#                                                                               #
#    This program is distributed in the hope that it will be useful,            #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of             #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              #
#    GNU General Public License for more details.                               #
#*******************************************************************************#

import astropy.units as u

#G = 6.67430e-8  # Gravitational constant in cm^3 g^-1 s^-2
G = 1

# StePS internal units
UNIT_D_P = (1 * u.Mpc).to(u.cm)                  # Unit distance
UNIT_T_P = 47.14829951063323 * u.Gyr             # Unit time
UNIT_V_P = UNIT_D_P.to(u.km) / UNIT_T_P.to(u.s)  # Unit velocity

UNIT_D = UNIT_D_P.value
UNIT_T = UNIT_T_P.value
UNIT_V = UNIT_V_P.value