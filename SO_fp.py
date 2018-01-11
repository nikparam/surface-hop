from functools import partial, reduce
import numpy as np
import matplotlib.pyplot as plt

#########################################################################################
# Composition of N functions
#########################################################################################

def compose(*func):
	init, *rest = reversed(func)
	return lambda *args, **kws: reduce( lambda a, x: x(a), rest, init(*args, **kws) )

#########################################################################################
# Decorate function g with func --> result = decorator(func)(g) = func( g )
#########################################################################################

def decorator(func):
	return lambda x: compose(func, x)

#########################################################################################
# Map resulting in list ( Python 2 style )
#########################################################################################

lmap = compose( list, map )

#########################################################################################
# Parameters of particle
#########################################################################################

mass = 1
omega = 1
force_const = mass * omega**2

#########################################################################################
# Initial conditions
#########################################################################################

q = 2
p = 0

get_xi  = lambda x, y: omega * x + 1j * y
get_eta = lambda x, y: 0.25 * np.log( omega / np.pi ) - 0.5 * omega * x**2 - 1j * x * y

#########################################################################################
# Parameters of map
#########################################################################################

Lim = 8.0
Npoints = 1024
delta = 2.0 * Lim / Npoints

#########################################################################################
# Mappings
#########################################################################################

x_mapping = np.linspace( -Lim, Lim, Npoints )

#########################################################################################
# Energies
#########################################################################################

V_energy = lambda x: 0.5 * force_const * x**2 

shift_map = compose( np.fft.fftshift, np.fft.fftfreq)
T_energy = compose( lambda x: 0.5 * x**2 / mass, shift_map )

#########################################################################################
# Propagators
#########################################################################################

def propagator(energy):
	return compose( lambda x: np.exp( -1j* x * dt), energy)

#########################################################################################
# Initial wave function
#########################################################################################

get_Psi_t = lambda x: np.exp( - 0.5 * omega * x**2 + get_xi(q,p) * x + get_eta(q,p) )
norm_Psi  = compose( lambda x: x * delta, sum, lambda x: x**2, np.abs )

#########################################################################################
# Parameters of propagation
#########################################################################################

dt = 0.01
Nsteps = 2500

#########################################################################################
# | Psi( t + dt ) > = U * | Psi( t ) >
#########################################################################################

def mult(seq1, seq2):
	return lmap( lambda x, y: x * y, seq1, seq2 ) 

U_x_to_p = partial( compose( np.fft.ifftshift, np.fft.ifft, mult ), \
		    propagator(V_energy)(x_mapping) )
U_p_to_x = partial( compose( np.fft.fft, np.fft.fftshift, mult ), \
		    propagator(T_energy)(Npoints, delta) )
full_U   = compose( U_p_to_x, U_x_to_p )

psi_mapping_t = get_Psi_t( x_mapping )

plt.ion()
fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.8,0.8])
ax.set_autoscale_on(True)
ax.set_ylim([-0.01,2.5])
line, = ax.plot( x_mapping, np.abs(get_Psi_t(x_mapping)) )

check_true = lambda x: x%25 == 0

for _ in range(Nsteps):
	psi_mapping_next_t = full_U(psi_mapping_t)
	psi_mapping_t = psi_mapping_next_t[:]

	if check_true(_):
		line.set_ydata( ( lambda x: abs(x) )(psi_mapping_t) )
		ax.autoscale_view(True, True, True)
		plt.draw()
		plt.pause(0.1)
