from operators import ModelSpace, HamiltonianSystem
import numpy as np
from qutip import *
from matplotlib import pyplot as plt
from itertools import count
from numpy import pi, exp

# Parameters given in paper
GAMMA = 3.2  # [meV]
BETA = 25.9  # kB * T [meV]
OMEGA_CUT = 6  # Cut frequency [meV]
KAPPA = 17  # [meV]
OMEGA_R = 20.7  # [meV]
ETA = GAMMA / 2 / np.pi / BETA

# Parameters I chose
N = 1  # Number of molecules
OMEGA_C = 10 * OMEGA_R  # [meV]
OMEGA_M = OMEGA_C
G = OMEGA_R / 2 / np.sqrt(N)

OMEGA_MINUS = OMEGA_C * np.sqrt(1 - 2 * G / OMEGA_C)
OMEGA_PLUS = OMEGA_C * np.sqrt(1 + 2 * G / OMEGA_C)

OMEGA_P = .01 * OMEGA_C  # Magnitude of driving power

OMEGA_K = [215]  # Frequencies of each phonon mode, k
LAMBDA_K = [0]  # Phonon-cavity interactions for each phonon mode, k

OMEGA_D_RANGE = np.linspace(
    2 * OMEGA_MINUS - OMEGA_C, 2 * OMEGA_PLUS - OMEGA_C, 201)

# Other
GAMMA = 25.8  # [1/cm]
TEMP = 209  # [1/cm]
BETA = 1/TEMP  # [cm]
ETA = GAMMA * BETA / 2 / pi


# Define bath noise power function
def _s(omega, beta=BETA, eta=ETA, omega_cut=OMEGA_CUT, **kwargs):
    if omega < 0:
        return (pi * _j(-omega, eta=eta, omega_cut=omega_cut) *
                _n(-omega, beta=beta))
    elif omega > 0:
        return (pi * _j(omega, eta=eta, omega_cut=omega_cut) *
                (1 + _n(omega, beta=beta)))
    else:
        return pi * eta / beta


def _j(omega, eta, omega_cut):
    return eta * omega * exp(-(omega/omega_cut)**2)


def _n(omega, beta):
    return 1/(exp(omega * beta) - 1)


xdat = np.linspace(0, 500, 501)
ydat = np.empty_like(xdat)

# Plot bath-noise power spectrum
for x, i in zip(xdat, count()):
    ydat[i] = _s(omega=x)
fig, ax = plt.subplots(1, 1)
ax.plot(xdat, ydat)
ax.set_xlabel('omega_d - omega [meV]')
ax.set_ylabel('Bath noise power S(omega)')
plt.show()

assert False


# Construct our model space
ms = ModelSpace(num_molecules=N, num_phonon_modes=1, common_bath=True)

ham = HamiltonianSystem(
    ms=ms, omega_c=OMEGA_C, omega_m=OMEGA_M, g=G, lambda_k=LAMBDA_K,
    omega_k=OMEGA_K, Omega_p=OMEGA_P, kappa=KAPPA, S_func=_s
)


# Obtain absorption spectrum A(w) = Tr[rho_ss(w)*a] for each frequency
absorption = []
for omega_d, i in zip(OMEGA_D_RANGE, count()):
    ham = HamiltonianSystem(
        ms=ms,
        omega_c=OMEGA_C,
        omega_m=OMEGA_M,
        g=G,
        lambda_k=LAMBDA_K,
        omega_k=OMEGA_K,
        Omega_p=OMEGA_P,
        temperature=BETA,
        omega_cut=OMEGA_CUT,
        eta=ETA,
        kappa=KAPPA,
        omega_d=omega_d
    )
    # rho_ss = steadystate(ham.h(), c_op_list=ham.c_ops())
    # op = rho_ss * ms.annihilator_a
    # ab = abs(op.tr())
    spec = spectrum(
        H=ham.h(),
        wlist=[omega_d],
        c_ops=ham.c_ops(),
        a_op=ms.creator_a,
        b_op=ms.annihilator_a,
    )
    ab = spec[0]
    absorption.append(ab)
    print('Step {} of {}'.format(i+1, len(OMEGA_D_RANGE)))
    print('  omega_d =    {:16.8f}'.format(omega_d))
    print('  Absorption = {:16.8E}'.format(ab))

# Make our plot
xdat = OMEGA_D_RANGE
ydat = absorption

fig, ax = plt.subplots(1, 1)
ax.plot(xdat, ydat, '-')
ax.set_xlabel('omega [meV]')
ax.set_ylabel('Absorption spectrum')
ax.set_yscale('log')

plt.show()

