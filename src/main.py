from operators import ModelSpace, HamiltonianSystem
import numpy as np
from qutip import *
from matplotlib import pyplot as plt
from itertools import count
from numpy import pi, exp, real, imag, conj
from scipy.integrate import quad

# Parameters given in paper
GAMMA = 3.2  # [meV]
BETA = 25.9  # kB * T [meV]
OMEGA_CUT = 6  # Cut frequency [meV]
# KAPPA = 17  # [meV]
KAPPA = 1  # [meV]
# OMEGA_R = 20.7  # [meV]
OMEGA_R = 6.5  # [meV]
ETA = GAMMA * BETA / 2 / pi

# Parameters I chose
COMMON_BATH = False
N = 3  # Number of molecules
NP = 1  # Number of phonon moes
OMEGA_C = 215  # [meV]
OMEGA_M = OMEGA_C
G = OMEGA_R / 2 / np.sqrt(N)

OMEGA_MINUS = OMEGA_C * np.sqrt(1 - 2 * G / OMEGA_C)
OMEGA_PLUS = OMEGA_C * np.sqrt(1 + 2 * G / OMEGA_C)

OMEGA_P = .01 * OMEGA_C  # Magnitude of driving power

OMEGA_K = [215]  # Frequencies of each phonon mode, k
LAMBDA_K = [1/215]  # Phonon-cavity interactions for each phonon mode, k


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


def _phi(tau):
    def intfn(omega):
        e = exp(1j * tau * omega)
        n = _n(omega=omega, beta=BETA)
        j = _j(omega=omega, eta=ETA, omega_cut=OMEGA_CUT)
        return j * (conj(e) * (n + 1) + e * n)
    phi_real = quad(func=lambda x: real(intfn(x)), a=0, b=np.inf)[0]
    phi_imag = quad(func=lambda x: imag(intfn(x)), a=0, b=np.inf)[0]
    return phi_real + 1j * phi_imag


# # Plot bath-noise power spectrum
# xdat = np.linspace(-1.5*OMEGA_R, 1.5*OMEGA_R, 501)
# ydat = np.empty_like(xdat)
# for x, i in zip(xdat, count()):
#     ydat[i] = _s(omega=x)
# fig, ax = plt.subplots(1, 1)
# ax.plot(xdat, ydat, color='green')
# ax.axvline(-OMEGA_R)
# ax.axvline(-OMEGA_R/2)
# ax.axvline(0)
# ax.axvline(OMEGA_R/2)
# ax.axvline(OMEGA_R)
# ax.set_xlabel('omega_d - omega [meV]')
# ax.set_ylabel('Bath noise power S(omega)')
# plt.show()

# # Plot autocorrelation function
# xdat = np.linspace(0, 1, 501)
# ydat = np.empty(len(xdat), dtype=np.complex)
# for x, i in zip(xdat, count()):
#     ydat[i] = _phi(x)
# fig, ax = plt.subplots(1, 1)
# ax.plot(xdat, real(ydat), color='red')
# ax.plot(xdat, imag(ydat), color='blue')
# ax.set_xlabel('tau')
# ax.set_ylabel('Autocorrelation function, phi(tau)')
# plt.show()


# Construct our model space
ms = ModelSpace(
    num_molecules=N, num_phonon_modes=NP, common_bath=COMMON_BATH
)

# Obtain absorption spectrum A(w) = Tr[rho_ss(w)*a] for each frequency
xbat = np.linspace(190, 240, 501)
xbat_n = []
ybat_n = []
for omega_d, i in zip(xbat, count()):
    ham = HamiltonianSystem(
        ms=ms,
        omega_c=OMEGA_C,
        omega_m=OMEGA_M,
        g=G,
        lambda_k=LAMBDA_K,
        omega_k=OMEGA_K,
        Omega_p=OMEGA_P,
        kappa=KAPPA,
        omega_d=omega_d,
        S_func=_s
    )
    try:
        rho_ss = steadystate(
            ham.h_s()+ham.h_phi()+ham.h_d()+ham.h_b(),
            c_op_list=ham.c_ops()
        )
        op = rho_ss * ms.ann_a()
        ab = abs(op.tr())
        xbat_n.append(omega_d)
        ybat_n.append(ab)
    except KeyError:
        continue
    print('Step {} of {}'.format(i+1, len(xbat)))
    print('  omega_d =    {:16.8f}'.format(omega_d))
    print('  Absorption = {:16.8E}'.format(ab))

fig, ax = plt.subplots(1, 1)
ax.plot(xbat_n, ybat_n, '-')
ax.set_xlabel('omega [meV]')
ax.set_ylabel('Absorption spectrum')
# ax.set_yscale('log')

plt.show()
