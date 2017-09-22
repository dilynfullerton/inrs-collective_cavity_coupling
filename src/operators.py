from numpy import sqrt, pi, exp
from qutip import *
from collections import deque


def orthogonalize(states, zero_op):
    """Given a sequence of states, returns a generator of mutually
    orthogonal states generating span(states).
    The number of returned states will be of the same size, containing
    zeros if |states| > dim(span(states)).
    """
    psf = zero_op
    states = deque(states)
    while len(states) > 0:
        s = states.pop()
        s -= psf * s
        psf += orthonormal_projection(s, zero_op=zero_op)
        yield s


def orthonormal_basis(states, zero_op):
    """Given a sequence of states, returns a generator of an orthonormal
    basis, generating span(states)
    """
    for s in orthogonalize(states, zero_op=zero_op):
        if s.norm() > 0:
            yield s.unit()


def orthonormal_projection(states, zero_op):
    if isinstance(states, Qobj):
        states = [states]
    return _orthonormal_projection(states=deque(states), zero_op=zero_op)


def _orthonormal_projection(states, zero_op, p_rest=None):
    if len(states) == 0:
        return zero_op
    elif len(states) == 1:
        s = states.pop()
        if s.norm() > 0:
            s = s.unit()
        return s * s.dag()
    else:
        s = states.pop()
        if p_rest is None:
            p_rest = _orthonormal_projection(states, zero_op=zero_op)
        return orthonormal_projection(s - p_rest * s, zero_op=zero_op) + p_rest


def _super_commutator(a_op, b_op):
    """Returns the superoperator S defined by
            S : rho --> [a * rho, b]
    """
    return sprepost(a_op, b_op) - spre(b_op * a_op)


def _outer_product(v1, v2=None):
    if v2 is None:
        v2 = v1
    return v1 * v2.dag()


class ModelSpace:
    def __init__(self, num_molecules, num_phonon_modes, common_bath):
        self.n = num_molecules
        self.num_phonons = num_phonon_modes
        self.common_bath = common_bath
        self.dim_vib = self.n
        if self.common_bath:
            self.dim_bath = self.num_phonons
        else:
            self.dim_bath = self.num_phonons * self.n
        self.dim = 1 + self.dim_vib + self.dim_bath

    def vac(self):
        return tensor([fock_dm(2)] * self.dim)

    def zero(self):
        return qzero([2] * self.dim)

    def one(self):
        return qeye([2] * self.dim)

    def ann_a(self):
        return tensor(
            [destroy(2)] +
            [qeye(2)] * self.dim_vib +
            [qeye(2)] * self.dim_bath
        )

    def cre_a(self):
        return self.ann_a().dag()

    def _ann_ops(self, idx):
        one = [qeye(2)] * self.dim
        one.insert(idx, destroy(2))
        one.pop(idx+1)
        return tensor(one)

    def ann_c(self, i):
        """Annihilator operator for ith molecule state
        """
        return self._ann_ops(idx=1+i)

    def cre_c(self, i):
        """Creator operator for ith molecule state
        """
        return self.ann_c(i).dag()

    def ann_b(self, k, i=0):
        """Annihilator operator for kth phonon mode associated with the
        ith molecule
        """
        if self.common_bath:
            return self._ann_ops(idx=1+self.dim_vib+k)
        else:
            return self._ann_ops(idx=1+self.dim_vib+k*self.n+i)

    def cre_b(self, k, i=0):
        """Creator operator for the kth phonon mode associated with the ith
        molecule state
        """
        return self.ann_b(k=k, i=i).dag()


class HamiltonianSystem:
    def __init__(
            self, ms, omega_c, omega_m, g, lambda_k, omega_k, Omega_p,
            omega_d, kappa, S_func, gamma_e=None, gamma_a=None, Gamma_e=None,
            Gamma_a=None, gamma_phi=None,
    ):
        """
        If any of the term gamma_e, gamma_a, Gamma_e, Gamma_a, or gamma_phi
        are not provided, these will be determined from the bath noise power
        spectrum S, according to
                gamma_i = 2 * S( omega_i ),
        where for the gamma terms listed above respectively, omega_i is
        Omega_R, -Omega_r, Omega_R/2, -Omega_R/2, 0.
        :param ms: Model space
        :param omega_c: Cavity mode frequency
        :param omega_m: Molecule mode frequency
        :param g: Cavity-molecule coupling frequency
        :param lambda_k: Array of phonon-molecule coupling coefficients,
        one for each phonon mode k
        :param omega_k: Array of phonon mode frequencies for phonon modes
        the the bath
        :param gamma_e: Polariton-polariton connection
        :param gamma_a: Polariton-polariton connection
        :param Gamma_e: Polariton-dark connection
        :param Gamma_a: Polariton-dark connection
        :param gamma_phi: Bare molecule dephasing
        """
        self.ms = ms
        self.N = self.ms.n
        self.Np = self.ms.num_phonons
        self.kappa = kappa

        self._S_func = S_func

        # Operators
        self._a = self.ms.ann_a()
        self._c = self.ms.ann_c
        self._b = self.ms.ann_b
        self._one = self.ms.one()
        self._zero = self.ms.zero()

        # Frequencies
        self.omega_c = omega_c
        self.omega_m = omega_m
        self.omega_k = omega_k
        self.omega_d = omega_d
        self.Omega_p = Omega_p

        # Coupling coefficients
        self.lambda_k = lambda_k
        self.g = g

        # Gamma values from secular approximation
        self.Omega_R = 2 * sqrt(self.ms.n) * self.g
        self.gamma_e = self._set_gamma(gamma_e, self.Omega_R)
        self.gamma_a = self._set_gamma(gamma_a, -self.Omega_R)
        self.Gamma_e = self._set_gamma(Gamma_e, self.Omega_R/2)
        self.Gamma_a = self._set_gamma(Gamma_a, -self.Omega_R/2)
        self.gamma_phi = self._set_gamma(gamma_phi, 0)

        # Important vectors
        self.VAC = self.ms.vac()
        self.B = self._bright()  # bright state
        self.UP = 1/sqrt(2) * (self._a * self.VAC + self.B)
        self.LP = 1/sqrt(2) * (self._a * self.VAC - self.B)

    def _proj_dark(self):
        return self._one - orthonormal_projection(
            states=[self.UP, self.LP], zero_op=self._zero)

    def _bright(self):
        b = 0
        for i in range(self.ms.n):
            b += self._c(i) * self.VAC
        return b / sqrt(self.ms.n)

    def _u_m_i(self, m, i):
        m += 1
        i += 1
        return exp(1j * 2*pi * m * i / self.N) / sqrt(self.N)

    def _dark_states0(self):
        for i in range(self.N - 1):
            di1 = self.ms.zero() * self.VAC
            for m in range(self.N):
                umi = self._u_m_i(m, i)
                di1 += umi * self._c(m).dag() * self.VAC
            yield di1

    def _dark_states(self):
        return orthonormal_basis(
            states=self._dark_states0(), zero_op=self._zero)

    def h(self):
        return self.h_s() + self.h_d() + self.h_phi() + self.h_b()

    def h_s(self):
        """System Hamiltonian, in rotating wave approximation
        """
        hs = self.omega_c * self._a.dag() * self._a
        for i in range(self.ms.n):
            hs += self.omega_m * self._c(i).dag() * self._c(i)
            hs += self.g * (
                self._a * self._c(i).dag() + self._a.dag() * self._c(i))
        return hs

    def h_phi(self):
        """System-bath interaction Hamiltonian
        """
        hphi = 0
        for i in range(self.N):
            hphi_n = 0
            for k in range(self.Np):
                hphi_n += self.lambda_k[k] * (
                    self._b(k, i).dag() + self._b(k, i))
            hphi += self._c(i).dag() * self._c(i) * hphi_n
        return hphi

    def h_b(self):
        """Bath Hamiltonian
        """
        if self.ms.common_bath:
            return self._h_b_com()
        else:
            return self._h_b_ind()

    def h_d(self):
        """Driving Hamiltonian
        """
        a = self._a
        return self.Omega_p * (a + a.dag()) - self.omega_d * a.dag() * a

    def _h_b_com(self):
        hb = 0
        for k in range(self.Np):
            hb += self.omega_k[k] * self._b(k).dag() * self._b(k)
        return hb

    def _h_b_ind(self):
        hb = 0
        for k in range(self.Np):
            for i in range(self.N):
                hb += self.omega_k[k] * self._b(k, i).dag() * self._b(k, i)
        return hb

    def c_ops(self):
        """Returns list of collapse operators
        """
        if self.ms.common_bath:
            return self._c_ops_com()
        else:
            return self._c_ops_ind()

    def _c_ops_com(self):
        sig = _outer_product
        d = self._proj_dark()
        return [
            sqrt(self.gamma_a/4) * sig(self.UP, self.LP),
            sqrt(self.gamma_e/4) * sig(self.LP, self.UP),
            sqrt(self.gamma_phi/4) * sig(self.UP),
            sqrt(self.gamma_phi/4) * sig(self.LP),
            sqrt(self.gamma_phi) * d,
            sqrt(self.kappa) * self._a
        ]

    def _c_ops_ind(self):
        sig = _outer_product
        projd = self._proj_dark()
        n = self.N
        pp = self.UP
        pm = self.LP
        cops = [
            sqrt(self.gamma_a/4/n) * sig(pp, pm),
            sqrt(self.gamma_e/4/n) * sig(pm, pp),
            sqrt(self.gamma_phi/4/n) * sig(pp),
            sqrt(self.gamma_phi/4/n) * sig(pm),
            sqrt(self.kappa) * self._a
        ]
        for d in self._dark_states():
            sop1 = self.Gamma_a / 4 / n * (
                _super_commutator(sig(d, pm), sig(d, pp)) -
                _super_commutator(sig(pp, d), sig(pm, d))
            )
            sop2 = self.Gamma_e / 4 / n * (
                _super_commutator(sig(d, pp), sig(d, pm)) -
                _super_commutator(sig(pm, d), sig(pp, d))
            )
            cops.extend(
                [
                    sqrt(self.Gamma_a/2/n) * sig(d, pm),
                    sqrt(self.Gamma_a/2/n) * sig(pp, d),
                    sqrt(self.Gamma_e/2/n) * sig(d, pp),
                    sqrt(self.Gamma_e/2/n) * sig(pm, d),
                    sop1 + sop1.dag(),
                    sop2 + sop2.dag(),
                    sqrt(self.gamma_phi/4/n) * sig(pp, pp)
                ]
            )
        for i in range(n):
            cops.append(
                sqrt(self.gamma_phi) * (
                    projd * self._c(i).dag() * self._c(i) * projd)
            )
        return cops

    def _set_gamma(self, param, omega=None):
        if param is not None:
            return param
        elif omega is not None and self._S_func is not None:
            return 2 * self._S_func(omega)
        else:
            return 0
