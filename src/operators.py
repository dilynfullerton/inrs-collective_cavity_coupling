from collections import deque
from numpy import sqrt, pi, exp
from qutip import *


def _super_commutator(a_op, b_op):
    """Returns the superoperator S defined by
            S : rho --> [a * rho, b]
    """
    return sprepost(a_op, b_op) - spre(b_op * a_op)


class ModelSpace:
    def __init__(self, num_molecules, num_phonons, common_bath):
        self.n = num_molecules
        self.num_phonons = num_phonons
        self.common_bath = common_bath
        self._dim_c = 1
        self._dim_m = self.n
        if self.common_bath:
            self._dim_b = self.num_phonons
        else:
            self._dim_b = self.num_phonons * self.n
        self.dim = self._dim_c + self._dim_b + self._dim_m

        # Operators
        self.zero = qzero([2] * self.dim)
        self.one = qeye([2] * self.dim)
        self.annihilator_c = tensor(
            destroy(2),
            qeye([2] * self._dim_m),
            qeye([2] * self._dim_b)
        )
        self.creator_c = self.annihilator_c.dag()
        self.proj_dark = self.one - self.proj(
            [self.polar_plus, self.polar_minus])

        # Vectors
        self.vac = tensor([basis(2)] * self.dim)
        self.bright = self._bright()
        self.polar_plus = 1/sqrt(2) * (self.creator_c * self.vac + self.bright)
        self.polar_minus = 1/sqrt(2) * (self.creator_c * self.vac - self.bright)
        assert self.polar_plus.dag() * self.polar_minus == self.zero * self.vac

        self._annihilator_m0 = tensor(
            qeye([2] * self._dim_c),
            destroy(2),
            qeye([2] * (self._dim_m - 1)),
            qeye([2] * self._dim_b)
        )
        self._annihilator_b0 = tensor(
            qeye([2] * self._dim_c),
            qeye([2] * self._dim_m),
            destroy(2),
            qeye([2] * (self._dim_b - 1))
        )

    def annihilator_m(self, i):
        idx = self._dim_c + i
        ops = [qeye(2)] * self.dim
        ops.insert(idx, destroy(2))
        ops.pop(idx+1)
        return tensor(ops)

    def creator_m(self, i):
        return self.annihilator_m(i).dag()

    def annihilator_b(self, k, i=0):
        if self.common_bath:
            i = 0
        idx = self._dim_c + self._dim_m + i * self._dim_b + k
        ops = [qeye(2)] * self.dim
        ops.insert(idx, destroy(2))
        ops.pop(idx+1)
        return tensor(ops)

    def creator_b(self, k, i=None):
        return self.annihilator_b(k, i).dag()

    def sigma(self, v1, v2):
        """Returns the operator |v1><v2|
        """
        return v1 * v2.dag()

    def proj(self, vectors):
        """Returns the operator which projects onto span(vectors) with the
        regular Euclidean projection. The vectors given must be mutually
        orthonormal.
        :param vectors: A sequence of orthonormal (qutip) vectors
        :return: A qutip operator, which projects onto span(vectors)
        """
        p = 0
        for v in vectors:
            p += self.sigma(v, v)
        return p

    def dark_states(self):
        p_ortho = self.sigma(self.bright, self.bright)
        for i in range(self.n - 1):
            # Get random basis vector
            v = tensor(basis(2), rand_ket(2 * self.n, dims=[2]*self.n))
            # Remove projection of v onto other states
            v = v - p_ortho * v
            v = v.unit()
            # Update projection operator
            p_ortho = p_ortho + self.sigma(v, v)
            yield v

    def _bright(self):
        b = 0
        for i in range(self.n):
            b += self.creator_m(i) * self.vac
        return b / sqrt(self.n)


class HamiltonianSystem:
    def __init__(
            self, ms, omega_c, omega_m, g, lambda_k, omega_k, Omega_p,
            temperature, omega_cut, omega_d=0, gamma_e=None, gamma_a=None,
            Gamma_e=None, Gamma_a=None, gamma_phi=None,
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
        :param temperature: Bath temperature * k_B
        :param omega_cut: Cutoff frequency for system-bath coupling
        :param omega_d: Driving frequency
        :param gamma_e: Polariton-polariton connection
        :param gamma_a: Polariton-polariton connection
        :param Gamma_e: Polariton-dark connection
        :param Gamma_a: Polariton-dark connection
        :param gamma_phi: Bare molecule dephasing
        """
        self._eta = self.gamma_phi / 2 / pi * self._beta
        self._omega_cut = omega_cut
        self._beta = 1 / temperature
        self.ms = ms
        self.omega_d = omega_d
        self.omega_c = omega_c
        self.omega_m = omega_m
        self.omega_k = omega_k
        self.lambda_k = lambda_k
        self.g = g
        self.Omega_p = Omega_p
        self.Omega_R = 2 * self.g / sqrt(self.ms.n)

        # Gamma values from secular approximation
        self.gamma_e = 2 * self._s(
            self.Omega_R) if gamma_e is None else gamma_e
        self.gamma_a = 2 * self._s(
            -self.Omega_R) if gamma_a is None else gamma_a
        self.Gamma_e = 2 * self._s(
            self.Omega_R/2) if Gamma_e is None else Gamma_e
        self.Gamma_a = 2 * self._s(
            -self.Omega_R/2) if Gamma_a is None else Gamma_a
        self.gamma_phi = 2 * self._s(0) if gamma_phi is None else gamma_phi

    def _j(self, omega):
        return self._eta * omega * exp(-(omega/self._omega_cut)**2)

    def _n(self, omega):
        return (exp(omega * self._beta) - 1)**(-1)

    def _s(self, omega):
        if omega < 0:
            return pi * self._j(-omega) * self._n(-omega)
        elif omega > 0:
            return pi * self._j(omega) * (1 + self._n(omega))
        else:
            return pi * self._eta / self._beta

    def h_s(self):
        """System Hamiltonian
        """
        a = self.ms.annihilator_c
        c = self.ms.annihilator_m
        hs = self.omega_c * a.dag() * a
        for i in range(self.ms.n):
            hs += self.omega_m * c(i).dag() * c(i)
            hs += self.g * (a * c(i).dag() + a.dag() * c(i))
        return hs

    def h_phi(self):
        """System-bath interaction Hamiltonian
        """
        c = self.ms.annihilator_c
        b = self.ms.annihilator_b
        hphi = 0
        for i in range(self.ms.n):
            hphi_n = 0
            for k in range(self.ms.num_phonons):
                hphi_n += self.lambda_k[k] * (b(k, i).dag() + b(k, i))
            hphi += c(i).dag() * c(i) * hphi_n
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
        return self.Omega_p * self.ms.annihilator_c

    def _h_b_com(self):
        hb = 0
        b = self.ms.annihilator_b
        for k in range(self.ms.num_phonons):
            hb += self.omega_k[k] * b(k).dag() * b(k)

    def _h_b_ind(self):
        hb = 0
        b = self.ms.annihilator_b
        for k in range(self.ms.num_phonons):
            for i in range(self.ms.n):
                hb += self.omega_k[k] * b(k, i).dag() * b(k, i)

    def c_ops(self):
        """Returns list of collapse operators
        """
        if self.ms.common_bath:
            return self._c_ops_com()
        else:
            return self._c_ops_ind()

    def _c_ops_com(self):
        sig = self.ms.sigma
        d = self.ms.proj_dark
        pp = self.ms.polar_plus
        pm = self.ms.polar_minus
        return [
            sqrt(self.gamma_a/4) * sig(pp, pm),
            sqrt(self.gamma_e/4) * sig(pm, pp),
            sqrt(self.gamma_phi/4) * sig(pp, pp),
            sqrt(self.gamma_phi/4) * sig(pm, pm),
            sqrt(self.gamma_phi**2/2) * d,
        ]

    def _c_ops_ind(self):
        sig = self.ms.sigma
        pp = self.ms.polar_plus
        pm = self.ms.polar_minus
        projd = self.ms.proj_dark
        c = self.ms.annihilator_c
        n = self.ms.n
        cops = [
            sqrt(self.gamma_a/4/n) * sig(pp, pm),
            sqrt(self.gamma_e/4/n) * sig(pm, pp),
            sqrt(self.gamma_phi/4/n) * sig(pp, pp),
            sqrt(self.gamma_phi/4/n) * sig(pm, pm),
        ]
        for d in self.ms.dark_states():
            cops.extend(
                [
                    sqrt(self.Gamma_a/2/n) * sig(d, pm),
                    sqrt(self.Gamma_a/2/n) * sig(pp, d),
                    sqrt(self.Gamma_e/2/n) * sig(d, pp),
                    sqrt(self.Gamma_e/2/n) * sig(pm, d),
                    self.Gamma_a / 4 / n * (
                        _super_commutator(sig(d, pm), sig(d, pp)) -
                        _super_commutator(sig(pp, d), sig(pm, d))
                    ),
                    self.Gamma_e / 4 / n * (
                        _super_commutator(sig(d, pp), sig(d, pm)) -
                        _super_commutator(sig(pm, d), sig(pp, d))
                    ),
                    sqrt(self.gamma_phi/4/n) * sig(pp, pp)
                ]
            )
        for i in range(n):
            cops.append(
                sqrt(self.gamma_phi) * (projd * c(i).dag() * c(i) * projd)
            )
        return cops
