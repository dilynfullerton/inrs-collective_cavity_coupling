from numpy import sqrt, pi, exp
from qutip import *
from itertools import product


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

        self.vac = tensor([basis(2)] * self.dim)  # vacuum state

        self.zero = qzero([2] * self.dim)  # zero operator
        self.one = qeye([2] * self.dim)  # identity operator

        # Creation and annihilation operators for cavity, labeled a
        self.annihilator_a = tensor(
            destroy(2),
            qeye([2] * self._dim_m),
            qeye([2] * self._dim_b)
        )
        self.creator_a = self.annihilator_a.dag()

    def annihilator_c(self, i):
        """Annihilator operator for ith molecule state
        """
        idx = self._dim_c + i
        ops = [qeye(2)] * self.dim
        ops.insert(idx, destroy(2))
        ops.pop(idx+1)
        return tensor(ops)

    def creator_c(self, i):
        """Creator operator for ith molecule state
        """
        return self.annihilator_c(i).dag()

    def total_annihilator_c(self):
        op = 0
        for i in range(self._dim_m):
            op += self.annihilator_c(i)
        return op

    def total_creator_c(self):
        return self.total_annihilator_c().dag()

    def annihilator_b(self, k, i=0):
        """Annihilator operator for kth phonon mode associated with the
        ith molecule
        """
        if self.common_bath:
            i = 0
        idx = self._dim_c + self._dim_m + i * self._dim_b + k
        ops = [qeye(2)] * self.dim
        ops.insert(idx, destroy(2))
        ops.pop(idx+1)
        return tensor(ops)

    def total_annihilator_b(self, k):
        if self.common_bath:
            return self.annihilator_b(k=k)
        else:
            op = 0
            for i in range(self._dim_m):
                op += self.annihilator_b(k=k, i=i)
            return op

    def total_creator_b(self, k):
        return self.total_annihilator_b(k=k).dag()

    def creator_b(self, k, i=0):
        """Creator operator for the kth phonon mode associated with the ith
        molecule state
        """
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


class HamiltonianSystem:
    def __init__(
            self, ms, omega_c, omega_m, g, lambda_k, omega_k, Omega_p,
            temperature, omega_cut, eta, kappa,
            omega_d=0, gamma_e=None, gamma_a=None,
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
        self.ms = ms
        self._omega_cut = omega_cut
        self._beta = 1 / temperature
        self._eta = eta
        self._kappa = kappa

        # Frequencies
        self.omega_c = omega_c
        self.omega_m = omega_m
        self.omega_k = omega_k
        self.lambda_k = lambda_k
        self.g = g
        self.Omega_p = Omega_p

        # Adjust frequencies for to driving frequency
        self.omega_d = omega_d

        # Gamma values from secular approximation
        self.Omega_R = 2 * sqrt(self.ms.n) * self.g
        self.gamma_e = 2 * self._s(
            self.Omega_R) if gamma_e is None else gamma_e
        self.gamma_a = 2 * self._s(
            -self.Omega_R) if gamma_a is None else gamma_a
        self.Gamma_e = 2 * self._s(
            self.Omega_R/2) if Gamma_e is None else Gamma_e
        self.Gamma_a = 2 * self._s(
            -self.Omega_R/2) if Gamma_a is None else Gamma_a
        self.gamma_phi = 2 * self._s(0) if gamma_phi is None else gamma_phi

        # Important vectors
        self.bright = self._bright()  # bright state
        self.polar_plus = 1/sqrt(2) * (
            self.ms.creator_a * self.ms.vac + self.bright)
        self.polar_minus = 1/sqrt(2) * (
            self.ms.creator_a * self.ms.vac - self.bright)

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

    def _proj_dark(self):
        return self.ms.one - self.ms.proj([self.polar_plus, self.polar_minus])

    def _dark_states(self):
        dark_ham = self.h_s() * self._proj_dark()
        dark_states = dark_ham.eigenstates()[1]
        assert len(dark_states) == self.ms.n - 1
        for d1 in dark_states:
            assert d1 == d1.unit()
            for d2 in dark_states:
                assert d1.dag() * d2 == 0
        return dark_states

    def _bright(self):
        b = 0
        for i in range(self.ms.n):
            b += self.ms.creator_c(i) * self.ms.vac
        return b / sqrt(self.ms.n)

    def h(self):
        # return self.h_s()
        # return self.h_s() + self.h_d()
        # return self.h_phi() + self.h_d()
        # return self.h_s() + self.h_d() + self.h_phi() + self.h_b()
        return self.h_s() + self.h_d(0)

    def h_s(self):
        """System Hamiltonian
        """
        a = self.ms.annihilator_a  # Cavity annihilation operator
        c = self.ms.annihilator_c  # Molecular annihilation operator
        hs = self.omega_c * a.dag() * a
        for i in range(self.ms.n):
            hs += self.omega_m * c(i).dag() * c(i)
            hs += self.g * (a * c(i).dag() + a.dag() * c(i))
        return hs

    def h_phi(self):
        """System-bath interaction Hamiltonian
        """
        c = self.ms.annihilator_c  # Molecular annihilation operator
        b = self.ms.annihilator_b  # Bath annihilation operator
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

    def h_d(self, t):
        """Driving Hamiltonian
        """
        a = self.ms.annihilator_a
        hd1 = self.Omega_p * (exp(-1j * self.omega_d * t)) * a
        return hd1 + hd1.dag()

    def _h_b_com(self):
        hb = 0
        b = self.ms.annihilator_b
        for k in range(self.ms.num_phonons):
            hb += self.omega_k[k] * b(k).dag() * b(k)
        return hb

    def _h_b_ind(self):
        hb = 0
        b = self.ms.annihilator_b
        for k in range(self.ms.num_phonons):
            for i in range(self.ms.n):
                hb += self.omega_k[k] * b(k, i).dag() * b(k, i)
        return hb

    def c_ops(self):
        """Returns list of collapse operators
        """
        if self.ms.common_bath:
            return self._c_ops_com()
        else:
            return self._c_ops_ind()

    def _c_ops_com(self):
        sig = self.ms.sigma
        d = self._proj_dark()
        pp = self.polar_plus
        pm = self.polar_minus
        return [
            sqrt(self.gamma_a/4) * sig(pp, pm),
            sqrt(self.gamma_e/4) * sig(pm, pp),
            sqrt(self.gamma_phi/4) * sig(pp, pp),
            sqrt(self.gamma_phi/4) * sig(pm, pm),
            sqrt(self.gamma_phi**2/4) * d,
            sqrt(self._kappa) * self.ms.annihilator_a,
        ]

    def _c_ops_ind(self):
        sig = self.ms.sigma
        pp = self.polar_plus
        pm = self.polar_minus
        projd = self._proj_dark()
        c = self.ms.annihilator_c
        n = self.ms.n
        cops = [
            sqrt(self.gamma_a/4/n) * sig(pp, pm),
            sqrt(self.gamma_e/4/n) * sig(pm, pp),
            sqrt(self.gamma_phi/4/n) * sig(pp, pp),
            sqrt(self.gamma_phi/4/n) * sig(pm, pm),
            sqrt(self._kappa) * self.ms.annihilator_a
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
                sqrt(self.gamma_phi) * (projd * c(i).dag() * c(i) * projd)
            )
        return cops
