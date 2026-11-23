from typing import Any, Callable, Optional, Union, cast

import linear_operator
import numpy as np
import torch
import gpytorch

from .pep_compute_base import PEP_Compute_Base

class OSGPR_PEP(PEP_Compute_Base):
	def __init__(self, X: torch.Tensor, Y: torch.Tensor, kernel: gpytorch.kernels.Kernel, mu_old: torch.Tensor, Su_old: torch.Tensor, Kaa_old: torch.Tensor, Z_old: torch.Tensor, Z: torch.Tensor, alpha: float, lk: gpytorch.likelihoods.GaussianLikelihood, verify: bool = False):
		# `lk` is already a fully-built (possibly batched) Likelihood, so
		# PEP_Compute_Base.__init__'s dual-mode `likelihood` resolution just
		# passes it straight through to self.likelihood.
		super().__init__(X, Y, kernel, lk, z=Z, verify=verify)

		self.covar_module = kernel

		self.register_parameter(name="inducing_points", parameter=torch.nn.Parameter(Z))
		self.alpha = alpha

		self.mu_old = mu_old
		self.Su_old = Su_old
		self.Kaa_old = Kaa_old
		self.Z_old = Z_old
		self.mean_module = gpytorch.means.ZeroMean()
		self.verify = verify

		if self.verify:
			Ma = self.Z_old.shape[-2]
			assert self.mu_old.ndim >= 2, "mu_old must be at least a 2D tensor."
			assert self.mu_old.shape[-2] == Ma, "mu_old must have the same number of rows as Z_old."
			assert self.Su_old.ndim >= 2, "Su_old must be at least a 2D tensor."
			assert self.Su_old.shape[-2:] == (Ma, Ma), "Su_old's last two dimensions must match the number of rows in Z_old."
			assert self.Kaa_old.ndim >= 2, "Kaa_old must be at least a 2D tensor."
			assert self.Kaa_old.shape[-2:] == (Ma, Ma), "Kaa_old's last two dimensions must match the number of rows in Z_old."
			assert self.Z_old.ndim >= 2, "Z_old must be at least a 2D tensor."
			assert self.Z_old.shape == self.inducing_points.shape, "Z_old must have the same shape as inducing_points."

	def Kba(self):
		Kba = self.covar_module(self.inducing_points, self.Z_old)
		if self.verify:
			assert Kba.ndim >= 2, "Kba must be at least a 2D tensor."
			assert Kba.shape[-2:] == (self.M, self.Z_old.shape[-2]), "Kba's last two dimensions must be (Mb, Ma) where Mb is the number of inducing points and Ma is the number of old inducing points."
		return Kba

	def Lbinv_Kba(self, Lb):
		Lbinv_Kba = torch.linalg.solve_triangular(Lb, self.Kba().to_dense(), upper=False)
		if self.verify:
			assert Lbinv_Kba.ndim >= 2, "Lbinv_Kba must be at least a 2D tensor."
			assert Lbinv_Kba.shape[-2:] == (self.M, self.Z_old.shape[-2]), "Lbinv_Kba's last two dimensions must be (Mb, Ma) where Mb is the number of inducing points and Ma is the number of old inducing points."
		return Lbinv_Kba

	def Kaa_cur(self):
		Ma = self.Z_old.shape[-2]
		Kaa_cur = self.covar_module(self.Z_old, self.Z_old).to_dense() + linear_operator.operators.ConstantDiagLinearOperator(self.jitter, Ma)
		if self.verify:
			assert Kaa_cur.ndim >= 2, "Kaa_cur must be at least a 2D tensor."
			assert Kaa_cur.shape[-2:] == (Ma, Ma), "Kaa_cur's last two dimensions must be (Ma, Ma) where Ma is the number of old inducing points."
		return Kaa_cur


	def Kaa(self) -> torch.Tensor:
		# Tensor + LinearOperator (the diagonal jitter term) produces a
		# LinearOperator - densify here, matching every other kernel-derived
		# matrix in this class (e.g. Kba(), Kaa_cur()), since callers below
		# (_symmetrize_psd, torch.linalg.solve/cholesky) all expect a plain
		# Tensor.
		return (self.Kaa_old + linear_operator.operators.ConstantDiagLinearOperator(self.jitter, self.Z_old.shape[-2])).to_dense()

	def LSa(self):
		Ma = self.Z_old.shape[-2]
		LSa = torch.linalg.cholesky(self.Su_old)
		if self.verify:
			assert LSa.shape[-2:] == (Ma, Ma), "LSa's last two dimensions must be (Ma, Ma) where Ma is the number of old inducing points."
		return LSa

	# tmp disable fix to see if other fix does the trick already
	def _symmetrize_psd(self, matrix: torch.Tensor, regularization: Union[float, torch.Tensor] = 1e-8) -> torch.Tensor:
		return matrix
		sym_matrix = 0.5 * (matrix + matrix.transpose(-2, -1))
		if sym_matrix.ndim != 2:
			return sym_matrix
		eye = torch.eye(sym_matrix.shape[-1], device=sym_matrix.device, dtype=sym_matrix.dtype)
		eig_min = torch.linalg.eigvalsh(sym_matrix).min()
		if eig_min < regularization:
			sym_matrix = sym_matrix + (regularization - eig_min) * eye
		return sym_matrix

	def LM(self, Lbinv_Kba, Kab_Lbinv):
		Ma = self.Z_old.shape[-2]
		Kaadiff = self.Kaa_cur() - (Kab_Lbinv @ Lbinv_Kba)
		Kaadiff = self._symmetrize_psd(Kaadiff.to_dense(), regularization=self.jitter)
		LM = torch.linalg.cholesky(Kaadiff)
		if self.verify:
			assert LM.shape[-2:] == (Ma, Ma), "LM's last two dimensions must be (Ma, Ma) where Ma is the number of old inducing points."
		return LM

	def Sainv_ma(self):
		Ma = self.Z_old.shape[-2]
		Sainv_ma = torch.linalg.solve(self.Su_old, self.mu_old)
		if self.verify:
			assert Sainv_ma.shape[-2:] == (Ma, 1), "Sainv_ma's last two dimensions must be (Ma, 1) where Ma is the number of old inducing points."
		return Sainv_ma

	def d1(self, Lbinv_Kbf, Dff):
		Lbinv_Kbf_LDff = self.Lbinv_Kbf_LDff(Lbinv_Kbf, Dff)
		# .T reverses every dim; only the trailing (Mb, N) pair should transpose.
		d1 = Lbinv_Kbf_LDff @ Lbinv_Kbf_LDff.transpose(-2, -1)
		if self.verify:
			assert d1.shape[-2:] == (self.M, self.M), "d1's last two dimensions must be (Mb, Mb) where Mb is the number of inducing points."
		return d1

	def LD(self, Lbinv_Kbf, Dff, Lbinv_Kba, Da_Kab_Lbinv, LMT, Q):
		d1 = self.d1(Lbinv_Kbf, Dff)
		d2 = Lbinv_Kba @ Da_Kab_Lbinv
		LMT_Da_Kab_Lbinv = LMT @ Da_Kab_Lbinv
		Qinv_t1 = torch.linalg.solve(Q, LMT_Da_Kab_Lbinv) # fix torch.linalg.solve_triangular(Q, LMT_Da_Kab_Lbinv, upper=True)
		t1_Qinv_t1 = LMT_Da_Kab_Lbinv.transpose(-2, -1) @ Qinv_t1
		d3 = -self.alpha * t1_Qinv_t1
		eye_M = torch.eye(self.M, device=d1.device, dtype=d1.dtype)
		D = eye_M + d1 + d2 + d3 + self.jitter * eye_M
		D = self._symmetrize_psd(D, regularization=self.jitter) # fix cholesky decomposition by ensuring D is symmetric positive definite
		LD = torch.linalg.cholesky(D)
		if self.verify:
			assert LD.shape[-2:] == (self.M, self.M), "LD's last two dimensions must be (Mb, Mb) where Mb is the number of inducing points."
		return LD

	def Da_Kab_Lbinv(self, Kaa, Kab_Lbinv):
		Ma = self.Z_old.shape[-2]
		Sainv_Kab_Lbinv = torch.linalg.solve(self.Su_old, Kab_Lbinv)
		Kainv_Kab_Lbinv = torch.linalg.solve(Kaa, Kab_Lbinv)
		Da_Kab_Lbinv = Sainv_Kab_Lbinv - Kainv_Kab_Lbinv
		if self.verify:
			assert Da_Kab_Lbinv.shape[-2:] == (Ma, self.M), "Da_Kab_Lbinv's last two dimensions must be (Ma, Mb) where Ma is the number of old inducing points and Mb is the number of inducing points."
		return Da_Kab_Lbinv

	def SK_LM(self, LM, Kaa):
		Ma = self.Z_old.shape[-2]
		Sainv_LM = torch.linalg.solve(self.Su_old, LM)
		Kainv_LM = torch.linalg.solve(Kaa, LM)
		SK_LM = Sainv_LM - Kainv_LM
		if self.verify:
			assert SK_LM.shape[-2:] == (Ma, Ma), "SK_LM's last two dimensions must be (Ma, Ma) where Ma is the number of old inducing points."
		return SK_LM

	def _common_terms(self):
		Ma = self.Z_old.shape[-2]
		Sainv_ma = self.Sainv_ma()
		Kaa = self.Kaa()
		Lb = self.Lb()

		Lbinv_Kbf = self.Lbinv_Kbf(Lb)
		Lbinv_Kba = self.Lbinv_Kba(Lb)
		Dff = self.Dff(Lbinv_Kbf)
		Kab_Lbinv = Lbinv_Kba.transpose(-2, -1)
		Da_Kab_Lbinv = self.Da_Kab_Lbinv(Kaa, Kab_Lbinv)
		LM = self.LM(Lbinv_Kba, Kab_Lbinv)
		SK_LM = self.SK_LM(LM, Kaa)

		LMT = LM.transpose(-2, -1)
		LMT_SK_LM = LMT @ SK_LM
		Q = torch.eye(self.M, device=LMT_SK_LM.device, dtype=LMT_SK_LM.dtype) + self.alpha * LMT_SK_LM
		Q = self._symmetrize_psd(Q, regularization=self.jitter) # fix cholesky decomposition by ensuring D is symmetric positive definite

		LD = self.LD(Lbinv_Kbf, Dff, Lbinv_Kba, Da_Kab_Lbinv, LMT, Q)

		LMT_Sainv_ma = LMT @ Sainv_ma
		Lbinv_Kba_Da = Da_Kab_Lbinv.transpose(-2, -1)
		Lbinv_Kba_Da_LM = Lbinv_Kba_Da @ LM
		Qinv_LMT_Sainv_ma = torch.linalg.solve(Q, LMT_Sainv_ma) # fix torch.linalg.solve_triangular(Q, LMT_Da_Kab_Lbinv, upper=True)
		Sinv_y = self.Sinv_y(Dff)
		c1 = Lbinv_Kbf @ Sinv_y
		c2 = Lbinv_Kba @ Sainv_ma
		c3 = -self.alpha * Lbinv_Kba_Da_LM @ Qinv_LMT_Sainv_ma
		c = c1 + c2 + c3
		LDinv_c = torch.linalg.solve_triangular(LD, c, upper=False)

		Kaa = self._symmetrize_psd(Kaa, regularization=self.jitter) # fix cholesky decomposition by ensuring D is symmetric positive definite
		La = torch.linalg.cholesky(Kaa)

		if self.verify:
			assert Q.shape[-2:] == (Ma, Ma), "Q's last two dimensions must be (Ma, Ma) where Ma is the number of old inducing points."
			assert LD.shape[-2:] == (self.M, self.M), "LD's last two dimensions must be (Mb, Mb) where Mb is the number of inducing points."
			assert Sainv_ma.shape[-2:] == (Ma, 1), "Sainv_ma's last two dimensions must be (Ma, 1) where Ma is the number of old inducing points."
			assert LDinv_c.shape[-2:] == (self.M, 1), "LDinv_c's last two dimensions must be (Mb, 1) where Mb is the number of inducing points."
			assert La.shape[-2:] == (Ma, Ma), "La's last two dimensions must be (Ma, Ma) where Ma is the number of old inducing points."
			assert Lb.shape[-2:] == (Ma, Ma), "Lb's last two dimensions must be (Ma, Ma) where Ma is the number of old inducing points."

		return (La, Lb, LD, LDinv_c, Dff, Sainv_ma, Q, LM)

	def _maximum_log_likelihood_objective(self):
		Ma = self.Z_old.shape[-2]

		# self.likelihood.noise may carry a leading batch dim (one entry per
		# jointly-trained output); keep it as-is (shape (..., 1)) so it broadcasts
		# against the (..., N) Dff below instead of indexing into the batch.
		# .noise is typed Tensor | LinearOperator generically by gpytorch's
		# stubs, but GaussianLikelihood's noise is always a plain Tensor.
		sigma2 = cast(torch.Tensor, self.likelihood.noise)
		alpha = self.alpha
		ma = self.mu_old
		N = self.X.shape[0]

		LSa = self.LSa()

		(La, Lb, LD, LDinv_c, Dff, Sainv_ma, Q, LM) = self._common_terms()
		Q = self._symmetrize_psd(Q, regularization=self.jitter) # fix cholesky decomposition by ensuring D is symmetric positive definite
		LQ = torch.linalg.cholesky(Q)

		Lainv_ma = torch.linalg.solve_triangular(LSa, ma, upper=False)

		bound1 = -0.5 * N * torch.log(torch.tensor(2 * torch.pi))

		# Every term below reduces only over the trailing (per-output) dimensions,
		# so with a leading batch dimension (one entry per jointly-trained output)
		# the result is a (..., F) tensor of independent per-output
		# log-likelihoods rather than a single scalar. Without a batch dimension
		# this collapses to the original scalar.
		bound2 = -0.5 * (self.err().square() / Dff).sum(dim=-1)
		bound3 = -0.5 * Lainv_ma.square().sum(dim=(-2, -1))
		bound4 = 0.5 * LDinv_c.square().sum(dim=(-2, -1))

		ma_Sainv_LM_transposed = LM.transpose(-2, -1) @ Sainv_ma
		Qinv_LM_Sainv_ma = torch.linalg.solve(Q, ma_Sainv_LM_transposed) # fix torch.linalg.solve_triangular(Q, LMT_Da_Kab_Lbinv, upper=True)
		bound5 = 0.5 * alpha * (ma_Sainv_LM_transposed.transpose(-2, -1) @ Qinv_LM_Sainv_ma).sum(dim=(-2, -1))

		bound6 = -0.5 * Dff.log().sum(dim=-1)
		bound7 = - LD.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)

		bound8 = -0.5 * (1 - alpha) / alpha * (Dff / sigma2).log().sum(dim=-1)

		bound9 = - 1 / alpha * LQ.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
		bound10 = La.to_dense().diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
		bound11 = -LSa.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)

		bound = bound1 + bound2 + bound3 + bound4 + bound5 + bound6 + bound7 + bound8 + bound9 + bound10 + bound11

		if self.verify:
			assert Lainv_ma.ndim >= 2, "Lainv_ma must be at least a 2D tensor."
			assert Lainv_ma.shape[-2:] == (Ma, 1), "Lainv_ma's last two dimensions must be (Ma, 1) where Ma is the number of old inducing points."
			assert ma_Sainv_LM_transposed.ndim >= 2, "ma_Sainv_LM_transposed must be at least a 2D tensor."
			assert ma_Sainv_LM_transposed.shape[-2:] == (Ma, 1), "ma_Sainv_LM_transposed's last two dimensions must be (Ma, 1) where Ma is the number of old inducing points."
			assert Qinv_LM_Sainv_ma.ndim >= 2, "Qinv_LM_Sainv_ma must be at least a 2D tensor."
			assert Qinv_LM_Sainv_ma.shape[-2:] == (Ma, 1), "Qinv_LM_Sainv_ma's last two dimensions must be (Ma, 1) where Ma is the number of old inducing points."

		return bound

	def _predict_f(self, Xnew: torch.Tensor):
		Kbs = self.covar_module(self.inducing_points, Xnew)
		(La, Lb, LD, LDinv_c, Dff, Sainv_ma, Q, LM) = self._common_terms()

		Lbinv_Kbs = torch.linalg.solve_triangular(Lb, Kbs.to_dense(), upper=False)
		LDinv_Lbinv_Kbs = torch.linalg.solve_triangular(LD, Lbinv_Kbs, upper=False)

		# transpose (not .T) the trailing (Mb, N_new) pair only, and squeeze (not
		# flatten) the trailing size-1 dim so a leading per-output batch dim
		# survives.
		mean = (LDinv_Lbinv_Kbs.transpose(-2, -1) @ LDinv_c).squeeze(-1)

		Kss = self.covar_module(Xnew, Xnew).to_dense() + 1e-4 * torch.eye(Xnew.shape[-2], device=Xnew.device, dtype=Xnew.dtype)
		var1 = Kss
		var2 = -(Lbinv_Kbs.transpose(-2, -1) @ Lbinv_Kbs)
		var3 = LDinv_Lbinv_Kbs.transpose(-2, -1) @ LDinv_Lbinv_Kbs
		var = var1 + var2 + var3

		mean = mean + self.mean_module(Xnew)

		if self.verify:
			assert Lbinv_Kbs.ndim >= 2, "Lbinv_Kbs must be at least a 2D tensor."
			assert Lbinv_Kbs.shape[-2:] == (self.inducing_points.shape[-2], Xnew.shape[-2]), "Lbinv_Kbs's last two dimensions must be (Mb, Nnew) where Mb is the number of inducing points and Nnew is the number of new data points."
			assert LDinv_Lbinv_Kbs.ndim >= 2, "LDinv_Lbinv_Kbs must be at least a 2D tensor."
			assert LDinv_Lbinv_Kbs.shape[-2:] == (self.inducing_points.shape[-2], Xnew.shape[-2]), "LDinv_Lbinv_Kbs's last two dimensions must be (Mb, Nnew) where Mb is the number of inducing points and Nnew is the number of new data points."
			assert mean.shape[-1] == Xnew.shape[-2], "Mean's last dimension must equal the number of data points in Xnew."
			assert var.shape[-2:] == (Xnew.shape[-2], Xnew.shape[-2]), "Variance's last two dimensions must be (N, N) where N is the number of data points in Xnew."
		return gpytorch.distributions.MultivariateNormal(mean, var)
