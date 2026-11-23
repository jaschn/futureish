from typing import Union, cast

import linear_operator # type: ignore
import torch
import gpytorch # type: ignore

from .pep_compute_base import PEP_Compute_Base

class SGPR_PEP(PEP_Compute_Base):
	def D(self, Lbinv_Kbf: torch.Tensor, Dff: Union[torch.Tensor, linear_operator.LinearOperator]):
		Dff = cast(torch.Tensor, Dff)  # Ensure Dff is of type torch.Tensor
		Lbinv_Kbf_LDff = self.Lbinv_Kbf_LDff(Lbinv_Kbf, Dff)
		# .T reverses *all* dims, which is wrong once a leading batch dim is
		# present - transpose only the trailing (Mb, N) pair.
		d1 = Lbinv_Kbf_LDff @ Lbinv_Kbf_LDff.transpose(-2, -1)
		D = torch.eye(self.M, device=d1.device, dtype=d1.dtype) + d1
		if self.verify:
			assert D.shape[-2:] == (self.M, self.M), "D's last two dimensions must be (Mb, Mb)."
		return D

	def LD(self, Lbinv_Kbf: torch.Tensor, Dff: Union[torch.Tensor, linear_operator.LinearOperator]):
		D = self.D(Lbinv_Kbf, Dff)
		ld = torch.linalg.cholesky(D) # type: ignore
		LD = cast(torch.Tensor, ld)  # Ensure LD is of type torch.Tensor
		if self.verify:
			assert LD.shape[-2:] == (self.M, self.M), "LD's last two dimensions must be (Mb, Mb)."
		return LD

	def c(self, Lbinv_Kbf: torch.Tensor, Sinv_y: torch.Tensor):
		c = Lbinv_Kbf @ Sinv_y
		if self.verify:
			assert c.shape[-2:] == (self.M, 1), "c's last two dimensions must be (Mb, 1)."
		return c

	def LDinv_c(self, LD: torch.Tensor, Lbinv_Kbf: torch.Tensor, Dff: Union[torch.Tensor, linear_operator.LinearOperator]):
		Sinv_y = self.Sinv_y(Dff)
		c = self.c(Lbinv_Kbf, Sinv_y)
		LDinv_c = torch.linalg.solve_triangular(LD, c, upper=False) # type: ignore
		LDinv_c = cast(torch.Tensor, LDinv_c)  # Ensure LDinv_c is of type torch.Tensor
		if self.verify:
			assert LDinv_c.shape[-2:] == (self.M, 1), "LDinv_c's last two dimensions must be (Mb, 1)."
		return LDinv_c

	def _maximum_log_likelihood_objective(self):
		# Every term below reduces only over the trailing (per-output) dimensions,
		# so with a leading batch dimension (one entry per jointly-trained output,
		# see GPyTorchModelWrapper/next_online_model) the result is a (..., F)
		# tensor of independent per-output log-likelihoods rather than a single
		# scalar. Without a batch dimension this collapses to the original scalar.
		Lb = self.Lb()
		Lbinv_Kbf = self.Lbinv_Kbf(Lb)

		Dff = self.Dff(Lbinv_Kbf)
		Dff = cast(torch.Tensor, Dff)  # Ensure Dff is of type torch.Tensor

		LD = self.LD(Lbinv_Kbf, Dff)

		LDinv_c = self.LDinv_c(LD, Lbinv_Kbf, Dff)

		bound2 = -0.5 * (self.err().square() / Dff).sum(dim=-1)
		bound3 = 0.5 * LDinv_c.square().sum(dim=(-2, -1))

		bound4 = -0.5 * Dff.log().sum(dim=-1)
		bound5 = -LD.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)

		bound6 = -0.5 * (1 - self.alpha) / self.alpha * (Dff / self.sigma2).log().sum(dim=-1)

		return self._bound1 + bound2 + bound3 + bound4 + bound5 + bound6

	def _predict_f(self, Xnew: torch.Tensor):
		Lb = self.Lb()
		Lbinv_Kbf = self.Lbinv_Kbf(Lb)

		Dff = self.Dff(Lbinv_Kbf)
		LD = self.LD(Lbinv_Kbf, Dff)

		LDinv_c = self.LDinv_c(LD, Lbinv_Kbf, Dff)

		Kbs = self.covar_module(self.inducing_points, Xnew).to_dense() # type: ignore
		Kbs = cast(torch.Tensor, Kbs)  # Ensure Kbs is of type torch.Tensor

		Lbinv_Kbs = torch.linalg.solve_triangular(Lb, Kbs, upper=False) # type: ignore
		Lbinv_Kbs = cast(torch.Tensor, Lbinv_Kbs)  # Ensure Lbinv_Kbs is of type torch.Tensor
		LDinv_Lbinv_Kbs = torch.linalg.solve_triangular(LD, Lbinv_Kbs, upper=False) # type: ignore
		LDinv_Lbinv_Kbs = cast(torch.Tensor, LDinv_Lbinv_Kbs)  # Ensure LDinv_Lbinv_Kbs is of type torch.Tensor

		# transpose (not .T, which reverses every dim) the trailing (Mb, N_new)
		# pair only, and squeeze (not flatten) the trailing size-1 dim so a
		# leading per-output batch dim survives.
		mean = (LDinv_Lbinv_Kbs.transpose(-2, -1) @ LDinv_c).squeeze(-1)

		Kss = self.covar_module(Xnew) + linear_operator.operators.ConstantDiagLinearOperator(self.jitter, Xnew.shape[-2]) # type: ignore
		Kss = cast(torch.Tensor, Kss)  # Ensure Kss is of type torch.Tensor

		var1 = Kss
		var2 = -(Lbinv_Kbs.transpose(-2, -1) @ Lbinv_Kbs)
		var3 = LDinv_Lbinv_Kbs.transpose(-2, -1) @ LDinv_Lbinv_Kbs
		var = var1 + var2 + var3

		mean = mean + self.mean_module(Xnew) # type: ignore
		mean = cast(torch.Tensor, mean)  # Ensure mean is of type torch.Tensor

		if self.verify:
			assert mean.shape[-1] == Xnew.shape[-2], "Mean's last dimension must equal the number of new data points."
			assert var.shape[-2:] == (Xnew.shape[-2], Xnew.shape[-2]), "Variance's last two dimensions must be (N_new, N_new)."

		return gpytorch.distributions.MultivariateNormal(mean, var)