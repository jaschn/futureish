from typing import Optional, Union, cast

import polars as pl

from ..utils.polars import AnyFrame, FrameT, frame_columns
from .data import ScalerFactory
from .base import ScalerBase
from .z_score import ZScoreScaler
from .._internal.constants import TIME_CL

class ChainedScaler(ScalerBase):
	def __init__(self, df: FrameT, raw_column_name: str, feature_column_name: str, *scaling_cls: ScalerFactory):
		super().__init__(df, raw_column_name, feature_column_name)
		assert len(scaling_cls) > 0, "ChainedScaler requires at least one scaler"

		self.scaling: list[ScalerBase] = []
		for scaler_cls in scaling_cls[:-1]:
			scaler = scaler_cls(df, raw_column_name, feature_column_name)
			self.scaling.append(scaler)
			df = scaler.scale(df).rename({self.feature_column_name: self.raw_column_name})

		scaler = scaling_cls[-1](df, raw_column_name, feature_column_name)
		self.scaling.append(scaler)

	@classmethod
	def from_scalers(cls, *scaling_cls: ScalerFactory) -> ScalerFactory:
		# __init__ itself stays FrameT-typed (like every other method here) so
		# its own internal scaler.scale(df) calls keep resolving fine - only
		# this wrapper closure needs to satisfy ScalerFactory's AnyFrame
		# contract, so the cast() lives here instead.
		def _init(df: AnyFrame, raw_column_name: str, feature_column_name: str) -> "ChainedScaler":
			return cls(cast(pl.DataFrame, df), raw_column_name, feature_column_name, *scaling_cls)
		return _init

	def scale(self, df: FrameT, over: Optional[Union[str, list[str]]] = None, with_t0: Optional[bool] = None) -> FrameT:
		if with_t0 is None:
			with_t0 = "t0" in frame_columns(df)
		for scaler in self.scaling[:-1]:
			df = scaler.scale(df, over=over, with_t0=with_t0).rename({self.feature_column_name: self.raw_column_name})

		return self.scaling[-1].scale(df, over=over, with_t0=with_t0)

	def _join_keys(self, over: Optional[Union[str, list[str]]]) -> list[Union[pl.Expr, str]]:
		return ["t0"] if over is None else ["t0", over] if isinstance(over, str) else ["t0", *over]

	@staticmethod
	def _safe_rename(df: FrameT, old: str, new: str) -> FrameT:
		if old == new:
			return df
		return df.with_columns(pl.col(old).alias(new)).drop(old)

	def _needs_t0_anchor_fix(self) -> bool:
		# A later scaler's inverse (e.g. DiffScaler's cumsum) needs its "t0" anchor
		# in the domain it was scaled in. If a ZScoreScaler runs earlier in the
		# chain, the anchor arrives in raw units, so it must be re-scaled into that
		# domain before inverting and restored to raw units afterwards.
		if not any(isinstance(scaler, ZScoreScaler) for scaler in self.scaling[:-1]):
			return False
		if not isinstance(self.scaling[0], ZScoreScaler):
			raise Exception("ZScoreScaler not at the beginning of the chain or at the end of the chain is not implemented")
		return True

	def _rescale_t0_anchor(self, df: FrameT, df_t0: FrameT, over: Optional[Union[str, list[str]]]) -> FrameT:
		# "t0" is null on every row except the anchor row itself (the
		# caller's job - see InferenceModel._unscale_one_step, which groups
		# by a separate id rather than repurposing "t0" itself), so a plain
		# null check identifies it even when several different anchors are
		# batched into one call: `over` already scopes this join's
		# "t0_scaled" to each row's own group/anchor.
		t0_scaled = self.scaling[0].scale(self._safe_rename(df_t0, self.feature_column_name, self.raw_column_name))
		# keep `over` (when it's a plain column, not an arbitrary Expr)
		# alongside "t0" - the join below needs it, and .select() otherwise
		# drops everything but "t0"/the value column.
		join_select: list[Union[str, pl.Expr]] = [pl.col("t0"), pl.col(self.feature_column_name).alias("t0_scaled")]
		if over is not None:
			if isinstance(over, str):
				over = [over]
			join_select.extend(over)

		# cast(): t0_scaled is FrameT (self.scaling[0].scale() ties it back to
		# df_t0's own concrete type just fine), but pyright loses that tie
		# again through the chained .select() call on it - a quirk of
		# resolving polars' .select() overloads against a TypeVar solved via
		# an earlier call rather than a plain parameter. cast() restores it
		# for the join below; see _restore_t0_anchor for the same pattern.
		t0_scaled_for_join = cast(FrameT, t0_scaled.select(join_select))
		return (
			df
			.join(
				t0_scaled_for_join,
				on=self._join_keys(over),
				how="left"
			)
			.with_columns(
				pl.when(pl.col("t0").is_null()).then(self.column).otherwise(pl.col("t0_scaled")).alias(self.feature_column_name)
			)
			.drop("t0_scaled")
		)

	def _restore_t0_anchor(self, df_scaled: FrameT, df_t0: FrameT, over: Optional[Union[str, list[str]]]) -> FrameT:
		old_for_join = cast(FrameT, self._safe_rename(df_t0, self.feature_column_name, "old").drop("time"))
		return (
			df_scaled
			.join(
				old_for_join,
				on=self._join_keys(over),
				how="left"
			)
			.with_columns(
				pl.when(pl.col("t0").is_null()).then(self.raw_column).otherwise("old").alias(self.raw_column_name)
			)
			.drop("old")
		)

	def inverse_scale(self, df: FrameT, over: Optional[Union[str, list[str]]] = None) -> FrameT:
		needs_t0_fix = self._needs_t0_anchor_fix()
		df_t0: Optional[FrameT] = None

		if over is not None and len(over) == 0:
			over = None

		if needs_t0_fix:
			# Grouped by `over` (when given) rather than a plain global max, so
			# a single call inverse-scaling several different t0's at once
			# (each its own group) finds each group's own anchor row instead
			# of every group being compared against the frame-wide max t0.
			anchor_time = pl.col("t0").max() if over is None else pl.col("t0").max().over(over)
			df_t0 = df.filter(TIME_CL == anchor_time)
			df_scaled = self._rescale_t0_anchor(df, df_t0, over)
		else:
			df_scaled = df

		for scaler in reversed(self.scaling[1:]):
			df_scaled = scaler.inverse_scale(df_scaled, over=over).rename({self.raw_column_name: self.feature_column_name})

		df_scaled = self.scaling[0].inverse_scale(df_scaled, over=over)

		if needs_t0_fix:
			assert df_t0 is not None
			df_scaled = self._restore_t0_anchor(df_scaled, df_t0, over)

		return df_scaled
