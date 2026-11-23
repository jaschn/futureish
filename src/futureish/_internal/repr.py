#type: ignore
"""Shared __repr__/_repr_html_ machinery for every gpr_ts class that declares
a `_repr_fields` property (dict[str, Any] of the fields worth showing).

ReprMixin walks that dict and recurses into any value that itself has
`_repr_fields` (or a list of such values) - so e.g. Forecaster's repr walks
down through its DataConfig into every InputFeature/TargetFeature/Scaler
without each of those needing its own __repr__/_repr_html_ implementation.
"""

import html
from typing import Any, Optional

_MAX_LIST_ITEMS = 10
_MAX_DEPTH = 8


def _repr_fields_of(value: Any) -> Optional[dict[str, Any]]:
	"""`value._repr_fields` if it's a dict (i.e. a proper `@property`), else
	None - guards against a class that (by bug) defines `_repr_fields` as a
	plain method, where accessing it without calling would hand back a bound
	method instead of a dict.
	"""
	fields: Optional[dict[str, Any]] = getattr(value, "_repr_fields", None)
	return fields if isinstance(fields, dict) else None


def _is_array_like(value: Any) -> bool:
	# duck-typed rather than isinstance-checked against numpy/torch/polars
	# directly - all three happen to be available here (gpytorch pulls in
	# numpy/torch, polars is a direct dependency), but this stays correct for
	# any other array-ish type (e.g. pandas) a future _repr_fields might hold.
	return hasattr(value, "shape") and not _repr_fields_of(value)


def _array_summary(value: Any) -> str:
	dtype = getattr(value, "dtype", None)
	shape = tuple(getattr(value, "shape", ()))
	dtype_str = f", dtype={dtype}" if dtype is not None else ""
	return f"{type(value).__name__}(shape={shape}{dtype_str})"


def _callable_summary(value: Any) -> str:
	name = getattr(value, "__qualname__", None) or getattr(value, "__name__", None) or repr(value)
	return f"<{name}>"


def _leaf_repr(value: Any) -> str:
	if _is_array_like(value):
		return _array_summary(value)
	if callable(value) and not isinstance(value, type):
		return _callable_summary(value)
	if isinstance(value, type):
		return value.__name__
	return repr(value)


def _indent(text: str, prefix: str = "\t") -> str:
	return "\n".join(prefix + line for line in text.splitlines())


def _attach_key(key: str, formatted: str) -> str:
	"""Render `key=formatted`, keeping a multi-line `formatted`'s opening
	bracket on the key's own line (`key=[` / `key=ClassName(`) instead of
	dropping it to its own line - matches how e.g. `black` formats nested
	literals, rather than one extra level of indentation for no reason.
	"""
	if "\n" not in formatted:
		return f"{key}={formatted}"
	first, *rest = formatted.splitlines()
	return "\n".join([f"{key}={first}", *rest])


def _format_value(value: Any, seen: set[int], depth: int) -> str:
	if depth >= _MAX_DEPTH:
		return _leaf_repr(value)

	fields = _repr_fields_of(value)
	if fields is not None:
		return _format_object(value, fields, seen, depth)

	if isinstance(value, (list, tuple)):
		items = list(value)
		if not items:
			return "[]"
		if not any(_repr_fields_of(item) is not None for item in items):
			return repr(value)
		return _format_list(items, seen, depth)

	return _leaf_repr(value)


def _format_object(value: Any, fields: dict[str, Any], seen: set[int], depth: int) -> str:
	class_name = value.__class__.__name__
	obj_id = id(value)
	if obj_id in seen:
		return f"{class_name}(...)"
	seen = seen | {obj_id}

	if not fields:
		return f"{class_name}()"

	lines: list[str] = []
	for key, field_value in fields.items():
		formatted = _format_value(field_value, seen, depth + 1)
		lines.append(_attach_key(key, formatted))

	body = ",\n".join(lines)
	return f"{class_name}(\n{_indent(body)}\n)"


def _format_list(items: list[Any], seen: set[int], depth: int) -> str:
	shown = items[:_MAX_LIST_ITEMS]
	lines = [_format_value(item, seen, depth + 1) for item in shown]
	if len(items) > _MAX_LIST_ITEMS:
		lines.append(f"... ({len(items) - _MAX_LIST_ITEMS} more)")
	body = ",\n".join(_indent(line) for line in lines)
	return f"[\n{body}\n]"


class ReprMixin:
	@property
	def _repr_fields(self) -> dict[str, Any]:
		return {}

	def __repr__(self) -> str:
		fields = _repr_fields_of(self)
		return _format_object(self, fields or {}, set(), 0)

	def _repr_html_(self) -> str:
		fields = _repr_fields_of(self)
		return _html_root(self, fields or {})


# --------------------------------------------------------------------------
# HTML rendering
# --------------------------------------------------------------------------

_CSS = """
<style>
.gpr-ts-repr { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; font-size: 13px; color: inherit; }
.gpr-ts-repr details { margin: 2px 0; }
.gpr-ts-repr summary { cursor: pointer; font-weight: 600; padding: 2px 0; }
.gpr-ts-repr summary:hover { color: #2563eb; }
.gpr-ts-repr table { border-collapse: collapse; margin: 4px 0 4px 1.2em; }
.gpr-ts-repr td, .gpr-ts-repr th { border: 1px solid rgba(128, 128, 128, 0.35); padding: 3px 8px; text-align: left; vertical-align: top; }
.gpr-ts-repr th { background: rgba(128, 128, 128, 0.12); font-weight: 600; }
.gpr-ts-repr .gpr-ts-field { font-weight: 600; white-space: nowrap; }
.gpr-ts-repr .gpr-ts-scalar { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.gpr-ts-repr .gpr-ts-empty { color: rgba(128, 128, 128, 0.8); font-style: italic; }
</style>
"""


def _esc(text: Any) -> str:
	return html.escape(str(text))


def _html_root(value: Any, fields: dict[str, Any]) -> str:
	body = _html_object(value, fields, set(), depth=0, open_=True)
	return f'{_CSS}<div class="gpr-ts-repr">{body}</div>'


def _html_value(value: Any, seen: set[int], depth: int) -> str:
	if depth >= _MAX_DEPTH:
		return f'<span class="gpr-ts-scalar">{_esc(_leaf_repr(value))}</span>'

	fields = _repr_fields_of(value)
	if fields is not None:
		return _html_object(value, fields, seen, depth)

	if isinstance(value, (list, tuple)):
		items = list(value)
		if items and any(_repr_fields_of(item) is not None for item in items):
			return _html_list(items, seen, depth)

	return f'<span class="gpr-ts-scalar">{_esc(_leaf_repr(value))}</span>'


def _html_object(value: Any, fields: dict[str, Any], seen: set[int], depth: int, open_: bool = False) -> str:
	"""One object as a small horizontal table: field names as column
	headers, its own values as the single row underneath - reads like a
	single-row dataframe rather than a vertical key/value list.
	"""
	class_name = value.__class__.__name__
	obj_id = id(value)
	if obj_id in seen:
		return f'<span class="gpr-ts-scalar">{_esc(class_name)}(...)</span>'
	seen = seen | {obj_id}

	if not fields:
		return f'<span class="gpr-ts-scalar">{_esc(class_name)}()</span>'

	header = "".join(f"<th>{_esc(key)}</th>" for key in fields)
	values = "".join(f"<td>{_html_value(field_value, seen, depth + 1)}</td>" for field_value in fields.values())
	table = f"<table><tr>{header}</tr><tr>{values}</tr></table>"

	open_attr = " open" if open_ else ""
	return f"<details{open_attr}><summary>{_esc(class_name)}</summary>{table}</details>"


_MISSING = object()


def _html_list(items: list[Any], seen: set[int], depth: int) -> str:
	shown = items[:_MAX_LIST_ITEMS]
	remaining = len(items) - len(shown)
	summary = f"{len(items)} item{'s' if len(items) != 1 else ''}"

	field_dicts = [_repr_fields_of(item) for item in shown]
	if all(f is not None for f in field_dicts):
		table = _html_uniform_list_table(shown, field_dicts, seen, depth, remaining)
	else:
		table = _html_mixed_list_table(shown, seen, depth, remaining)

	return f"<details><summary>{summary}</summary>{table}</details>"


def _html_uniform_list_table(shown: list[Any], field_dicts: list[dict[str, Any]], seen: set[int], depth: int, remaining: int) -> str:
	"""Every item has its own `_repr_fields` - render as one dataframe-like
	table, columns as the (unioned, order-preserving) field names and one
	row per item, rather than a nested `<details>` per item.
	"""
	keys: list[str] = []
	for fields in field_dicts:
		for key in fields:
			if key not in keys:
				keys.append(key)

	header = "".join(f"<th>{_esc(key)}</th>" for key in ["#", *keys])
	rows: list[str] = []
	for i, fields in enumerate(field_dicts):
		cells = [f"<td>{i}</td>"]
		for key in keys:
			field_value = fields.get(key, _MISSING)
			cell = "" if field_value is _MISSING else _html_value(field_value, seen, depth + 1)
			cells.append(f"<td>{cell}</td>")
		rows.append("<tr>" + "".join(cells) + "</tr>")

	if remaining > 0:
		rows.append(f'<tr><td colspan="{len(keys) + 1}" class="gpr-ts-empty">... ({remaining} more)</td></tr>')

	return f"<table><tr>{header}</tr>{''.join(rows)}</table>"


def _html_mixed_list_table(shown: list[Any], seen: set[int], depth: int, remaining: int) -> str:
	rows: list[str] = []
	for i, item in enumerate(shown):
		rows.append(
			"<tr>"
			f'<td class="gpr-ts-field">[{i}]</td>'
			f"<td>{_html_value(item, seen, depth + 1)}</td>"
			"</tr>"
		)
	if remaining > 0:
		rows.append(f'<tr><td colspan="2" class="gpr-ts-empty">... ({remaining} more)</td></tr>')
	return f"<table>{''.join(rows)}</table>"
