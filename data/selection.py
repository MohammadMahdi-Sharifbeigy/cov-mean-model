"""Select task-design columns without relying on substring matching."""

from collections.abc import Mapping, Sequence
from typing import Any, NoReturn

import numpy as np


def select_task_variables(
    prep: Mapping[str, Any], names: Sequence[str] | None
) -> tuple[np.ndarray, np.ndarray]:
    """Return dense and sparse task matrices restricted to ``names``.

    ``dense_indices`` and ``sparse_indices`` identify columns in the complete
    task-variable list, while the matrices are already type-filtered.  The
    selector therefore translates complete-list indexes into each matrix's
    local column positions before slicing.
    """
    dense = prep["X_dense_task"]
    sparse = prep["X_sparse_task"]
    if names is None:
        return dense, sparse

    requested_names = list(names)
    if len(set(requested_names)) != len(requested_names):
        raise ValueError("Task-variable requests must be unique")

    task_index_by_name = {name: index for index, name in enumerate(prep["task_var_names"])}
    requested_indexes = [
        task_index_by_name[name] if name in task_index_by_name else _missing(name)
        for name in requested_names
    ]
    dense_columns_by_task_index = {
        task_index: column for column, task_index in enumerate(prep["dense_indices"])
    }
    sparse_columns_by_task_index = {
        task_index: column for column, task_index in enumerate(prep["sparse_indices"])
    }

    dense_columns = [
        dense_columns_by_task_index[index]
        for index in requested_indexes
        if index in dense_columns_by_task_index
    ]
    sparse_columns = [
        sparse_columns_by_task_index[index]
        for index in requested_indexes
        if index in sparse_columns_by_task_index
    ]
    return dense[:, dense_columns], sparse[:, sparse_columns]


def _missing(name: str) -> NoReturn:
    raise KeyError(f"Unknown task variable: {name}")
