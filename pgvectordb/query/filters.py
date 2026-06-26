"""Metadata filter SQL compilation helpers."""

from __future__ import annotations

import re
from typing import Any

from ..base import ValidationError


class MetadataFilterCompiler:
    """Compile Mongo-style metadata filters into SQL and bound parameters."""

    def build(self, filter: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Build a SQL WHERE clause and parameter map from a metadata filter."""
        if not filter:
            return "1=1", {}
        where_clause, params, _ = self.parse(filter, {}, 0)
        return where_clause, params

    def parse(
        self, filter: dict[str, Any], params: dict[str, Any], counter: int
    ) -> tuple[str, dict[str, Any], int]:
        """Recursively parse filter conditions."""
        filter_expressions = []

        for key, condition in filter.items():
            if key == "$and":
                and_clauses = []
                for sub_filter in condition:
                    clause, params, counter = self.parse(sub_filter, params, counter)
                    and_clauses.append(f"({clause})")
                filter_expressions.append(" AND ".join(and_clauses))
                continue

            if key == "$or":
                or_clauses = []
                for sub_filter in condition:
                    clause, params, counter = self.parse(sub_filter, params, counter)
                    or_clauses.append(f"({clause})")
                filter_expressions.append("(" + " OR ".join(or_clauses) + ")")
                continue

            if isinstance(condition, dict):
                for op_key, value in condition.items():
                    expr, params, counter = self.build_single_condition(
                        key, op_key, value, params, counter
                    )
                    filter_expressions.append(expr)
            else:
                expr, params, counter = self.build_single_condition(
                    key, "$eq", condition, params, counter
                )
                filter_expressions.append(expr)

        where_clause = " AND ".join(filter_expressions) if filter_expressions else "1=1"
        return where_clause, params, counter

    def build_single_condition(
        self,
        key: str,
        operator: str,
        value: Any,
        params: dict[str, Any],
        counter: int,
    ) -> tuple[str, dict[str, Any], int]:
        """Build a single filter condition with proper type handling."""
        param_name = f"param_{counter}"
        counter += 1

        if not re.match(r"^[a-zA-Z0-9_]+$", key):
            raise ValidationError(f"Invalid metadata key: '{key}'. Keys must be alphanumeric.")

        is_numeric = isinstance(value, (int, float))
        if operator == "$between" and isinstance(value, (list, tuple)) and len(value) > 0:
            is_numeric = isinstance(value[0], (int, float))

        field_expr = (
            f"(langchain_metadata->>'{key}')::numeric"
            if is_numeric
            else f"langchain_metadata->>'{key}'"
        )

        op_map = {
            "$eq": "=",
            "$ne": "!=",
            "$lt": "<",
            "$lte": "<=",
            "$gt": ">",
            "$gte": ">=",
        }
        if operator in op_map:
            params[param_name] = value
            return f"{field_expr} {op_map[operator]} :{param_name}", params, counter

        if operator == "$in":
            if not isinstance(value, (list, tuple)):
                value = [value]
            if not is_numeric:
                value = [str(item) for item in value]
            params[param_name] = tuple(value)
            return f"{field_expr} = ANY(:{param_name})", params, counter

        if operator == "$nin":
            if not isinstance(value, (list, tuple)):
                value = [value]
            if not is_numeric:
                value = [str(item) for item in value]
            params[param_name] = tuple(value)
            return f"{field_expr} != ALL(:{param_name})", params, counter

        if operator == "$between":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValidationError(f"$between requires a list/tuple of 2 values, got: {value}")
            param_name_2 = f"param_{counter}"
            counter += 1
            params[param_name] = value[0]
            params[param_name_2] = value[1]
            return (
                f"{field_expr} BETWEEN :{param_name} AND :{param_name_2}",
                params,
                counter,
            )

        if operator == "$exists":
            condition = "IS NOT NULL" if value else "IS NULL"
            return f"langchain_metadata->>'{key}' {condition}", params, counter

        if operator == "$like":
            params[param_name] = value
            return f"langchain_metadata->>'{key}' LIKE :{param_name}", params, counter

        if operator == "$ilike":
            params[param_name] = value
            return f"langchain_metadata->>'{key}' ILIKE :{param_name}", params, counter

        raise ValidationError(f"Unsupported operator: {operator}")


def compile_metadata_filter(filter: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Compile a metadata filter with a fresh compiler instance."""
    return MetadataFilterCompiler().build(filter)
