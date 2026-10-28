# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Row-level CSV Sources that keep one exact record per retrieved citation.

A generic text capture cannot point a citation back at one table record: the
whole CSV is ambiguous to quote and a row number moves as soon as the file is
sorted again. This module makes the record itself the Source, so the identity of
an evidence item is the document plus the primary-key value that selected it.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Annotated
from urllib.parse import quote

from pydantic import BaseModel, Field, field_validator

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

CSV_ROW_SOURCE_NAME = "csv-row"
CSV_ROW_IDENTITY_PREFIX = "csv"
"""Prefix of every row identity, matching ``csv:<document>:<key>``."""

_UTF8_BOM = b"\xef\xbb\xbf"
_JSON_SEPARATORS = (",", ":")


class InvalidCSVRowError(ValueError):
    """A CSV document or row identity that cannot be captured as evidence."""


def _validate_document_id(document_id: str) -> str:
    if not document_id or not document_id.strip():
        raise InvalidCSVRowError("document_id must be a non-empty string")  # noqa: TRY003
    if document_id != document_id.strip():
        raise InvalidCSVRowError("document_id must not contain leading or trailing whitespace")  # noqa: TRY003
    return document_id


def _validate_primary_key(primary_key: str) -> str:
    if not primary_key:
        raise InvalidCSVRowError("primary_key must be a non-empty string")  # noqa: TRY003
    return primary_key


def _validate_primary_key_value(value: str) -> str:
    if not value.strip():
        raise InvalidCSVRowError("primary key value must not be empty or blank")  # noqa: TRY003
    return value


class CSVDocumentInput(BaseModel):
    """One offline CSV document to be captured row by row."""

    document_id: str
    primary_key: str
    content: bytes

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        return _validate_document_id(value)

    @field_validator("primary_key")
    @classmethod
    def validate_primary_key(cls, value: str) -> str:
        return _validate_primary_key(value)


class CSVRowInput(BaseModel):
    """One parsed record: the header order plus the exact field strings."""

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        return _validate_document_id(value)

    @field_validator("primary_key")
    @classmethod
    def validate_primary_key(cls, value: str) -> str:
        return _validate_primary_key(value)

    @field_validator("primary_key_value")
    @classmethod
    def validate_primary_key_value(cls, value: str) -> str:
        return _validate_primary_key_value(value)


class CSVRowSource(Source):
    """Captured record whose identity is derived, never positional."""

    source_id: Annotated[str, Field(min_length=1, max_length=MAX_SOURCE_ID_LENGTH)]
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        return _validate_document_id(value)

    @field_validator("primary_key")
    @classmethod
    def validate_primary_key(cls, value: str) -> str:
        return _validate_primary_key(value)

    @field_validator("primary_key_value")
    @classmethod
    def validate_primary_key_value(cls, value: str) -> str:
        return _validate_primary_key_value(value)


def csv_row_source_id(document_id: str, primary_key_value: str) -> str:
    """Return the stable, delimiter-safe identity of one captured record."""

    _validate_document_id(document_id)
    _validate_primary_key_value(primary_key_value)
    encoded_document = quote_component(document_id)
    encoded_key = quote_component(primary_key_value)
    source_id = f"{CSV_ROW_IDENTITY_PREFIX}:{encoded_document}:{encoded_key}"
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        raise InvalidCSVRowError(f"csv row identity must not exceed {MAX_SOURCE_ID_LENGTH} characters")  # noqa: TRY003
    return source_id


def quote_component(component: str) -> str:
    """Percent-encode one identity component so ``:`` inside it cannot collide."""

    return quote(component, safe="")


def parse_csv_document(value: CSVDocumentInput, /) -> tuple[CSVRowInput, ...]:
    """Parse and validate a whole CSV document, or fail without returning rows."""

    document_id = _validate_document_id(value.document_id)
    primary_key = _validate_primary_key(value.primary_key)
    records = _read_records(value.content)
    if not records:
        raise InvalidCSVRowError("CSV document must contain a header record")  # noqa: TRY003
    header = records[0]
    _validate_header(header, primary_key)
    key_index = header.index(primary_key)
    rows: list[CSVRowInput] = []
    seen_keys: set[str] = set()
    for record in records[1:]:
        if not record:
            continue
        row = _build_row(document_id, primary_key, tuple(header), key_index, record)
        if row.primary_key_value in seen_keys:
            raise InvalidCSVRowError("primary key values must be unique within one CSV document")  # noqa: TRY003
        seen_keys.add(row.primary_key_value)
        rows.append(row)
    return tuple(rows)


def _read_records(content: bytes) -> list[list[str]]:
    if content.startswith(_UTF8_BOM):
        content = content[len(_UTF8_BOM) :]
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise InvalidCSVRowError("CSV document must be valid UTF-8") from error  # noqa: TRY003
    stream = io.StringIO(text, newline="")
    try:
        return list(csv.reader(stream, strict=True))
    except csv.Error as error:
        raise InvalidCSVRowError("CSV document violates the quoting rules") from error  # noqa: TRY003
    finally:
        stream.close()


def _validate_header(header: list[str], primary_key: str) -> None:
    if not header:
        raise InvalidCSVRowError("CSV header must not be empty")  # noqa: TRY003
    for field in header:
        if not field.strip():
            raise InvalidCSVRowError("CSV header fields must not be empty or blank")  # noqa: TRY003
    if len(set(header)) != len(header):
        raise InvalidCSVRowError("CSV header fields must be unique")  # noqa: TRY003
    if primary_key not in header:
        raise InvalidCSVRowError("primary_key must match a CSV header field exactly")  # noqa: TRY003


def _build_row(
    document_id: str,
    primary_key: str,
    columns: tuple[str, ...],
    key_index: int,
    record: list[str],
) -> CSVRowInput:
    if len(record) != len(columns):
        raise InvalidCSVRowError("CSV record width must equal the header width")  # noqa: TRY003
    return CSVRowInput(
        document_id=document_id,
        primary_key=primary_key,
        primary_key_value=record[key_index],
        columns=columns,
        values=tuple(record),
    )


class CSVRowSourceAdapter:
    """Resolve and read row Sources without touching a catalog."""

    input_class = CSVRowInput
    name = CSV_ROW_SOURCE_NAME
    source_class = CSVRowSource

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        source_id = csv_row_source_id(value.document_id, value.primary_key_value)
        return CSVRowSource(
            name=source_id,
            materialization=SourceMaterialization.CAPTURED,
            source_id=source_id,
            document_id=value.document_id,
            primary_key=value.primary_key,
            primary_key_value=value.primary_key_value,
            columns=value.columns,
            values=value.values,
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=source.columns,
            values=source.values,
        )


class CSVRowTextEvidenceProjection:
    """Expose one record as canonical text evidence without type conversion."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = CSVRowSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        row = dict(zip(source.columns, source.values, strict=True))
        return TextEvidence(
            source_type=CSV_ROW_SOURCE_NAME,
            source_id=source.source_id,
            content=json.dumps(row, ensure_ascii=False, separators=_JSON_SEPARATORS),
            metadata={
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": row,
            },
        )


CSV_ROW_SOURCE_ADAPTER = CSVRowSourceAdapter()
CSV_ROW_SOURCE_DEFINITION = AdapterSourceDefinition(
    CSV_ROW_SOURCE_ADAPTER,
    projections=(CSVRowTextEvidenceProjection(),),
)


__all__ = [
    "CSV_ROW_SOURCE_DEFINITION",
    "CSV_ROW_SOURCE_NAME",
    "CSVDocumentInput",
    "CSVRowInput",
    "CSVRowSource",
    "InvalidCSVRowError",
    "csv_row_source_id",
    "parse_csv_document",
]
