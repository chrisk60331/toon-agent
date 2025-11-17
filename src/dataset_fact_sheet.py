from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

MAX_FIELD_LINES = 8
MAX_TOP_LEVEL_KEYS = 12
MAX_COMMON_VALUES = 8
MAX_STRING_LENGTH = 80
SEQUENCE_SAMPLE_LIMIT = 200
MAX_NESTED_COLLECTIONS = 2
STRING_UNIQUE_TRACK_LIMIT = 512


def _type_name(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return "array"
    return type(value).__name__


def _is_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float))


def _try_parse_number(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        try:
            return float(candidate)
        except ValueError:
            return None
    return None


def _format_value(value: object) -> str:
    text = ""
    if isinstance(value, str):
        text = value.strip()
    elif isinstance(value, Mapping):
        text = "{...}"
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        text = "[...]"
    else:
        text = str(value)

    if len(text) > MAX_STRING_LENGTH:
        return text[: MAX_STRING_LENGTH - 3] + "..."
    return text


def _normalize_token(token: str) -> str:
    cleaned = token.strip()
    if ":" in cleaned:
        cleaned = cleaned.split(":", 1)[1]
    cleaned = cleaned.replace("_", " ").replace("-", " ").strip()
    if not cleaned:
        return ""
    if any(char.isupper() for char in cleaned[1:]):
        return cleaned
    return cleaned.title()


def _join_values(values: Sequence[str]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    return ", ".join(values[:-1]) + f", and {values[-1]}"


def _compose_name(record: Mapping[str, Any]) -> str | None:
    firstname = record.get("firstname") or record.get("first_name") or record.get("givenname")
    surname = record.get("surname") or record.get("last_name")
    name_field = record.get("name")

    def _clean(value: object) -> str | None:
        if isinstance(value, str):
            stripped = value.strip()
            return " ".join(stripped.split()) if stripped else None
        return None

    first_clean = _clean(firstname)
    last_clean = _clean(surname)
    name_clean = _clean(name_field)

    if first_clean and last_clean:
        return f"{first_clean} {last_clean}"
    if first_clean:
        return first_clean
    if last_clean:
        return last_clean
    return name_clean


@dataclass
class FieldTracker:
    total: int = 0
    non_null: int = 0
    types: Counter[str] = field(default_factory=Counter)
    numeric_min: float | None = None
    numeric_max: float | None = None
    numeric_count: int = 0
    string_values: Counter[str] = field(default_factory=Counter)
    string_overflow: bool = False
    string_unique_seen: set[str] = field(default_factory=set)
    nested_sequence_lengths: list[int] = field(default_factory=list)
    nested_sequence_item_types: Counter[str] = field(default_factory=Counter)
    nested_mapping_keys: Counter[str] = field(default_factory=Counter)

    def update(self, value: object) -> None:
        self.total += 1
        value_type = _type_name(value)
        self.types[value_type] += 1

        if value is None:
            return

        self.non_null += 1

        numeric_value = _try_parse_number(value)
        if numeric_value is not None:
            self.numeric_count += 1
            if self.numeric_min is None or numeric_value < self.numeric_min:
                self.numeric_min = numeric_value
            if self.numeric_max is None or numeric_value > self.numeric_max:
                self.numeric_max = numeric_value

        if isinstance(value, str):
            if len(self.string_unique_seen) < STRING_UNIQUE_TRACK_LIMIT:
                self.string_unique_seen.add(value)
            if not self.string_overflow:
                if len(self.string_values) >= MAX_COMMON_VALUES and value not in self.string_values:
                    self.string_overflow = True
                    self.string_values.clear()
                else:
                    self.string_values[value] += 1

        if isinstance(value, Mapping):
            for key in value.keys():
                self.nested_mapping_keys[_normalize_token(str(key)) or str(key)] += 1
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            seq = list(value)
            self.nested_sequence_lengths.append(len(seq))
            for item in seq[:MAX_COMMON_VALUES]:
                self.nested_sequence_item_types[_type_name(item)] += 1


@dataclass
class CollectionSummary:
    name: str
    count: int
    field_trackers: dict[str, FieldTracker]

    def render(self) -> list[str]:
        lines: list[str] = []
        lines.append(f"- Records: {self.count}")
        if not self.field_trackers:
            return lines

        sorted_fields = sorted(
            self.field_trackers.items(),
            key=lambda item: (-item[1].non_null, item[0]),
        )[:MAX_FIELD_LINES]

        lines.append("- Field highlights:")
        for field_name, tracker in sorted_fields:
            coverage = tracker.non_null / tracker.total if tracker.total else 0.0
            type_summary = "/".join(t for t, _ in tracker.types.most_common())
            pretty_name = _normalize_token(field_name) or field_name
            coverage_text = f"{coverage * 100:.0f}%"

            bullet_parts: list[str] = [f"{pretty_name}: {type_summary}, {coverage_text} coverage"]

            if tracker.numeric_count:
                min_val = tracker.numeric_min
                max_val = tracker.numeric_max
                if min_val is not None and max_val is not None and min_val != max_val:
                    min_str = f"{min_val:.2f}".rstrip("0").rstrip(".")
                    max_str = f"{max_val:.2f}".rstrip("0").rstrip(".")
                    bullet_parts.append(f"range {min_str}–{max_str}")

            display_values: list[str] = []
            if tracker.string_values:
                common = sorted(
                    tracker.string_values.items(),
                    key=lambda item: (-item[1], item[0]),
                )[:MAX_COMMON_VALUES]
                display_values = [_normalize_token(value) or value for value, _ in common]
                if display_values:
                    bullet_parts.append(f"common values: {_join_values(display_values)}")

            unique_count = len(tracker.string_unique_seen)
            if unique_count:
                if tracker.string_overflow:
                    bullet_parts.append(f"at least {unique_count} unique values")
                elif unique_count > len(display_values):
                    bullet_parts.append(f"{unique_count} unique values")

            if tracker.nested_mapping_keys:
                top_nested_keys = [
                    key for key, _ in tracker.nested_mapping_keys.most_common(MAX_COMMON_VALUES)
                ]
                nested_keys = _join_values(top_nested_keys)
                if nested_keys:
                    bullet_parts.append(f"nested keys: {nested_keys}")

            if tracker.nested_sequence_lengths:
                lengths = tracker.nested_sequence_lengths
                avg_len = sum(lengths) / len(lengths)
                avg_str = f"{avg_len:.2f}".rstrip("0").rstrip(".")
                item_types = ", ".join(
                    t for t, _ in tracker.nested_sequence_item_types.most_common(MAX_COMMON_VALUES)
                )
                nested_summary = f"nested array avg len {avg_str}"
                if item_types:
                    nested_summary += f" (item types: {item_types})"
                multi_count = sum(1 for length in lengths if length > 1)
                if multi_count:
                    pct = (multi_count / len(lengths)) * 100
                    nested_summary += f"; >1 items in {pct:.0f}% of parent records"
                bullet_parts.append(nested_summary)

            bullet = "; ".join(bullet_parts)
            lines.append(f"  • {bullet}")

        return lines


class FactSheetBuilder:
    def __init__(self, payload: object, file_name: str) -> None:
        self.payload = payload
        self.file_name = file_name

    def build(self) -> str:
        lines: list[str] = []
        lines.append(f"FACT SHEET FOR {self.file_name}")
        self._append_summary(self.payload, lines, indent_level=0, name="root")
        return "\n".join(lines)

    def _append_summary(self, value: object, lines: list[str], indent_level: int, name: str | None) -> None:
        indent = "  " * indent_level
        value_type = _type_name(value)
        header = f"{indent}- {name or 'value'}: {value_type}"
        if isinstance(value, Mapping):
            header += f" with {len(value)} keys"
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            header += f" ({len(value)} items)"
        lines.append(header)

        if isinstance(value, Mapping):
            self._summarize_mapping(value, lines, indent_level + 1)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            self._summarize_sequence(name or "collection", value, lines, indent_level + 1)

    def _summarize_mapping(self, payload: Mapping[str, Any], lines: list[str], indent_level: int) -> None:
        keys = list(payload.keys())
        indent = "  " * indent_level
        if keys:
            display_keys = keys[:MAX_TOP_LEVEL_KEYS]
            keys_text = ", ".join(display_keys)
            lines.append(f"{indent}- keys: {keys_text}")
            if len(keys) > MAX_TOP_LEVEL_KEYS:
                lines.append(f"{indent}- (truncated list of keys)")

        for key in keys[:MAX_TOP_LEVEL_KEYS]:
            value = payload[key]
            value_type = _type_name(value)
            if value_type in {"object", "array"}:
                self._append_summary(value, lines, indent_level, key)
            else:
                summary = _format_value(value)
                lines.append(f"{indent}- {key}: {summary} ({value_type})")

    def _summarize_sequence(
        self,
        name: str,
        payload: Sequence[Any],
        lines: list[str],
        indent_level: int,
    ) -> None:
        indent = "  " * indent_level
        total_items = len(payload)
        lines.append(f"{indent}- items analysed: {total_items} of {total_items}")
        if not payload:
            return

        first_item = payload[0]
        if isinstance(first_item, Mapping):
            trackers: dict[str, FieldTracker] = {}
            nested_trackers_map: dict[str, dict[str, FieldTracker]] = {}
            nested_totals: dict[str, int] = {}
            gender_counts: Counter[str] = Counter()
            org_names: Counter[str] = Counter()
            country_set: set[str] = set()
            multi_prize_names: Counter[str] = Counter()
            category_start_years: dict[str, int] = {}
            category_display_names: dict[str, str] = {}

            for entry in payload:
                if not isinstance(entry, Mapping):
                    continue
                lower_entry = {str(k).lower(): v for k, v in entry.items()}
                for field_name, field_value in entry.items():
                    field_key = str(field_name)
                    tracker = trackers.setdefault(field_key, FieldTracker())
                    tracker.update(field_value)

                    if isinstance(field_value, Sequence) and not isinstance(field_value, (str, bytes, bytearray)):
                        seq_list = list(field_value)
                        nested_totals[field_key] = nested_totals.get(field_key, 0) + len(seq_list)
                        if any(isinstance(item, Mapping) for item in seq_list):
                            nested_trackers = nested_trackers_map.setdefault(field_key, {})
                            for nested_entry in seq_list:
                                if not isinstance(nested_entry, Mapping):
                                    continue
                                lower_nested = {str(k).lower(): v for k, v in nested_entry.items()}
                                for nested_key, nested_value in nested_entry.items():
                                    nested_tracker = nested_trackers.setdefault(str(nested_key), FieldTracker())
                                    nested_tracker.update(nested_value)
                                category_value = lower_nested.get("category")
                                year_value = lower_nested.get("year")
                                year_number = _try_parse_number(year_value)
                                if isinstance(category_value, str) and year_number is not None:
                                    category_clean = category_value.strip()
                                    if category_clean:
                                        key = category_clean.lower()
                                        category_display_names.setdefault(key, category_clean)
                                        year_int = int(year_number)
                                        current = category_start_years.get(key)
                                        if current is None or year_int < current:
                                            category_start_years[key] = year_int

                gender_value = lower_entry.get("gender")
                if isinstance(gender_value, str):
                    gender_clean = gender_value.strip().lower()
                    if gender_clean:
                        gender_counts[gender_clean] += 1
                        if gender_clean == "org":
                            name = _compose_name(lower_entry)
                            if name:
                                org_names[name] += 1

                born_country = lower_entry.get("borncountry")
                if isinstance(born_country, str):
                    cleaned_country = born_country.strip()
                    if cleaned_country:
                        country_set.add(cleaned_country)

                prizes_value = lower_entry.get("prizes")
                if isinstance(prizes_value, Sequence) and not isinstance(prizes_value, (str, bytes, bytearray)):
                    if len(prizes_value) > 1:
                        name = _compose_name(lower_entry)
                        if name:
                            multi_prize_names[name] += 1

            collection_summary = CollectionSummary(name=name, count=total_items, field_trackers=trackers)
            for line in collection_summary.render():
                lines.append(indent + line if line.startswith("-") else indent + line)

            nested_fields = list(nested_trackers_map.items())[:MAX_NESTED_COLLECTIONS]
            for nested_field, nested_trackers in nested_fields:
                nested_summary = CollectionSummary(
                    name=f"{name}.{nested_field}",
                    count=nested_totals.get(nested_field, 0),
                    field_trackers=nested_trackers,
                )
                lines.append(f"{indent}- Nested collection '{nested_field}':")
                for nested_line in nested_summary.render():
                    lines.append(
                        f"{indent}  {nested_line}" if nested_line.startswith("-") else f"{indent}  {nested_line}"
                    )

            derived_insights: list[str] = []
            total_genders = sum(gender_counts.values())
            female_count = gender_counts.get("female", 0)
            org_count = gender_counts.get("org", 0)
            if female_count and total_genders:
                female_pct = (female_count / total_genders) * 100
                derived_insights.append(
                    f"Female laureates: {female_count}/{total_genders} (~{female_pct:.1f}%)"
                )
            if org_count:
                top_orgs = [name for name, _ in org_names.most_common(3)]
                if top_orgs:
                    derived_insights.append(f"Organizational recipients include {', '.join(top_orgs)}")
            if country_set:
                derived_insights.append(f"Birth countries represented: {len(country_set)}+ unique values")
            if multi_prize_names:
                top_multi = [name for name, _ in multi_prize_names.most_common(3)]
                derived_insights.append(f"Multi-prize laureates include {', '.join(top_multi)}")
            if category_start_years:
                ordered_categories = sorted(
                    category_start_years.items(),
                    key=lambda item: (item[1], item[0]),
                )
                components = []
                for cat, year in ordered_categories:
                    display_name = category_display_names.get(cat, cat.title())
                    if year <= 1901:
                        components.append(f"{display_name} (annual, started {year})")
                    else:
                        components.append(f"{display_name} (started {year}, added as official category)")
                derived_insights.append(f"Prize category launch years: {', '.join(components)}")

            if derived_insights:
                lines.append(f"{indent}- Derived insights:")
                for insight in derived_insights[:MAX_FIELD_LINES]:
                    lines.append(f"{indent}  • {insight}")
        else:
            sample_values = [_format_value(item) for item in payload[:MAX_COMMON_VALUES]]
            lines.append(f"{indent}- sample values: {_join_values(sample_values)}")


def build_fact_sheet(payload: object, file_name: str) -> str:
    builder = FactSheetBuilder(payload=payload, file_name=file_name)
    return builder.build()

