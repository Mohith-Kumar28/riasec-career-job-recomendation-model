from __future__ import annotations

import io
import re
import urllib.request
import http.cookiejar
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


RIASEC_CODE_TO_NAME: dict[str, str] = {
    "R": "Realistic",
    "I": "Investigative",
    "A": "Artistic",
    "S": "Social",
    "E": "Enterprising",
    "C": "Conventional",
}
RIASEC_NAME_TO_CODE: dict[str, str] = {v: k for k, v in RIASEC_CODE_TO_NAME.items()}
VALID_RIASEC_LETTERS: frozenset[str] = frozenset(RIASEC_CODE_TO_NAME.keys())


class InvalidRIASECCode(ValueError):
    pass


def normalize_riasec_code(code: str | None) -> str | None:
    if code is None:
        return None
    raw = code.strip()
    if raw == "":
        return None
    compact = re.sub(r"[^A-Za-z]", "", raw).upper()
    if compact == "":
        raise InvalidRIASECCode(
            f"Invalid RIASEC code {code!r}. Provide 1-6 letters from {''.join(sorted(VALID_RIASEC_LETTERS))}."
        )
    invalid = sorted({ch for ch in compact if ch not in VALID_RIASEC_LETTERS})
    if invalid:
        raise InvalidRIASECCode(
            f"Invalid RIASEC code {code!r}. Invalid letters: {''.join(invalid)}. Allowed: {''.join(sorted(VALID_RIASEC_LETTERS))}."
        )
    if len(compact) > 6:
        raise InvalidRIASECCode(f"Invalid RIASEC code {code!r}. Max length is 6.")
    if len(set(compact)) != len(compact):
        raise InvalidRIASECCode(f"Invalid RIASEC code {code!r}. Letters must be unique.")
    return compact


def _extract_string(value: Any) -> str:
    s = "" if value is None else str(value)
    out = []
    for ch in s:
        if ch.isalpha() or ch == " ":
            out.append(ch.lower())
    return "".join(out)


def build_combined_features(df: pd.DataFrame) -> pd.Series:
    df_te = df.fillna("").replace("-", "", regex=False).astype(str)
    df_te = df_te.apply(lambda c: c.map(_extract_string))
    return df_te.apply(lambda r: " ".join(r.values.tolist()), axis=1)


VectorMethod = Literal["count", "tfidf"]


def text_vector_search(
    df: pd.DataFrame,
    query: str,
    *,
    combined_features: pd.Series | None = None,
    method: VectorMethod = "tfidf",
    top_k: int = 10,
) -> pd.DataFrame:
    if query.strip() == "":
        raise ValueError("Text query must be non-empty.")
    if combined_features is None:
        combined_features = build_combined_features(df)
    cleaned_query = _extract_string(query)
    vectorizer = TfidfVectorizer() if method == "tfidf" else CountVectorizer()
    matrix = vectorizer.fit_transform(combined_features.tolist())
    q = vectorizer.transform([cleaned_query])
    sims = cosine_similarity(q, matrix).ravel()
    top_idx = sims.argsort()[::-1][: max(0, top_k)]
    out = df.iloc[top_idx].copy()
    out.insert(0, "rank", range(1, len(out) + 1))
    out.insert(1, "similarity", sims[top_idx])
    return out.reset_index(drop=True)


@dataclass(frozen=True)
class RIASECMatchWeights:
    first: int = 3
    second: int = 2
    third: int = 1
    exact_first_bonus: int = 6
    exact_second_bonus: int = 3
    exact_third_bonus: int = 1


def _row_letters(row: pd.Series) -> tuple[str | None, str | None, str | None]:
    def to_letter(v: Any) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        if s == "" or s == "-":
            return None
        return RIASEC_NAME_TO_CODE.get(s)

    return (to_letter(row.get("First Personality")), to_letter(row.get("Second Personality")), to_letter(row.get("Third Personality")))


def riasec_code_match(
    df: pd.DataFrame,
    code: str,
    *,
    top_k: int = 10,
    weights: RIASECMatchWeights = RIASECMatchWeights(),
) -> pd.DataFrame:
    normalized = normalize_riasec_code(code)
    if normalized is None:
        raise InvalidRIASECCode("RIASEC code must be non-empty.")

    def score_row(row: pd.Series) -> tuple[int, str]:
        first, second, third = _row_letters(row)
        letters = (first, second, third)
        detail_parts: list[str] = []
        total = 0
        for pos, ch in enumerate(normalized):
            order_weight = len(normalized) - pos
            hit = None
            if letters[0] == ch:
                total += weights.first * order_weight
                hit = "First"
            elif letters[1] == ch:
                total += weights.second * order_weight
                hit = "Second"
            elif letters[2] == ch:
                total += weights.third * order_weight
                hit = "Third"
            if hit is not None:
                detail_parts.append(f"{ch}:{hit}")
        if len(normalized) >= 1 and letters[0] == normalized[0]:
            total += weights.exact_first_bonus
        if len(normalized) >= 2 and letters[1] == normalized[1]:
            total += weights.exact_second_bonus
        if len(normalized) >= 3 and letters[2] == normalized[2]:
            total += weights.exact_third_bonus
        return total, ",".join(detail_parts)

    scores: list[int] = []
    details: list[str] = []
    for _, row in df.iterrows():
        sc, det = score_row(row)
        scores.append(sc)
        details.append(det)

    out = df.copy()
    out.insert(0, "match_details", details)
    out.insert(0, "match_score", scores)
    out = out[out["match_score"] > 0]
    out = out.sort_values(
        by=["match_score", "Occupation", "Job Family"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    out = out.head(max(0, top_k)).copy()
    out.insert(0, "rank", range(1, len(out) + 1))
    return out.reset_index(drop=True)


def riasec_knn_search(
    df: pd.DataFrame,
    code: str,
    *,
    top_k: int = 10,
    first_weight: float = 3.0,
    second_weight: float = 2.0,
    third_weight: float = 1.0,
) -> pd.DataFrame:
    normalized = normalize_riasec_code(code)
    if normalized is None:
        raise InvalidRIASECCode("RIASEC code must be non-empty.")

    letter_order = ["R", "I", "A", "S", "E", "C"]
    letter_to_idx = {k: i for i, k in enumerate(letter_order)}

    matrix = np.zeros((len(df), len(letter_order)), dtype=float)
    for i, (_, row) in enumerate(df.iterrows()):
        first, second, third = _row_letters(row)
        if first is not None:
            matrix[i, letter_to_idx[first]] += first_weight
        if second is not None:
            matrix[i, letter_to_idx[second]] += second_weight
        if third is not None:
            matrix[i, letter_to_idx[third]] += third_weight

    q = np.zeros((1, len(letter_order)), dtype=float)
    for pos, ch in enumerate(normalized):
        positional = max(1.0, 3.0 - float(pos))
        q[0, letter_to_idx[ch]] += positional

    k = min(max(0, top_k), len(df))
    if k == 0:
        return df.head(0).copy()

    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(matrix)
    distances, indices = nn.kneighbors(q, n_neighbors=k)
    idx = indices[0].tolist()
    sims = (1.0 - distances[0]).tolist()

    out = df.iloc[idx].copy()
    out.insert(0, "rank", range(1, len(out) + 1))
    out.insert(1, "knn_similarity", sims)
    return out.reset_index(drop=True)


@dataclass(frozen=True)
class DualSearchResult:
    riasec_code: str | None
    text_query: str | None
    riasec_results: pd.DataFrame | None
    text_results: pd.DataFrame | None
    text_results_by_method: dict[str, pd.DataFrame]
    riasec_knn_results: pd.DataFrame | None
    errors: dict[str, str]


def dual_search(
    df: pd.DataFrame,
    *,
    riasec_code: str | None,
    text_query: str | None,
    top_k: int = 10,
    text_method: VectorMethod = "tfidf",
    text_methods: Iterable[VectorMethod] | None = None,
    include_riasec_knn: bool = False,
) -> DualSearchResult:
    errors: dict[str, str] = {}
    normalized_code = None
    if riasec_code is not None and riasec_code.strip() != "":
        try:
            normalized_code = normalize_riasec_code(riasec_code)
        except InvalidRIASECCode as e:
            errors["riasec"] = str(e)
    normalized_text = None
    if text_query is not None and text_query.strip() != "":
        normalized_text = text_query

    if normalized_code is None and normalized_text is None:
        raise ValueError("Provide at least one of riasec_code or text_query.")

    methods: list[VectorMethod] = list(text_methods) if text_methods is not None else [text_method]
    methods = [m for m in methods if m in ("tfidf", "count")]
    if not methods:
        methods = [text_method]

    combined = None
    if normalized_text is not None:
        combined = build_combined_features(df)

    riasec_future = None
    knn_future = None
    text_futures: dict[str, Any] = {}
    max_workers = 1 + (1 if normalized_code is not None else 0) + (1 if include_riasec_knn and normalized_code is not None else 0) + (len(methods) if normalized_text is not None else 0)
    with ThreadPoolExecutor(max_workers=max(2, min(8, max_workers))) as ex:
        if normalized_code is not None:
            riasec_future = ex.submit(riasec_code_match, df, normalized_code, top_k=top_k)
            if include_riasec_knn:
                knn_future = ex.submit(riasec_knn_search, df, normalized_code, top_k=top_k)
        if normalized_text is not None:
            for m in methods:
                text_futures[m] = ex.submit(
                    text_vector_search,
                    df,
                    normalized_text,
                    combined_features=combined,
                    method=m,
                    top_k=top_k,
                )

    riasec_results = None
    if riasec_future is not None:
        try:
            riasec_results = riasec_future.result()
        except Exception as e:
            errors["riasec"] = str(e)

    riasec_knn_results = None
    if knn_future is not None:
        try:
            riasec_knn_results = knn_future.result()
        except Exception as e:
            errors["riasec_knn"] = str(e)

    text_results_by_method: dict[str, pd.DataFrame] = {}
    for m, fut in text_futures.items():
        try:
            text_results_by_method[m] = fut.result()
        except Exception as e:
            errors[f"text_{m}"] = str(e)

    primary_method: VectorMethod = text_method if text_method in text_results_by_method else (methods[0] if methods else text_method)
    text_results = text_results_by_method.get(primary_method)

    return DualSearchResult(
        riasec_code=normalized_code,
        text_query=normalized_text,
        riasec_results=riasec_results,
        text_results=text_results,
        text_results_by_method=text_results_by_method,
        riasec_knn_results=riasec_knn_results,
        errors=errors,
    )


def format_dual_search_result(result: DualSearchResult, *, max_rows: int = 10) -> str:
    parts: list[str] = []
    if result.riasec_code is not None:
        parts.append("RIASEC Code Matching Results")
        if result.riasec_results is None:
            parts.append(f"  error: {result.errors.get('riasec', 'unknown error')}")
        elif len(result.riasec_results) == 0:
            parts.append("  no matches")
        else:
            parts.append(result.riasec_results.head(max_rows).to_string(index=False))

    if result.riasec_code is not None and result.riasec_knn_results is not None:
        parts.append("")
        parts.append("RIASEC KNN Results")
        parts.append(result.riasec_knn_results.head(max_rows).to_string(index=False))

    if result.text_query is not None:
        for method in ["tfidf", "count"]:
            if method not in result.text_results_by_method and result.text_results is None:
                continue
            if method in result.text_results_by_method:
                df = result.text_results_by_method[method]
                parts.append("")
                parts.append(f"Text Vector Search Results ({method.upper()})")
                if len(df) == 0:
                    parts.append("  no matches")
                else:
                    parts.append(df.head(max_rows).to_string(index=False))

    if result.errors and (result.riasec_code is None or result.text_query is None):
        parts.append("")
        parts.append("Errors")
        for k, v in result.errors.items():
            parts.append(f"{k}: {v}")
    return "\n".join(parts).strip() + "\n"


def load_default_riasec_dataset() -> pd.DataFrame:
    def read_link(link: str) -> io.BytesIO:
        file_id = link.split("/")[-2]
        base = f"https://drive.google.com/uc?export=download&id={file_id}"
        jar = http.cookiejar.CookieJar()
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))

        def fetch(url: str) -> bytes:
            with opener.open(url) as resp:
                return resp.read()

        data = fetch(base)
        if b"Google Drive - Virus scan warning" in data or data.lstrip().startswith(b"<!DOCTYPE html"):
            m = re.search(br"confirm=([0-9A-Za-z_]+)", data)
            confirm = m.group(1).decode("utf-8") if m else "t"
            data = fetch(base + f"&confirm={confirm}")
            if data.lstrip().startswith(b"<!DOCTYPE html"):
                raise ValueError("Failed to download CSV from Google Drive (received HTML).")

        return io.BytesIO(data)

    riasec_df = pd.concat(
        [
            pd.read_csv(read_link("https://drive.google.com/file/d/1Yw8Q-okC156xESWz9ZdYJY8kOZpl3SjR/view?usp=sharing")),
            pd.read_csv(read_link("https://drive.google.com/file/d/1fTj0tFJtQ4htBEa1dpLVjhWe93PYOHA1/view?usp=sharing")),
            pd.read_csv(read_link("https://drive.google.com/file/d/1CR2IHnrhKC-7EtUjP5s-x8nkie6zSfiZ/view?usp=sharing")),
            pd.read_csv(read_link("https://drive.google.com/file/d/1Me5geVIjvEtMPldfzMOwBWavJDjy-e0c/view?usp=sharing")),
            pd.read_csv(read_link("https://drive.google.com/file/d/1WOTNJ7htmu5jR3gvaCPNLKg0ca1p3jaR/view?usp=sharing")),
            pd.read_csv(read_link("https://drive.google.com/file/d/1crJJh-svX5jGfVgs3oZlciEWJas_YYPY/view?usp=sharing")),
        ],
        ignore_index=True,
    )
    occupation_dataset = pd.read_csv(
        read_link("https://drive.google.com/file/d/1GAURhgjxXMGdlFs2Gk4jaByoSPPPkdPy/view?usp=sharing")
    )
    riasec_df = riasec_df.merge(occupation_dataset, left_on="O*NET-SOC Code", right_on="Code", how="left")
    riasec_df.drop(riasec_df.columns[[0, 1, 2, 6, 7]], axis=1, inplace=True)
    riasec_df.drop_duplicates(inplace=True)
    riasec_df = riasec_df.reset_index(drop=True)
    riasec_df.rename(
        columns={
            "First Interest Area": "First Personality",
            "Second Interest Area": "Second Personality",
            "Third Interest Area": "Third Personality",
        },
        inplace=True,
    )
    riasec_df = riasec_df.fillna("-")
    return riasec_df
