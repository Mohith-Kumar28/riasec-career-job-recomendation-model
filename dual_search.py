import argparse
import json
import sys

from job_recommender.dual_output import dual_search, format_dual_search_result, load_default_riasec_dataset


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="dual_search")
    parser.add_argument("--riasec", dest="riasec_code", default=None, help="RIASEC code input (e.g. RIA, S-E-C)")
    parser.add_argument("--text", dest="text_query", default=None, help="Free-text query input")
    parser.add_argument("--top-k", dest="top_k", type=int, default=10)
    parser.add_argument("--method", dest="method", choices=["tfidf", "count"], default="tfidf")
    parser.add_argument("--compare", dest="compare", action="store_true")
    parser.add_argument("--json", dest="as_json", action="store_true")
    args = parser.parse_args(argv)

    df = load_default_riasec_dataset()
    text_methods = ["tfidf", "count"] if args.compare else None
    result = dual_search(
        df,
        riasec_code=args.riasec_code,
        text_query=args.text_query,
        top_k=args.top_k,
        text_method=args.method,
        text_methods=text_methods,
        include_riasec_knn=args.compare,
    )

    if args.as_json:
        payload = {
            "riasec_code": result.riasec_code,
            "text_query": result.text_query,
            "errors": result.errors,
            "riasec_results": None if result.riasec_results is None else result.riasec_results.to_dict(orient="records"),
            "text_results": None if result.text_results is None else result.text_results.to_dict(orient="records"),
            "text_results_by_method": {k: v.to_dict(orient="records") for k, v in result.text_results_by_method.items()},
            "riasec_knn_results": None
            if result.riasec_knn_results is None
            else result.riasec_knn_results.to_dict(orient="records"),
        }
        sys.stdout.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    else:
        sys.stdout.write(format_dual_search_result(result, max_rows=args.top_k))
    return 0 if not result.errors else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
