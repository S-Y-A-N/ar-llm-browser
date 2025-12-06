import argparse
from pathlib import Path
from datasets import load_dataset
import json


def main():
    parser = argparse.ArgumentParser(
        prog="parse",
        description="%(prog)s is a command-line interface used to parse results details files.",
    )

    parser.add_argument(
        "details_file",
        help="- Specify path to details file (*.parquet).",
    )

    args = parser.parse_args()
    print(args)

    parse_details(args.details_file)


def parse_details(details_file):
    details = load_dataset("parquet", data_files=details_file, split="train").to_list()

    for detail in details:
        detail["model_response"].pop("input_tokens", None)
        detail["model_response"].pop("output_tokens", None)
        print(detail)

    d = details_file.split("/")
    output_file = "evaluation/parsed_details/" + "/".join(d[3:])[:-8] + ".json"
    Path("/".join(output_file.split("/")[:-1])).mkdir(parents=True, exist_ok=True)

    details_json = json.dumps(details, indent=4, ensure_ascii=False)
    with open(output_file, "w") as f:
        f.write(details_json)

    print("Parsed details written to:", output_file)


if __name__ == "__main__":
    main()
