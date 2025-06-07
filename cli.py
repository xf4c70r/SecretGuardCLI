import argparse
from scanner.regex_scanner import scan_with_regex
from scanner.llm_scanner import scan_with_llm
from scanner.hybrid_scanner import scan_hybrid

def main():
    parser = argparse.ArgumentParse(description="SecretGuard-CLI: Scan code for hardcoded secrets")
    parser.add_argument("path", help="Path to the file or directory to scan")
    parser.add_argument("--llm", action="store_true", help="Enable LLM-based scanning")
    parser.add_argument("--regex-only", action="store_true", help="Use only regex-based scanning")

    args = parser.parse_args()

    if args.regex_only:
        scan_with_regex(args.path)
    elif args.llm:
        scan_with_llm(args.path)
    else:
        scan_hybrid(args.path)


__name__ == '__main__':
main()