"""Compute FID with torch-fidelity for two image directories."""
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real", required=True); p.add_argument("--generated", required=True)
    a = p.parse_args()
    try:
        from torch_fidelity import calculate_metrics
    except ImportError as exc:
        raise SystemExit("Install torch-fidelity (`pip install torch-fidelity`) to evaluate FID.") from exc
    print(calculate_metrics(input1=a.real, input2=a.generated, cuda=True, fid=True, isc=False, kid=False))


if __name__ == "__main__": main()
