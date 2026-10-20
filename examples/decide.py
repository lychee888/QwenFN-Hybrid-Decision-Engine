"""Send a decision request. Prints the actual HTTP response, never a canned result."""
import argparse
import json
import os
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default=os.environ.get('QFN_URL', 'http://127.0.0.1:8888'))
    args = parser.parse_args()
    body = {
        'kind': 'choice',
        'state': 'The service is responding normally after a restart.',
        'question': 'Which action fits the current state?',
        'options': [{'key': 'observe', 'description': 'Continue observation'},
                    {'key': 'restart', 'description': 'Restart the service again'}],
        'thinking': 'off', 'arm': 'B0',
    }
    request = urllib.request.Request(args.url.rstrip('/') + '/v1/decide',
                                     data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            print(response.read().decode())
    except urllib.error.HTTPError as exc:
        print(exc.read().decode())
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
