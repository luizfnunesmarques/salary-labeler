import argparse
import os
import requests
import json


def main():
    parser = argparse.ArgumentParser(description='Send POST request to API endpoint.')
    parser.add_argument('--url', type=str, help='URL of the API endpoint')
    parser.add_argument('--params.age', type=int, required=True, help='Age')
    parser.add_argument('--params.workclass', type=str, required=True, help='Workclass')
    parser.add_argument('--params.fnlgt', type=int, required=True, help='Fnlgt')
    parser.add_argument('--params.education', type=str, required=True, help='Education')
    parser.add_argument('--params.education-num', type=int, required=True, help='Education number')
    parser.add_argument('--params.marital-status', type=str, required=True, help='Marital status')
    parser.add_argument('--params.occupation', type=str, required=True, help='Occupation')
    parser.add_argument('--params.relationship', type=str, required=True, help='Relationship')
    parser.add_argument('--params.race', type=str, required=True, help='Race')
    parser.add_argument('--params.sex', type=str, required=True, help='Sex')
    parser.add_argument('--params.capital-gain', type=int, required=True, help='Capital gain')
    parser.add_argument('--params.capital-loss', type=int, required=True, help='Capital loss')
    parser.add_argument('--params.hours-per-week', type=int, required=True, help='Hours per week')
    parser.add_argument('--params.native-country', type=str, required=True, help='Native country')
    args = parser.parse_args()

    url = args.url

    if not url:
        url = os.getenv('API_URL')
        if not url:
            print("Error: No URL provided. Please provide a URL using --url argument or set the API_URL environment variable.")
            return

    payload = {
        "age": getattr(args, 'params.age'),
        "workclass": getattr(args, 'params.workclass'),
        "fnlgt": getattr(args, 'params.fnlgt'),
        "education": getattr(args, 'params.education'),
        "education-num": getattr(args, 'params.education_num'),
        "marital-status": getattr(args, 'params.marital_status'),
        "occupation": getattr(args, 'params.occupation'),
        "relationship": getattr(args, 'params.relationship'),
        "race": getattr(args, 'params.race'),
        "sex": getattr(args, 'params.sex'),
        "capital-gain": getattr(args, 'params.capital_gain'),
        "capital-loss": getattr(args, 'params.capital_loss'),
        "hours-per-week": getattr(args, 'params.hours_per_week'),
        "native-country": getattr(args, 'params.native_country')
    }

    json_payload = json.dumps(payload)

    headers = {'Content-Type': 'application/json'}

    response = requests.post(url, data=json_payload, headers=headers)

    if response.status_code == 200:
        print("Request successful.")
        print("Response:")
        print(response.json())
    else:
        print("Request failed with status code:", response.status_code)


if __name__ == "__main__":
    main()
