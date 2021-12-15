from pathlib import Path
import json
import re
import logging
import os
from termcolor import colored


format = colored('%(asctime)s [%(levelname)s] <%(name)s>\n↳ ', 'green') + \
         colored('%(message)s', 'yellow')
logging.basicConfig(level=logging.INFO, format=format)
logger = logging.getLogger(__name__)


def create_tweet_dataset(dataset: str):
    dataset_dir = Path('datasets') / dataset
    for name in ['raw_train', 'raw_test']:

        logger.info(f'Now processing the {name} split of {dataset}')

        # Extract the tweet IDs
        logger.info(f'Extracting tweet IDs')
        path = dataset_dir / f'{name}.jsonl'
        lines = [line for line in path.read_text().split('\n') if line != '']
        data_dicts = [json.loads(line) for line in lines]
        tweet_ids = [str(data_dict['tweet_id']) for data_dict in data_dicts]
        tweetid_path = dataset_dir / f'{name}_tweetids.txt'
        with tweetid_path.open('w') as f:
            f.write('\n'.join(tweet_ids))

        # Hydrate the tweets
        logger.info(f'Hydrating tweets')
        tweet_path = dataset_dir / f'{name}_tweets.jsonl'
        os.system(f'twarc hydrate {tweetid_path} > {tweet_path}')

        # Load the hydrated tweets
        lines = [line for line in tweet_path.read_text().split('\n')
                      if line != '']
        tweets= [json.loads(line) for line in lines]

        # Anonymise the tweets
        logger.info(f'Anonymising tweets')
        records = list()
        for tweet in tweets:
            for data_dict in data_dicts:
                if data_dict['tweet_id'] == tweet['id']:
                    text = tweet['full_text']
                    text = re.sub(r'@[a-zA-Z0-9_]+', '@USER', text)
                    text = re.sub(r'http[.\/?&a-zA-Z0-9\-\:]+', '[LINK]', text)
                    text = re.sub(r' +', ' ', text)
                    record = dict(text=text.strip(), label=data_dict['label'])
                    records.append(json.dumps(record))

        # Save the anonymised tweets
        output_path = dataset_dir / f'{name}_processed.jsonl'
        with output_path.open('w') as f:
            f.write('\n'.join(records))
        logger.info(f'Anonymised tweets saved to {output_path}')

        # Delete auxilliary files
        tweetid_path.unlink()
        tweet_path.unlink()


if __name__ == '__main__':
    create_tweet_dataset('twitter_sent')
