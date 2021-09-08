'''Testing script'''


def process_sdt():
    from pathlib import Path
    import json
    from tqdm.auto import tqdm
    import re

    sdt_dir = Path('datasets/sdt')
    if not sdt_dir.exists():
        sdt_dir.mkdir()

    input_paths = [Path('datasets/sdt-train.conllu'),
                   Path('datasets/sdt-dev.conllu'),
                   Path('datasets/sdt-test.conllu')]
    output_paths = [Path('datasets/sdt/train.jsonl'),
                    Path('datasets/sdt/val.jsonl'),
                    Path('datasets/sdt/test.jsonl')]

    for input_path, output_path in zip(input_paths, output_paths):
        tokens = list()
        pos_tags = list()
        heads = list()
        deps  = list()
        ids = list()
        doc = ''
        lines = input_path.read_text().split('\n')
        for idx, line in enumerate(tqdm(lines)):
            if line.startswith('# text = '):
                doc = re.sub('# text = ', '', line)
            elif line.startswith('#'):
                continue
            elif line == '':
                if tokens != []:
                    data_dict = dict(ids=ids,
                                     doc=doc,
                                     tokens=tokens,
                                     pos_tags=pos_tags,
                                     heads=heads,
                                     deps=deps)
                    json_line = json.dumps(data_dict)
                    with output_path.open('a') as f:
                        f.write(json_line)
                        if idx < len(lines) - 1:
                            f.write('\n')
                ids = list()
                tokens = list()
                pos_tags = list()
                heads = list()
                deps = list()
                doc = ''
            else:
                try:
                    data = line.split('\t')
                    ids.append(data[0])
                    tokens.append(data[1])
                    pos_tags.append(data[3])
                    heads.append(data[6])
                    deps.append(data[7])
                except:
                    print(data)


def process_norne_nn():
    from pathlib import Path
    import json
    from tqdm.auto import tqdm
    import re

    norne_dir = Path('datasets/norne_nn')
    if not norne_dir.exists():
        norne_dir.mkdir()

    input_paths = [Path('datasets/no_nynorsk-ud-train.conllu'),
                   Path('datasets/no_nynorsk-ud-dev.conllu'),
                   Path('datasets/no_nynorsk-ud-test.conllu')]
    output_paths = [Path('datasets/norne_nn/train.jsonl'),
                    Path('datasets/norne_nn/val.jsonl'),
                    Path('datasets/norne_nn/test.jsonl')]

    for input_path, output_path in zip(input_paths, output_paths):
        tokens = list()
        pos_tags = list()
        heads = list()
        deps  = list()
        ner_tags = list()
        ids = list()
        doc = ''
        lines = input_path.read_text().split('\n')
        for idx, line in enumerate(tqdm(lines)):
            if line.startswith('# text = '):
                doc = re.sub('# text = ', '', line)
            elif line.startswith('#'):
                continue
            elif line == '':
                if tokens != []:
                    data_dict = dict(ids=ids,
                                     doc=doc,
                                     tokens=tokens,
                                     pos_tags=pos_tags,
                                     heads=heads,
                                     deps=deps,
                                     ner_tags=ner_tags)
                    json_line = json.dumps(data_dict)
                    with output_path.open('a') as f:
                        f.write(json_line)
                        if idx < len(lines) - 1:
                            f.write('\n')
                ids = list()
                tokens = list()
                pos_tags = list()
                heads = list()
                deps = list()
                ner_tags = list()
                doc = ''
            else:
                data = line.split('\t')
                ids.append(data[0])
                tokens.append(data[1])
                pos_tags.append(data[3])
                heads.append(data[6])
                deps.append(data[7])
                ner_tags.append(data[9].replace('name=', '').split('|')[-1])


def process_norne_nb():
    from pathlib import Path
    import json
    from tqdm.auto import tqdm
    import re

    norne_dir = Path('datasets/norne_nb')
    if not norne_dir.exists():
        norne_dir.mkdir()

    input_paths = [Path('datasets/no_bokmaal-ud-train.conllu'),
                   Path('datasets/no_bokmaal-ud-dev.conllu'),
                   Path('datasets/no_bokmaal-ud-test.conllu')]
    output_paths = [Path('datasets/norne_nb/train.jsonl'),
                    Path('datasets/norne_nb/val.jsonl'),
                    Path('datasets/norne_nb/test.jsonl')]

    for input_path, output_path in zip(input_paths, output_paths):
        tokens = list()
        pos_tags = list()
        heads = list()
        deps  = list()
        ner_tags = list()
        ids = list()
        doc = ''
        lines = input_path.read_text().split('\n')
        for idx, line in enumerate(tqdm(lines)):
            if line.startswith('# text = '):
                doc = re.sub('# text = ', '', line)
            elif line.startswith('#'):
                continue
            elif line == '':
                if tokens != []:
                    data_dict = dict(ids=ids,
                                     doc=doc,
                                     tokens=tokens,
                                     pos_tags=pos_tags,
                                     heads=heads,
                                     deps=deps,
                                     ner_tags=ner_tags)
                    json_line = json.dumps(data_dict)
                    with output_path.open('a') as f:
                        f.write(json_line)
                        if idx < len(lines) - 1:
                            f.write('\n')
                ids = list()
                tokens = list()
                pos_tags = list()
                heads = list()
                deps = list()
                ner_tags = list()
                doc = ''
            else:
                data = line.split('\t')
                ids.append(data[0])
                tokens.append(data[1])
                pos_tags.append(data[3])
                heads.append(data[6])
                deps.append(data[7])
                ner_tags.append(data[9].replace('name=', '').split('|')[-1])


def process_nordial():
    from pathlib import Path
    import json
    from tqdm.auto import tqdm
    import pandas as pd
    import json

    dataset_dir = Path('datasets/nordial')
    if not dataset_dir.exists():
        dataset_dir.mkdir()

    train_input_path = Path('datasets/nordial_train.json')
    val_input_path = Path('datasets/nordial_val.json')
    test_input_path = Path('datasets/nordial_test.json')
    output_paths = [dataset_dir / 'train.jsonl', dataset_dir / 'test.jsonl']

    train = pd.read_json(train_input_path, orient='records').dropna()
    val = pd.read_json(val_input_path, orient='records').dropna()
    train = train.append(val)
    test = pd.read_json(test_input_path, orient='records').dropna()

    for split, output_path in zip([train, test], output_paths):
        for idx, row in tqdm(split.iterrows()):
            data_dict = dict(text=row.text, label=row.category)
            json_line = json.dumps(data_dict)
            with output_path.open('a') as f:
                f.write(json_line)
                if idx < len(split) - 1:
                    f.write('\n')


def process_norec():
    from pathlib import Path
    import json
    from tqdm.auto import tqdm
    import pandas as pd
    import json

    dataset_dir = Path('datasets/norec')
    if not dataset_dir.exists():
        dataset_dir.mkdir()

    train_input_path = Path('datasets/norec_train.json')
    val_input_path = Path('datasets/norec_val.json')
    test_input_path = Path('datasets/norec_test.json')
    output_paths = [dataset_dir / 'train.jsonl', dataset_dir / 'test.jsonl']

    train = pd.read_json(train_input_path, orient='records').dropna()
    val = pd.read_json(val_input_path, orient='records').dropna()
    train = train.append(val)
    test = pd.read_json(test_input_path, orient='records').dropna()

    for split, output_path in zip([train, test], output_paths):
        for idx, row in tqdm(split.iterrows()):
            data_dict = dict(text=row.text, label=row.label)
            json_line = json.dumps(data_dict)
            with output_path.open('a') as f:
                f.write(json_line)
                if idx < len(split) - 1:
                    f.write('\n')


def process_twitter_subj():
    from pathlib import Path
    import json
    from tqdm.auto import tqdm
    import pandas as pd

    Path('datasets/twitter_subj').mkdir()

    input_path = Path('datasets/twitter_sent.csv')
    output_paths = [Path('datasets/twitter_subj/train.jsonl'),
                    Path('datasets/twitter_subj/test.jsonl')]

    df = pd.read_csv(input_path, header=0).dropna()

    train = df.query('part == "train"')
    test = df.query('part == "test"')

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    for split, output_path in zip([train, test], output_paths):
        for idx, row in tqdm(split.iterrows()):
            data_dict = dict(tweet_id=row.twitterid, label=row['sub/obj'])
            json_line = json.dumps(data_dict)
            with output_path.open('a') as f:
                f.write(json_line)
                if idx < len(split) - 1:
                    f.write('\n')


def process_twitter_sent_sentiment():
    from pathlib import Path
    import json
    from tqdm.auto import tqdm
    import pandas as pd

    Path('datasets/twitter_sent').mkdir()

    input_path = Path('datasets/twitter_sent.csv')
    output_paths = [Path('datasets/twitter_sent/train.jsonl'),
                    Path('datasets/twitter_sent/test.jsonl')]

    df = pd.read_csv(input_path, header=0).dropna()

    train = df.query('part == "train"')
    test = df.query('part == "test"')

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    for split, output_path in zip([train, test], output_paths):
        for idx, row in tqdm(split.iterrows()):
            data_dict = dict(tweet_id=row.twitterid, label=row.sentiment)
            json_line = json.dumps(data_dict)
            with output_path.open('a') as f:
                f.write(json_line)
                if idx < len(split) - 1:
                    f.write('\n')


def process_lcc():
    from pathlib import Path
    import json
    from tqdm.auto import tqdm
    import pandas as pd
    from sklearn.model_selection import train_test_split

    Path('datasets/lcc').mkdir()

    input_paths = [Path('datasets/lcc1.csv'), Path('datasets/lcc2.csv')]
    output_paths = [Path('datasets/lcc/train.jsonl'),
                    Path('datasets/lcc/test.jsonl')]

    dfs = list()
    for input_path in input_paths:
        df = (pd.read_csv(input_path, header=0)
                .dropna(subset=['valence', 'text']))

        for idx, row in df.iterrows():
            try:
                int(row.valence)
            except:
                df = df.drop(idx)
                continue
            if row.text.strip() == '':
                df = df.drop(idx)
            else:
                if int(row.valence) > 0:
                    sentiment = 'positiv'
                elif int(row.valence) < 0:
                    sentiment = 'negativ'
                else:
                    sentiment = 'neutral'
                df.loc[idx, 'valence'] = sentiment

        dfs.append(df)

    df = pd.concat(dfs, axis=0, ignore_index=True)

    train, test = train_test_split(df, test_size=0.3, stratify=df.valence)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    for split, output_path in zip([train, test], output_paths):
        for idx, row in tqdm(split.iterrows()):
            data_dict = dict(text=row.text, label=row.valence)
            json_line = json.dumps(data_dict)
            with output_path.open('a') as f:
                f.write(json_line)
                if idx < len(split) - 1:
                    f.write('\n')


def process_lcc2():
    from pathlib import Path
    import json
    from tqdm.auto import tqdm
    import pandas as pd
    from sklearn.model_selection import train_test_split

    Path('datasets/lcc2').mkdir()

    input_path = Path('datasets/lcc2.csv')
    output_paths = [Path('datasets/lcc2/train.jsonl'),
                    Path('datasets/lcc2/test.jsonl')]

    df = pd.read_csv(input_path, header=0).dropna(subset=['valence', 'text'])

    for idx, row in df.iterrows():
        try:
            int(row.valence)
        except:
            df = df.drop(idx)
            continue
        if row.text.strip() == '':
            df = df.drop(idx)
        else:
            if int(row.valence) > 0:
                sentiment = 'positiv'
            elif int(row.valence) < 0:
                sentiment = 'negativ'
            else:
                sentiment = 'neutral'
            df.loc[idx, 'valence'] = sentiment

    train, test = train_test_split(df, test_size=0.3, stratify=df.valence)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    for split, output_path in zip([train, test], output_paths):
        for idx, row in tqdm(split.iterrows()):
            data_dict = dict(text=row.text, label=row.valence)
            json_line = json.dumps(data_dict)
            with output_path.open('a') as f:
                f.write(json_line)
                if idx < len(split) - 1:
                    f.write('\n')


def process_lcc1():
    from pathlib import Path
    import json
    from tqdm.auto import tqdm
    import pandas as pd
    from sklearn.model_selection import train_test_split

    Path('datasets/lcc1').mkdir()

    input_path = Path('datasets/lcc1.csv')
    output_paths = [Path('datasets/lcc1/train.jsonl'),
                    Path('datasets/lcc1/test.jsonl')]

    df = pd.read_csv(input_path, header=0).dropna(subset=['valence', 'text'])

    for idx, row in df.iterrows():
        try:
            int(row.valence)
        except:
            df = df.drop(idx)
            continue
        if row.text.strip() == '':
            df = df.drop(idx)
        else:
            if int(row.valence) > 0:
                sentiment = 'positiv'
            elif int(row.valence) < 0:
                sentiment = 'negativ'
            else:
                sentiment = 'neutral'
            df.loc[idx, 'valence'] = sentiment

    train, test = train_test_split(df, test_size=0.3, stratify=df.valence)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    for split, output_path in zip([train, test], output_paths):
        for idx, row in tqdm(split.iterrows()):
            data_dict = dict(text=row.text, label=row.valence)
            json_line = json.dumps(data_dict)
            with output_path.open('a') as f:
                f.write(json_line)
                if idx < len(split) - 1:
                    f.write('\n')


def process_europarl_subj():
    from pathlib import Path
    import json
    from tqdm.auto import tqdm
    import pandas as pd
    from sklearn.model_selection import train_test_split

    Path('datasets/europarl_subj').mkdir()

    input_path = Path('datasets/europarl2.csv')
    output_paths = [Path('datasets/europarl_subj/train.jsonl'),
                    Path('datasets/europarl_subj/test.jsonl')]

    df = pd.read_csv(input_path, header=0).dropna()

    train, test = train_test_split(df, test_size=0.3, stratify=df['sub/obj'])
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    for split, output_path in zip([train, test], output_paths):
        for idx, row in tqdm(split.iterrows(), total=len(split)):
            data_dict = dict(text=row.text, label=row['sub/obj'])
            json_line = json.dumps(data_dict)
            with output_path.open('a') as f:
                f.write(json_line)
                if idx < len(split) - 1:
                    f.write('\n')


def process_europarl2():
    from pathlib import Path
    import json
    from tqdm.auto import tqdm
    import pandas as pd
    from sklearn.model_selection import train_test_split

    Path('datasets/europarl2').mkdir()

    input_path = Path('datasets/europarl2.csv')
    output_paths = [Path('datasets/europarl2/train.jsonl'),
                    Path('datasets/europarl2/test.jsonl')]

    df = pd.read_csv(input_path, header=0).dropna()

    train, test = train_test_split(df, test_size=0.3, stratify=df.polarity)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    for split, output_path in zip([train, test], output_paths):
        for idx, row in tqdm(split.iterrows(), total=len(split)):
            data_dict = dict(text=row.text, label=row.polarity)
            json_line = json.dumps(data_dict)
            with output_path.open('a') as f:
                f.write(json_line)
                if idx < len(split) - 1:
                    f.write('\n')


def process_europarl1():
    from pathlib import Path
    import json
    from tqdm.auto import tqdm
    import pandas as pd
    from sklearn.model_selection import train_test_split

    input_path = Path('datasets/europarl1.csv')
    output_paths = [Path('datasets/europarl1/train.jsonl'),
                    Path('datasets/europarl1/test.jsonl')]

    df = pd.read_csv(input_path, header=0).dropna(subset=['valence', 'text'])

    for idx, row in df.iterrows():
        try:
            int(row.valence)
        except:
            df = df.drop(idx)
            continue
        if row.text.strip() == '':
            df = df.drop(idx)
        else:
            if int(row.valence) > 0:
                sentiment = 'positiv'
            elif int(row.valence) < 0:
                sentiment = 'negativ'
            else:
                sentiment = 'neutral'
            df.loc[idx, 'valence'] = sentiment

    train, test = train_test_split(df, test_size=0.3, stratify=df.valence)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    for split, output_path in zip([train, test], output_paths):
        for idx, row in tqdm(split.iterrows()):
            data_dict = dict(text=row.text, label=row.valence)
            json_line = json.dumps(data_dict)
            with output_path.open('a') as f:
                f.write(json_line)
                if idx < len(split) - 1:
                    f.write('\n')


def process_angrytweets():
    from pathlib import Path
    import json
    from tqdm.auto import tqdm
    import pandas as pd
    from sklearn.model_selection import train_test_split

    input_path = Path('datasets/angry_tweets.csv')
    output_paths = [Path('datasets/angry_tweets/train.jsonl'),
                    Path('datasets/angry_tweets/test.jsonl')]

    df = pd.read_csv(input_path, header=0)

    for idx, row in df.iterrows():
        labels = json.loads(row.annotation.replace('\'', '\"'))
        if len(set(labels)) > 1 or 'skip' in labels:
            df = df.drop(idx)
        else:
            df.loc[idx, 'annotation'] = labels[0]

    train, test = train_test_split(df, test_size=0.3, stratify=df.annotation)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    for split, output_path in zip([train, test], output_paths):
        for idx, row in tqdm(split.iterrows()):
            data_dict = dict(tweet_id=row.twitterid, label=row.annotation)
            json_line = json.dumps(data_dict)
            with output_path.open('a') as f:
                f.write(json_line)
                if idx < len(split) - 1:
                    f.write('\n')


def process_dkhate():
    from pathlib import Path
    import json
    from tqdm.auto import tqdm
    import pandas as pd

    input_paths = [Path('datasets/dkhate/dkhate.train.tsv'),
                   Path('datasets/dkhate/dkhate.test.tsv')]
    output_paths = [Path('datasets/dkhate/dkhate_train.jsonl'),
                    Path('datasets/dkhate/dkhate_test.jsonl')]

    for input_path, output_path in zip(input_paths, output_paths):
        df = pd.read_csv(input_path, sep='\t')
        for idx, row in tqdm(df.iterrows()):
            data_dict = dict(text=row.tweet, label=row.subtask_a)
            json_line = json.dumps(data_dict)
            with output_path.open('a') as f:
                f.write(json_line)
                if idx < len(df) - 1:
                    f.write('\n')

def process_dane():
    from pathlib import Path
    import json
    from tqdm.auto import tqdm
    import re

    input_paths = [Path('datasets/dane/ddt.train.conllu'),
                   Path('datasets/dane/ddt.dev.conllu'),
                   Path('datasets/dane/ddt.test.conllu')]
    output_paths = [Path('datasets/dane/dane_train.jsonl'),
                    Path('datasets/dane/dane_val.jsonl'),
                    Path('datasets/dane/dane_test.jsonl')]

    for input_path, output_path in zip(input_paths, output_paths):
        tokens = list()
        pos_tags = list()
        heads = list()
        deps  = list()
        ner_tags = list()
        ids = list()
        doc = ''
        lines = input_path.read_text().split('\n')
        for idx, line in enumerate(tqdm(lines)):
            if line.startswith('# text = '):
                doc = re.sub('# text = ', '', line)
            elif line.startswith('#'):
                continue
            elif line == '':
                if tokens != []:
                    data_dict = dict(ids=ids,
                                     doc=doc,
                                     tokens=tokens,
                                     pos_tags=pos_tags,
                                     heads=heads,
                                     deps=deps,
                                     ner_tags=ner_tags)
                    json_line = json.dumps(data_dict)
                    with output_path.open('a') as f:
                        f.write(json_line)
                        if idx < len(lines) - 1:
                            f.write('\n')
                ids = list()
                tokens = list()
                pos_tags = list()
                heads = list()
                deps = list()
                ner_tags = list()
                doc = ''
            else:
                data = line.split('\t')
                ids.append(data[0])
                tokens.append(data[1])
                pos_tags.append(data[3])
                heads.append(data[6])
                deps.append(data[7])
                ner_tags.append(data[9].replace('name=', '').split('|')[0])

def process_dalaj():
    from pathlib import Path
    import pandas as pd
    import json
    from tqdm.auto import tqdm

    dataset_dir = Path('datasets/dalaj')
    if not dataset_dir.exists():
        dataset_dir.mkdir()

    input_path = Path('datasets/datasetDaLAJsplit.csv')
    train_output_path = dataset_dir / 'train.jsonl'
    test_output_path = dataset_dir / 'test.jsonl'

    cols = ['original sentence', 'corrected sentence', 'split']
    df = pd.read_csv(input_path)[cols]

    train = (df.query('(split == "train") or (split == "valid")')
               .drop(columns='split'))
    test = df.query('split == "test"').drop(columns='split')

    def reorganise(df: pd.DataFrame) -> pd.DataFrame:
        df_correct = (df[['corrected sentence']]
                      .rename(columns={'corrected sentence': 'doc'}))
        df_incorrect = (df[['original sentence']]
                        .rename(columns={'original sentence': 'doc'}))
        df_correct['label'] = ['correct' for _ in range(len(df_correct))]
        df_incorrect['label'] = ['incorrect' for _ in range(len(df_incorrect))]
        return pd.concat((df_correct, df_incorrect), ignore_index=True)

    train = reorganise(train)
    test = reorganise(test)

    def export_as_jsonl(df: pd.DataFrame, output_path: Path):
        for idx, row in tqdm(df.iterrows()):
            data_dict = dict(text=row.doc, label=row.label)
            json_line = json.dumps(data_dict)
            with output_path.open('a') as f:
                f.write(json_line)
                if idx < len(df) - 1:
                    f.write('\n')

    export_as_jsonl(train, train_output_path)
    export_as_jsonl(test, test_output_path)


def process_absabank_imm():
    from pathlib import Path
    import pandas as pd
    import json
    from tqdm.auto import tqdm

    dataset_dir = Path('datasets/absabank_imm')
    if not dataset_dir.exists():
        dataset_dir.mkdir()

    input_paths = [Path('datasets/split10_consecutive_average/train00.tsv'),
                   Path('datasets/split10_consecutive_average/dev00.tsv'),
                   Path('datasets/split10_consecutive_average/test00.tsv')]
    output_paths = [Path('datasets/absabank_imm/train.jsonl'),
                   Path('datasets/absabank_imm/val.jsonl'),
                   Path('datasets/absabank_imm/test.jsonl')]

    def export_as_jsonl(df: pd.DataFrame, output_path: Path):
        for idx, row in tqdm(df.iterrows()):
            data_dict = dict(text=row.text, label=row.label)
            json_line = json.dumps(data_dict)
            with output_path.open('a') as f:
                f.write(json_line)
                if idx < len(df) - 1:
                    f.write('\n')

    def convert_label(label: float) -> str:
        if label < 2.5:
            return 'negative'
        elif label > 3.5:
            return 'positive'
        else:
            return 'neutral'

    for input_path, output_path in zip(input_paths, output_paths):
        df = pd.read_csv(input_path, sep='\t')[['text', 'label']]
        df['label'] = df.label.map(convert_label)
        export_as_jsonl(df, output_path)


if __name__ == '__main__':
    process_sdt()
