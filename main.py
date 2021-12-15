'''Dataset preprocessing scripts'''


def process_mim_gold_ner():
    from pathlib import Path
    import pandas as pd
    from tqdm.auto import tqdm
    import json
    import re
    from collections import defaultdict

    conversion_dict = {
        'O': 'O',
        'B-Person': 'B-PER',
        'I-Person': 'I-PER',
        'B-Location': 'B-LOC',
        'I-Location': 'I-LOC',
        'B-Organization': 'B-ORG',
        'I-Organization': 'I-ORG',
        'B-Miscellaneous': 'B-MISC',
        'I-Miscellaneous': 'I-MISC',
        'B-Date': 'O',
        'I-Date': 'O',
        'B-Time': 'O',
        'I-Time': 'O',
        'B-Money': 'O',
        'I-Money': 'O',
        'B-Percent': 'O',
        'I-Percent': 'O'
    }

    def get_df(path: Path):
        lines = path.read_text().split('\n')
        data_dict = defaultdict(list)
        tokens = list()
        tags = list()
        for line in tqdm(lines):
            if line != '':
                token, tag = line.split('\t')
                tag = conversion_dict[tag]
                tokens.append(token)
                tags.append(tag)
            else:
                doc = ' '.join(tokens)
                doc = re.sub(' ([.,])', '\1', doc)
                data_dict['doc'].append(doc)
                data_dict['tokens'].append(tokens)
                data_dict['ner_tags'].append(tags)
                tokens = list()
                tags = list()

        return pd.DataFrame(data_dict)

    def export_as_jsonl(df: pd.DataFrame, output_path: Path):
        for idx, row in tqdm(list(df.iterrows())):
            data_dict = dict(doc=row.doc,
                             tokens=row.tokens,
                             ner_tags=row.ner_tags)
            json_line = json.dumps(data_dict)
            with output_path.open('a') as f:
                f.write(json_line)
                if idx < len(df) - 1:
                    f.write('\n')

    data_dir = Path('datasets') / 'mim_gold_ner'
    train_input_path = data_dir / 'raw_train'
    val_input_path = data_dir / 'raw_val'
    test_input_path = data_dir / 'raw_test'
    train_output_path = data_dir / 'train.jsonl'
    test_output_path = data_dir / 'test.jsonl'

    train_df = pd.concat((get_df(train_input_path), get_df(val_input_path)))
    test_df = get_df(test_input_path)

    export_as_jsonl(train_df, train_output_path)
    export_as_jsonl(test_df, test_output_path)


def process_fdt():
    from pathlib import Path
    import json
    from tqdm.auto import tqdm
    import re

    dep_conversion_dict = {
       'acl': 'acl',
       'acl:relcl': 'acl',
       'acl:cleft': 'acl',
       'advcl': 'advcl',
       'advmod': 'advmod',
       'advmod:emph': 'advmod',
       'advmod:lmod': 'advmod',
       'amod': 'amod',
       'appos': 'appos',
       'aux': 'aux',
       'aux:pass': 'aux',
       'case': 'case',
       'cc': 'cc',
       'cc:preconj': 'cc',
       'ccomp': 'ccomp',
       'clf': 'clf',
       'compound': 'compound',
       'compound:lvc': 'compound',
       'compound:prt': 'compound',
       'compound:redup': 'compound',
       'compound:svc': 'compound',
       'conj': 'conj',
       'cop': 'cop',
       'csubj': 'csubj',
       'csubj:pass': 'csubj',
       'dep': 'dep',
       'det': 'det',
       'det:numgov': 'det',
       'det:nummod': 'det',
       'det:poss': 'det',
       'discourse': 'discourse',
       'dislocated': 'dislocated',
       'expl': 'expl',
       'expl:impers': 'expl',
       'expl:pass': 'expl',
       'expl:pv': 'expl',
       'fixed': 'fixed',
       'flat': 'flat',
       'flat:foreign': 'flat',
       'flat:name': 'flat',
       'goeswith': 'goeswith',
       'iobj': 'iobj',
       'list': 'list',
       'mark': 'mark',
       'nmod': 'nmod',
       'nmod:poss': 'nmod',
       'nmod:tmod': 'nmod',
       'nsubj': 'nsubj',
       'nsubj:pass': 'nsubj',
       'nummod': 'nummod',
       'nummod:gov': 'nummod',
       'obj': 'obj',
       'obl': 'obl',
       'obl:agent': 'obl',
       'obl:arg': 'obl',
       'obl:lmod': 'obl',
       'obl:loc': 'obl',
       'obl:tmod': 'obl',
       'orphan': 'orphan',
       'parataxis': 'parataxis',
       'punct': 'punct',
       'reparandum': 'reparandum',
       'root': 'root',
       'vocative': 'vocative',
       'xcomp': 'xcomp'
    }

    dataset_dir = Path('datasets/fdt')
    if not dataset_dir.exists():
        dataset_dir.mkdir()

    input_paths = [Path('datasets/fo_farpahc-ud-train.conllu'),
                   Path('datasets/fo_farpahc-ud-dev.conllu'),
                   Path('datasets/fo_farpahc-ud-test.conllu')]
    output_paths = [Path('datasets/fdt/train.jsonl'),
                    Path('datasets/fdt/val.jsonl'),
                    Path('datasets/fdt/test.jsonl')]

    for input_path, output_path in zip(input_paths, output_paths):
        tokens = list()
        pos_tags = list()
        heads = list()
        deps  = list()
        ids = list()
        doc = ''
        lines = input_path.read_text().split('\n')
        store = True
        for idx, line in enumerate(tqdm(lines)):
            if line.startswith('# text = '):
                doc = re.sub('# text = ', '', line)
                store = True
            elif line.startswith('#'):
                continue
            elif line == '':
                if tokens != [] and store:
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
                data = line.split('\t')
                ids.append(data[0])
                tokens.append(data[1])
                pos_tags.append(data[3])
                heads.append(data[6])
                try:
                    deps.append(dep_conversion_dict[data[7]])
                except KeyError:
                    store = False


def process_wikiann_fo():
    from pathlib import Path
    import json
    from tqdm.auto import tqdm
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import re

    dataset_dir = Path('datasets/wikiann_fo')
    if not dataset_dir.exists():
        dataset_dir.mkdir()

    input_path = Path('datasets/wikiann-fo.bio')
    train_output_path = Path('datasets/wikiann_fo/train.jsonl')
    test_output_path = Path('datasets/wikiann_fo/test.jsonl')

    corpus = input_path.read_text().split('\n')

    tokens = list()
    ner_tags = list()
    records = list()
    for line in corpus:
        if line != '':
            data = line.split(' ')
            tokens.append(data[0])
            ner_tags.append(data[-1])

        else:
            assert len(tokens) == len(ner_tags)
            doc = ' '.join(tokens)
            doc = re.sub(' ([.,])', '\1', doc)
            records.append(dict(doc=doc, tokens=tokens, ner_tags=ner_tags))
            tokens = list()
            ner_tags = list()

    # Show the NER tags in the dataset, as a sanity check
    print(sorted(set([tag for record in records
                      for tag in record['ner_tags']])))

    # Count the number of each NER tag, as a sanity check
    tags = ['PER', 'LOC', 'ORG', 'MISC']
    for tag in tags:
        num = len([t for record in records for t in record['ner_tags']
                   if t[2:] == tag])
        print(tag, num)

    df = pd.DataFrame.from_records(records)
    train, test = train_test_split(df, test_size=0.3)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    def export_as_jsonl(df: pd.DataFrame, output_path: Path):
        for idx, row in tqdm(df.iterrows()):
            data_dict = dict(doc=row.doc,
                             tokens=row.tokens,
                             ner_tags=row.ner_tags)
            json_line = json.dumps(data_dict)
            with output_path.open('a') as f:
                f.write(json_line)
                if idx < len(df) - 1:
                    f.write('\n')

    export_as_jsonl(train, train_output_path)
    export_as_jsonl(test, test_output_path)


def process_idt():
    from pathlib import Path
    import json
    from tqdm.auto import tqdm
    import re

    dep_conversion_dict = {
       'acl': 'acl',
       'acl:relcl': 'acl',
       'acl:cleft': 'acl',
       'advcl': 'advcl',
       'advmod': 'advmod',
       'advmod:emph': 'advmod',
       'advmod:lmod': 'advmod',
       'amod': 'amod',
       'appos': 'appos',
       'aux': 'aux',
       'aux:pass': 'aux',
       'case': 'case',
       'cc': 'cc',
       'cc:preconj': 'cc',
       'ccomp': 'ccomp',
       'clf': 'clf',
       'compound': 'compound',
       'compound:lvc': 'compound',
       'compound:prt': 'compound',
       'compound:redup': 'compound',
       'compound:svc': 'compound',
       'conj': 'conj',
       'cop': 'cop',
       'csubj': 'csubj',
       'csubj:pass': 'csubj',
       'dep': 'dep',
       'det': 'det',
       'det:numgov': 'det',
       'det:nummod': 'det',
       'det:poss': 'det',
       'discourse': 'discourse',
       'dislocated': 'dislocated',
       'expl': 'expl',
       'expl:impers': 'expl',
       'expl:pass': 'expl',
       'expl:pv': 'expl',
       'fixed': 'fixed',
       'flat': 'flat',
       'flat:foreign': 'flat',
       'flat:name': 'flat',
       'goeswith': 'goeswith',
       'iobj': 'iobj',
       'list': 'list',
       'mark': 'mark',
       'nmod': 'nmod',
       'nmod:poss': 'nmod',
       'nmod:tmod': 'nmod',
       'nsubj': 'nsubj',
       'nsubj:pass': 'nsubj',
       'nummod': 'nummod',
       'nummod:gov': 'nummod',
       'obj': 'obj',
       'obl': 'obl',
       'obl:agent': 'obl',
       'obl:arg': 'obl',
       'obl:lmod': 'obl',
       'obl:loc': 'obl',
       'obl:tmod': 'obl',
       'orphan': 'orphan',
       'parataxis': 'parataxis',
       'punct': 'punct',
       'reparandum': 'reparandum',
       'root': 'root',
       'vocative': 'vocative',
       'xcomp': 'xcomp'
    }

    dataset_dir = Path('datasets/idt')
    if not dataset_dir.exists():
        dataset_dir.mkdir()

    input_paths = [Path('datasets/is_modern-ud-train.conllu'),
                   Path('datasets/is_modern-ud-dev.conllu'),
                   Path('datasets/is_modern-ud-test.conllu')]
    output_paths = [Path('datasets/idt/train.jsonl'),
                    Path('datasets/idt/val.jsonl'),
                    Path('datasets/idt/test.jsonl')]

    for input_path, output_path in zip(input_paths, output_paths):
        tokens = list()
        pos_tags = list()
        heads = list()
        deps  = list()
        ids = list()
        doc = ''
        lines = input_path.read_text().split('\n')
        store = True
        for idx, line in enumerate(tqdm(lines)):
            if line.startswith('# text = '):
                doc = re.sub('# text = ', '', line)
                store = True
            elif line.startswith('#'):
                continue
            elif line == '':
                if tokens != [] and store:
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
                data = line.split('\t')
                ids.append(data[0])
                tokens.append(data[1])
                pos_tags.append(data[3])
                heads.append(data[6])
                try:
                    deps.append(dep_conversion_dict[data[7]])
                except KeyError:
                    store = False


def process_suc3():
    from pathlib import Path
    import json
    from tqdm.auto import tqdm
    import re
    from lxml import etree
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import io

    sdt_dir = Path('datasets/suc3')
    if not sdt_dir.exists():
        sdt_dir.mkdir()

    conversion_dict = dict(O='O', animal='MISC', event='MISC', inst='ORG',
                           myth='MISC', other='MISC', person='PER',
                           place='LOC', product='MISC', work='MISC')

    input_path = Path('datasets/suc3.xml')
    train_output_path = Path('datasets/suc3/train.jsonl')
    test_output_path = Path('datasets/suc3/test.jsonl')

    print('Parsing XML file...')
    xml_data = input_path.read_bytes()
    context = etree.iterparse(io.BytesIO(xml_data), events=('start', 'end'))

    ner_tag = 'O'
    records = list()
    for action, elt in context:
        if elt.tag == 'name' and action == 'start':
            ner_tag = f'B-{conversion_dict[elt.attrib["type"]]}'

        elif elt.tag == 'name' and action == 'end':
            ner_tag = 'O'

        elif elt.tag == 'w' and action == 'start':
            if elt.text:
                tokens.append(elt.text)
                ner_tags.append(ner_tag)

        elif elt.tag == 'w' and action == 'end':
            if ner_tag.startswith('B-'):
                ner_tag = f'I-{ner_tag[2:]}'

        elif elt.tag == 'sentence' and action == 'end':
            if len(tokens):
                doc = ' '.join(tokens)
                doc = re.sub(' ([.,])', '\1', doc)
                assert len(tokens) == len(ner_tags)
                record = dict(doc=doc, tokens=tokens, ner_tags=ner_tags)
                records.append(record)

        elif elt.tag == 'sentence' and action == 'start':
            tokens = list()
            ner_tags = list()
            ner_tag = 'O'

    # Count the number of each NER tag, as a sanity check
    tags = ['PER', 'LOC', 'ORG', 'MISC']
    for tag in tags:
        num = len([t for record in records for t in record['ner_tags']
                   if t[2:] == tag])
        print(tag, num)

    df = pd.DataFrame.from_records(records)
    train, test = train_test_split(df, test_size=0.3)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    def export_as_jsonl(df: pd.DataFrame, output_path: Path):
        for idx, row in tqdm(df.iterrows()):
            data_dict = dict(doc=row.doc,
                             tokens=row.tokens,
                             ner_tags=row.ner_tags)
            json_line = json.dumps(data_dict)
            with output_path.open('a') as f:
                f.write(json_line)
                if idx < len(df) - 1:
                    f.write('\n')

    export_as_jsonl(train, train_output_path)
    export_as_jsonl(test, test_output_path)


def process_sdt():
    from pathlib import Path
    import json
    from tqdm.auto import tqdm
    import re

    dep_conversion_dict = {
       'acl': 'acl',
       'acl:relcl': 'acl',
       'acl:cleft': 'acl',
       'advcl': 'advcl',
       'advmod': 'advmod',
       'advmod:emph': 'advmod',
       'advmod:lmod': 'advmod',
       'amod': 'amod',
       'appos': 'appos',
       'aux': 'aux',
       'aux:pass': 'aux',
       'case': 'case',
       'cc': 'cc',
       'cc:preconj': 'cc',
       'ccomp': 'ccomp',
       'clf': 'clf',
       'compound': 'compound',
       'compound:lvc': 'compound',
       'compound:prt': 'compound',
       'compound:redup': 'compound',
       'compound:svc': 'compound',
       'conj': 'conj',
       'cop': 'cop',
       'csubj': 'csubj',
       'csubj:pass': 'csubj',
       'dep': 'dep',
       'det': 'det',
       'det:numgov': 'det',
       'det:nummod': 'det',
       'det:poss': 'det',
       'discourse': 'discourse',
       'dislocated': 'dislocated',
       'expl': 'expl',
       'expl:impers': 'expl',
       'expl:pass': 'expl',
       'expl:pv': 'expl',
       'fixed': 'fixed',
       'flat': 'flat',
       'flat:foreign': 'flat',
       'flat:name': 'flat',
       'goeswith': 'goeswith',
       'iobj': 'iobj',
       'list': 'list',
       'mark': 'mark',
       'nmod': 'nmod',
       'nmod:poss': 'nmod',
       'nmod:tmod': 'nmod',
       'nsubj': 'nsubj',
       'nsubj:pass': 'nsubj',
       'nummod': 'nummod',
       'nummod:gov': 'nummod',
       'obj': 'obj',
       'obl': 'obl',
       'obl:agent': 'obl',
       'obl:arg': 'obl',
       'obl:lmod': 'obl',
       'obl:loc': 'obl',
       'obl:tmod': 'obl',
       'orphan': 'orphan',
       'parataxis': 'parataxis',
       'punct': 'punct',
       'reparandum': 'reparandum',
       'root': 'root',
       'vocative': 'vocative',
       'xcomp': 'xcomp'
    }

    sdt_dir = Path('datasets/sdt')
    if not sdt_dir.exists():
        sdt_dir.mkdir()

    input_paths = [Path('datasets/sv_talbanken-ud-train.conllu'),
                   Path('datasets/sv_talbanken-ud-dev.conllu'),
                   Path('datasets/sv_talbanken-ud-test.conllu')]
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
        store = True
        for idx, line in enumerate(tqdm(lines)):
            if line.startswith('# text = '):
                doc = re.sub('# text = ', '', line)
                store = True
            elif line.startswith('#'):
                continue
            elif line == '':
                if tokens != [] and store:
                    data_dict = dict(ids=ids,
                                     doc=(doc.replace(' s k', ' s.k.')
                                             .replace('S k', 'S.k.')
                                             .replace(' bl a', ' bl.a.')
                                             .replace('Bl a', 'Bl.a.')
                                             .replace(' t o m', ' t.o.m.')
                                             .replace('T o m', 'T.o.m.')
                                             .replace(' fr o m', ' fr.o.m.')
                                             .replace('Fr o m', 'Fr.o.m.')
                                             .replace(' o s v', ' o.s.v.')
                                             .replace('O s v', 'O.s.v.')
                                             .replace(' d v s', ' d.v.s.')
                                             .replace('D v s', 'D.v.s.')
                                             .replace(' m fl', ' m.fl.')
                                             .replace('M fl', 'M.fl.')
                                             .replace(' t ex', ' t.ex.')
                                             .replace('T ex', 'T.ex.')
                                             .replace(' f n', ' f.n.')
                                             .replace('F n', 'F.n.')),
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
                data = line.split('\t')
                ids.append(data[0])
                tokens.append(data[1].replace('s k', 's.k.')
                                     .replace('S k', 'S.k.')
                                     .replace('t o m', 't.o.m.')
                                     .replace('T o m', 'T.o.m.')
                                     .replace('fr o m', 'fr.o.m.')
                                     .replace('Fr o m', 'Fr.o.m.')
                                     .replace('bl a', 'bl.a.')
                                     .replace('Bl a', 'Bl.a.')
                                     .replace('m fl', 'm.fl.')
                                     .replace('M fl', 'M.fl.')
                                     .replace('o s v', 'o.s.v.')
                                     .replace('O s v', 'O.s.v.')
                                     .replace('d v s', 'd.v.s.')
                                     .replace('D v s', 'D.v.s.')
                                     .replace('t ex', 't.ex.')
                                     .replace('T ex', 'T.ex.')
                                     .replace('f n', 'f.n.')
                                     .replace('F n', 'F.n.'))
                pos_tags.append(data[3])
                heads.append(data[6])
                try:
                    deps.append(dep_conversion_dict[data[7]])
                except KeyError:
                    store = False


def process_norne_nn():
    from pathlib import Path
    import json
    from tqdm.auto import tqdm
    import re

    ner_conversion_dict = {'O': 'O',
                           'B-LOC': 'B-LOC',
                           'I-LOC': 'I-LOC',
                           'B-PER': 'B-PER',
                           'I-PER': 'I-PER',
                           'B-ORG': 'B-ORG',
                           'I-ORG': 'I-ORG',
                           'B-MISC': 'B-MISC',
                           'I-MISC': 'I-MISC',
                           'B-GPE_LOC': 'B-LOC',
                           'I-GPE_LOC': 'I-LOC',
                           'B-GPE_ORG': 'B-ORG',
                           'I-GPE_ORG': 'I-ORG',
                           'B-PROD': 'B-MISC',
                           'I-PROD': 'I-MISC',
                           'B-DRV': 'B-MISC',
                           'I-DRV': 'I-MISC',
                           'B-EVT': 'B-MISC',
                           'I-EVT': 'I-MISC'}

    dep_conversion_dict = {
       'acl': 'acl',
       'acl:relcl': 'acl',
       'acl:cleft': 'acl',
       'advcl': 'advcl',
       'advmod': 'advmod',
       'advmod:emph': 'advmod',
       'advmod:lmod': 'advmod',
       'amod': 'amod',
       'appos': 'appos',
       'aux': 'aux',
       'aux:pass': 'aux',
       'case': 'case',
       'cc': 'cc',
       'cc:preconj': 'cc',
       'ccomp': 'ccomp',
       'clf': 'clf',
       'compound': 'compound',
       'compound:lvc': 'compound',
       'compound:prt': 'compound',
       'compound:redup': 'compound',
       'compound:svc': 'compound',
       'conj': 'conj',
       'cop': 'cop',
       'csubj': 'csubj',
       'csubj:pass': 'csubj',
       'dep': 'dep',
       'det': 'det',
       'det:numgov': 'det',
       'det:nummod': 'det',
       'det:poss': 'det',
       'discourse': 'discourse',
       'dislocated': 'dislocated',
       'expl': 'expl',
       'expl:impers': 'expl',
       'expl:pass': 'expl',
       'expl:pv': 'expl',
       'fixed': 'fixed',
       'flat': 'flat',
       'flat:foreign': 'flat',
       'flat:name': 'flat',
       'goeswith': 'goeswith',
       'iobj': 'iobj',
       'list': 'list',
       'mark': 'mark',
       'nmod': 'nmod',
       'nmod:poss': 'nmod',
       'nmod:tmod': 'nmod',
       'nsubj': 'nsubj',
       'nsubj:pass': 'nsubj',
       'nummod': 'nummod',
       'nummod:gov': 'nummod',
       'obj': 'obj',
       'obl': 'obl',
       'obl:agent': 'obl',
       'obl:arg': 'obl',
       'obl:lmod': 'obl',
       'obl:loc': 'obl',
       'obl:tmod': 'obl',
       'orphan': 'orphan',
       'parataxis': 'parataxis',
       'punct': 'punct',
       'reparandum': 'reparandum',
       'root': 'root',
       'vocative': 'vocative',
       'xcomp': 'xcomp'
    }

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
                deps.append(dep_conversion_dict[data[7]])
                tag = data[9].replace('name=', '').split('|')[-1]
                ner_tags.append(ner_conversion_dict[tag])


def process_norne_nb():
    from pathlib import Path
    import json
    from tqdm.auto import tqdm
    import re

    ner_conversion_dict = {'O': 'O',
                           'B-LOC': 'B-LOC',
                           'I-LOC': 'I-LOC',
                           'B-PER': 'B-PER',
                           'I-PER': 'I-PER',
                           'B-ORG': 'B-ORG',
                           'I-ORG': 'I-ORG',
                           'B-MISC': 'B-MISC',
                           'I-MISC': 'I-MISC',
                           'B-GPE_LOC': 'B-LOC',
                           'I-GPE_LOC': 'I-LOC',
                           'B-GPE_ORG': 'B-ORG',
                           'I-GPE_ORG': 'I-ORG',
                           'B-PROD': 'B-MISC',
                           'I-PROD': 'I-MISC',
                           'B-DRV': 'B-MISC',
                           'I-DRV': 'I-MISC',
                           'B-EVT': 'B-MISC',
                           'I-EVT': 'I-MISC'}

    dep_conversion_dict = {
       'acl': 'acl',
       'acl:relcl': 'acl',
       'acl:cleft': 'acl',
       'advcl': 'advcl',
       'advmod': 'advmod',
       'advmod:emph': 'advmod',
       'advmod:lmod': 'advmod',
       'amod': 'amod',
       'appos': 'appos',
       'aux': 'aux',
       'aux:pass': 'aux',
       'case': 'case',
       'cc': 'cc',
       'cc:preconj': 'cc',
       'ccomp': 'ccomp',
       'clf': 'clf',
       'compound': 'compound',
       'compound:lvc': 'compound',
       'compound:prt': 'compound',
       'compound:redup': 'compound',
       'compound:svc': 'compound',
       'conj': 'conj',
       'cop': 'cop',
       'csubj': 'csubj',
       'csubj:pass': 'csubj',
       'dep': 'dep',
       'det': 'det',
       'det:numgov': 'det',
       'det:nummod': 'det',
       'det:poss': 'det',
       'discourse': 'discourse',
       'dislocated': 'dislocated',
       'expl': 'expl',
       'expl:impers': 'expl',
       'expl:pass': 'expl',
       'expl:pv': 'expl',
       'fixed': 'fixed',
       'flat': 'flat',
       'flat:foreign': 'flat',
       'flat:name': 'flat',
       'goeswith': 'goeswith',
       'iobj': 'iobj',
       'list': 'list',
       'mark': 'mark',
       'nmod': 'nmod',
       'nmod:poss': 'nmod',
       'nmod:tmod': 'nmod',
       'nsubj': 'nsubj',
       'nsubj:pass': 'nsubj',
       'nummod': 'nummod',
       'nummod:gov': 'nummod',
       'obj': 'obj',
       'obl': 'obl',
       'obl:agent': 'obl',
       'obl:arg': 'obl',
       'obl:lmod': 'obl',
       'obl:loc': 'obl',
       'obl:tmod': 'obl',
       'orphan': 'orphan',
       'parataxis': 'parataxis',
       'punct': 'punct',
       'reparandum': 'reparandum',
       'root': 'root',
       'vocative': 'vocative',
       'xcomp': 'xcomp'
    }

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
                deps.append(dep_conversion_dict[data[7]])
                tag = data[9].replace('name=', '').split('|')[-1]
                ner_tags.append(ner_conversion_dict[tag])


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

    Path('datasets/twitter_sent').mkdir(exist_ok=True)

    input_path = Path('datasets/twitter_sent/twitter_sent.csv')
    output_paths = [Path('datasets/twitter_sent/train.jsonl'),
                    Path('datasets/twitter_sent/test.jsonl')]

    df = pd.read_csv(input_path, header=0).dropna()

    train = df.query('part == "train"')
    test = df.query('part == "test"')

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    for split, output_path in zip([train, test], output_paths):
        for idx, row in tqdm(split.iterrows()):
            data_dict = dict(tweet_id=row.twitterid, label=row.polarity)
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

    input_path = Path('datasets/angry_tweets/angry_tweets.csv')
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

    conversion_dict = {
       'acl': 'acl',
       'acl:relcl': 'acl',
       'acl:cleft': 'acl',
       'advcl': 'advcl',
       'advmod': 'advmod',
       'advmod:emph': 'advmod',
       'advmod:lmod': 'advmod',
       'amod': 'amod',
       'appos': 'appos',
       'aux': 'aux',
       'aux:pass': 'aux',
       'case': 'case',
       'cc': 'cc',
       'cc:preconj': 'cc',
       'ccomp': 'ccomp',
       'clf': 'clf',
       'compound': 'compound',
       'compound:lvc': 'compound',
       'compound:prt': 'compound',
       'compound:redup': 'compound',
       'compound:svc': 'compound',
       'conj': 'conj',
       'cop': 'cop',
       'csubj': 'csubj',
       'csubj:pass': 'csubj',
       'dep': 'dep',
       'det': 'det',
       'det:numgov': 'det',
       'det:nummod': 'det',
       'det:poss': 'det',
       'discourse': 'discourse',
       'dislocated': 'dislocated',
       'expl': 'expl',
       'expl:impers': 'expl',
       'expl:pass': 'expl',
       'expl:pv': 'expl',
       'fixed': 'fixed',
       'flat': 'flat',
       'flat:foreign': 'flat',
       'flat:name': 'flat',
       'goeswith': 'goeswith',
       'iobj': 'iobj',
       'list': 'list',
       'mark': 'mark',
       'nmod': 'nmod',
       'nmod:poss': 'nmod',
       'nmod:tmod': 'nmod',
       'nsubj': 'nsubj',
       'nsubj:pass': 'nsubj',
       'nummod': 'nummod',
       'nummod:gov': 'nummod',
       'obj': 'obj',
       'obl': 'obl',
       'obl:agent': 'obl',
       'obl:arg': 'obl',
       'obl:lmod': 'obl',
       'obl:loc': 'obl',
       'obl:tmod': 'obl',
       'orphan': 'orphan',
       'parataxis': 'parataxis',
       'punct': 'punct',
       'reparandum': 'reparandum',
       'root': 'root',
       'vocative': 'vocative',
       'xcomp': 'xcomp'
    }

    dataset_dir = Path('datasets/dane')
    if not dataset_dir.exists():
        dataset_dir.mkdir()

    input_paths = [Path('datasets/ddt.train.conllu'),
                   Path('datasets/ddt.dev.conllu'),
                   Path('datasets/ddt.test.conllu')]
    output_paths = [Path('datasets/dane/train.jsonl'),
                    Path('datasets/dane/val.jsonl'),
                    Path('datasets/dane/test.jsonl')]

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
                deps.append(conversion_dict[data[7]])
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
    process_twitter_sent_sentiment()
