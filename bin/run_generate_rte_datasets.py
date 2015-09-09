import os

from model.utils import get_dataset


_entailment_map = \
    {
        'for': 'ENTAILMENT',
        'against': 'CONTRADICTION',
        'observing': 'UNKNOWN'
    }


def _generate_xml(df):
    xml = '<?xml version="1.0" encoding="UTF-8"?>'
    xml += '<entailment-corpus lang="EN">\n'
    for idx, row in df.iterrows():
        xml += '<pair id="{0:d}" entailment="{1:s}" task="IE">\n'.format(idx,
                                                                         _entailment_map[row.articleHeadlineStance])
        xml += '<t>{0:s}</t>\n'.format(row.articleHeadline)
        xml += '<h>{0:s}</h>\n'.format(row.claimHeadline)
        xml += '</pair>\n'
    xml += '</entailment-corpus>\n'
    return xml

_dataset_files = ['url-versions-2015-06-14-clean-test.csv', 'url-versions-2015-06-14-clean-train.csv',
                  'url-versions-2015-06-14-clean-train-lite.csv']

if __name__ == '__main__':
    for ds_filename in _dataset_files:
        df = get_dataset(ds_filename)
        output_filename = '{0:s}-rte.xml'.format(ds_filename.split('.')[0])
        with open(os.path.join('..', 'data', 'emergent', output_filename), 'w') as f:
            f.write(_generate_xml(df))