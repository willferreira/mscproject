import os

from model.utils import get_dataset

if __name__ == '__main__':
    df_clean_test = get_dataset('url-versions-2015-06-14-clean-test.csv')

    xml = '<entailment-corpus lang="EN">\n'
    for idx, row in df_clean_test.iterrows():
        xml += '<pair id="{0:d}">\n'.format(idx)
        xml += '<t>{0:s}</t>\n'.format(row.articleHeadline)
        xml += '<h>{0:s}</h>\n'.format(row.claimHeadline)
        xml += '</pair>\n'
    xml += '</entailment-corpus>\n'

    with open(os.path.join('..', 'data', 'emergent', 'url-versions-2015-06-14-clean-test-rte.xml'), 'w') as f:
        f.write(xml)