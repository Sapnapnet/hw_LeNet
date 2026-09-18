"""Read-only verification of the 340-record Level 2 handoff; no inference."""
import collections
import json
import prepare_member7_level2 as prep


def main():
    data = prep.DATA
    rows = prep.readcsv(data / 'level2_manifest.csv')
    labels = prep.readcsv(data / 'label.csv')
    sources = json.loads((data / 'source_manifest.json').read_text(encoding='utf8'))
    version = json.loads((data / 'dataset_version.json').read_text(encoding='utf8'))
    state = json.loads((data / 'processing_state.json').read_text(encoding='utf8'))
    ids = [r['sample_id'] for r in rows]
    assert len(ids) == len(set(ids)) == 340, 'Expected 340 unique sample IDs'
    assert collections.Counter(r['label'] for r in rows) == {str(i): 34 for i in range(10)}
    by_label = {r['sample_id']: r for r in labels}
    by_source = {r['sample_id']: r for r in sources['samples']}
    assert len(labels) == len(sources['samples']) == 340
    assert set(ids) == set(by_label) == set(by_source) == set(state['samples'])
    for path, expected in version['code_sha256'].items():
        assert prep.sha(prep.ROOT / path) == expected, 'Changed processing code: ' + path
    for r in rows:
        sid = r['sample_id']
        label = by_label[sid]
        assert r['label'] == str(by_source[sid]['label']) == label['label']
        assert r['source_relative'] == label['original_path'].removeprefix('original/')
        assert r['filename'] == label['filename'] == by_source[sid]['filename']
        assert prep.sha(prep.local(label['original_path'])) == r['input_sha256'] == by_source[sid]['sha256']
        assert r['pipeline_sha256'] == version['pipeline_sha256'] == state['samples'][sid]['pipeline_sha256']
        assert state['samples'][sid]['input_sha256'] == r['input_sha256']
        prep.verify(r)
    hashes = json.loads((prep.ROOT / 'results/member8_level2/delivery_files_sha256.json').read_text(encoding='utf8'))
    for path, expected in hashes.items():
        assert prep.sha(prep.ROOT / path) == expected, 'Changed handoff file: ' + path
    print('PASS: 340 records, 34/class; 680 PGM and 680 Fixed files; labels, paths, hashes and encoding verified.')
    print('Keep all 340 records, including duplicates and difficult images. No inference executed.')


if __name__ == '__main__':
    main()
