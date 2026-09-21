"""Host-only extraction/provenance guards; no build or runtime execution."""
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
import capture_temporal_linux as capture

class LinuxArchiveTests(unittest.TestCase):
    def test_actual_native_archive_has_identical_candidate_bytes(self):
        build=json.loads((capture.NATIVE/"build.json").read_bytes())
        members=capture.archive_members(capture.NATIVE/"source.zip",build)
        self.assertEqual(len(members),38)
        self.assertEqual(len([n for n in members if not n.startswith("capture/")]),30)
        self.assertTrue(all(capture.sha(capture.ROOT/n)==h for n,h in capture.PINNED.items()))

    def test_changed_missing_and_added_archive_members_rejected(self):
        build=json.loads((capture.NATIVE/"build.json").read_bytes())
        members=capture.archive_members(capture.NATIVE/"source.zip",build)
        for kind in ['changed','missing','added']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                changed=dict(members)
                if kind=='changed':changed['src/temporal_candidate.rs']+=b'\n'
                if kind=='missing':del changed['src/temporal_candidate.rs']
                if kind=='added':changed['src/unlisted.rs']=b'// not frozen\n'
                path=Path(tmp)/'source.zip'
                with zipfile.ZipFile(path,'w') as z:
                    for name,raw in changed.items():z.writestr(name,raw)
                with self.assertRaises(ValueError):capture.archive_members(path,build)

if __name__=='__main__':unittest.main()
