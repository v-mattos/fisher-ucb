"""Failure cases for the publication boundary, not duplicated solver formulas."""
import csv
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from evidence import Evidence
from reproduce_paper import protect,ensure_reference
from runner import run_job
import reproduce_paper


class ReferenceExtractionTests(unittest.TestCase):
    def test_refresh_replaces_stale_data_and_removes_obsolete_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'data').mkdir()
            reference=root/'data/reference';archive=root/'data/reference.zip'
            with patch.object(reproduce_paper,'ROOT',root),patch.object(reproduce_paper,'REFERENCE',reference):
                with zipfile.ZipFile(archive,'w') as z:
                    z.writestr('reference/value.txt','first')
                    z.writestr('reference/obsolete.txt','old')
                ensure_reference()
                with zipfile.ZipFile(archive,'w') as z:z.writestr('reference/value.txt','updated')
                ensure_reference()
                self.assertEqual((reference/'value.txt').read_text(),'first')
                ensure_reference(refresh=True)
                self.assertEqual((reference/'value.txt').read_text(),'updated')
                self.assertFalse((reference/'obsolete.txt').exists())

    def test_failed_refresh_preserves_extracted_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);reference=root/'data/reference';reference.mkdir(parents=True)
            (reference/'value.txt').write_text('keep')
            (root/'data/reference.zip').write_bytes(b'not a ZIP archive')
            with patch.object(reproduce_paper,'ROOT',root),patch.object(reproduce_paper,'REFERENCE',reference):
                with self.assertRaises(zipfile.BadZipFile):ensure_reference(refresh=True)
            self.assertEqual((reference/'value.txt').read_text(),'keep')


class PublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_reference()

    def copy_group(self,temporary):
        source=ROOT/'data/reference/statistics/output/results'
        target=Path(temporary)/'statistics/output/results';target.mkdir(parents=True)
        for suffix in ['.csv','_per_seed.csv','_metadata.json']:
            shutil.copy2(source/('main_bernoulli'+suffix),target/('main_bernoulli'+suffix))
        return target

    def test_missing_seed_is_rejected_before_checkpoint_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            target=self.copy_group(temporary);p=target/'main_bernoulli_per_seed.csv'
            lines=p.read_text().splitlines(True);p.write_text(''.join([lines[0],*lines[2:]]))
            with self.assertRaisesRegex(ValueError,'Seed coverage'):
                Evidence(temporary).validate()

    def test_missing_policy_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            target=self.copy_group(temporary);p=target/'main_bernoulli_per_seed.csv'
            with p.open(newline='') as f:rows=list(csv.DictReader(f))
            with p.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
                w.writerows(r for r in rows if r['policy']!='UCB1')
            with self.assertRaisesRegex(ValueError,'Missing/extra policies'):
                Evidence(temporary).validate()

    def test_duplicate_seed_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            target=self.copy_group(temporary);p=target/'main_bernoulli_per_seed.csv'
            lines=p.read_text().splitlines(True);p.write_text(''.join(lines+[lines[1]]))
            with self.assertRaisesRegex(ValueError,'Seed coverage'):
                Evidence(temporary).validate()

    def test_approved_profile_detects_missing_experiment(self):
        profile=json.loads((ROOT/'data/reference/publication_profile.json').read_text())
        with tempfile.TemporaryDirectory() as temporary:
            self.copy_group(temporary)
            with self.assertRaisesRegex(ValueError,'Missing/extra statistical experiments'):
                Evidence(temporary).validate_profile(profile)

    def test_reference_overlap_is_rejected(self):
        reference=ROOT/'data/reference'
        for destination in [reference,reference/'new',ROOT/'data',ROOT]:
            with self.assertRaises(ValueError):protect(destination,reference)

    def test_saved_seeds_resume_despite_old_source_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec=dict(experiment='resume',T=5,policy='fixture')
            calls=[]
            def execute(seed):
                calls.append(seed)
                return np.arange(5,dtype=float)
            run_job(temporary,spec,1,execute)
            manifest=next(Path(temporary).glob('resume/*/manifest.json'))
            metadata=json.loads(manifest.read_text())
            metadata.update(source='previous code version',environment={'python':'previous version'})
            manifest.write_text(json.dumps(metadata))
            saved=manifest.with_name('seed_0.npz').read_bytes()
            records=run_job(temporary,spec,2,execute)
            self.assertEqual(calls,[0,1])
            self.assertEqual(saved,manifest.with_name('seed_0.npz').read_bytes())
            self.assertEqual(json.loads(manifest.read_text()),metadata)
            self.assertEqual(float(records[1]['final']),4.)


if __name__=='__main__':unittest.main()
