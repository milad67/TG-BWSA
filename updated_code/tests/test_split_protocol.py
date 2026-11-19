import argparse
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from split_protocol import check_partitions, read_cases, resolve_split_files, add_split_arguments, record_split_manifest, internal_validation
from make_internal_split import choose_internal_cases


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def lists(self,train,val,test):
        out={}
        for name,ids in zip(['train','val','test'],[train,val,test]):
            path=self.root/(name+'.txt');path.write_text('\n'.join(ids)+'\n');out[name]=str(path)
        return out
    def test_rejects_different_timepoints_of_same_patient(self):
        paths=self.lists(['BraTS-GLI-00001-000'],['BraTS-GLI-00001-001'],['BraTS-GLI-00002-000'])
        with self.assertRaisesRegex(ValueError,'patients'):check_partitions(paths)
    def test_rejects_duplicate_case(self):
        paths=self.lists(['a','a'],['b'],['c'])
        with self.assertRaises(ValueError):check_partitions(paths)
    def test_rejects_exact_test_overlap(self):
        paths=self.lists(['a'],['b'],['a'])
        with self.assertRaises(ValueError):check_partitions(paths)
    def test_explicit_lists_and_recorded_roles(self):
        paths=self.lists(['a'],['b'],['c'])
        p=argparse.ArgumentParser();add_split_arguments(p)
        args=p.parse_args(['--train-list',paths['train'],'--val-list',paths['val'],'--test-list',paths['test']])
        args.datapath=str(self.root);args.resume=None
        checked=resolve_split_files(args)
        record=record_split_manifest(checked,self.root/'run',1024)
        self.assertEqual(record['checkpoint_selection_partition'],'val')
        self.assertEqual(record['files']['test']['cases'],1)
        self.assertEqual(json.loads((self.root/'run/split_manifest.json').read_text()),record)
    def test_test_evaluation_requires_explicit_checkpoint(self):
        paths=self.lists(['a'],['b'],['c'])
        args=SimpleNamespace(datapath=str(self.root),train_list=paths['train'],val_list=paths['val'],test_list=paths['test'],val_every=1,evaluate_test=True,resume=None)
        with self.assertRaisesRegex(ValueError,'checkpoint'):resolve_split_files(args)
    def test_both_generated_resplits_are_patient_disjoint(self):
        for seed in [2026,2027]:
            path=ROOT.parent/f'data/splits/brats2023/split_{seed}'
            parts=check_partitions({role:path/(role+'.txt') for role in ['train','val','test']})
            self.assertEqual([len(parts[k]) for k in ['train','val','test']],[900,100,251])
            self.assertEqual(set(parts['train']+parts['val']),set(read_cases(path/'development.txt')))
            train,val=choose_internal_cases(read_cases(path/'development.txt'),parts['test'],seed)
            self.assertEqual(train,parts['train']);self.assertEqual(val,parts['val'])
    def test_selection_uses_the_internal_loader_and_three_regions(self):
        class Model:
            training=True
            module=SimpleNamespace(is_training=True)
            def eval(self):self.training=False
            def train(self,flag=True):self.training=flag
        model=Model();loader=object();seen=[]
        def evaluator(received_loader,model,**kwargs):
            self.assertIs(received_loader,loader);seen.append(kwargs['feature_mask'])
            return [.9,.8,.1,.7]
        fake_torch=SimpleNamespace(no_grad=nullcontext)
        args=SimpleNamespace(validation_mode='full',dice_indices='0,1,3',dataname='BRATS2023')
        with patch.dict(sys.modules,{'torch':fake_torch}):
            result=internal_validation(model,loader,evaluator,args,[[False]*4,[True]*4])
        self.assertAlmostEqual(result,.8);self.assertEqual(seen,[[True]*4]);self.assertTrue(model.training)


if __name__=='__main__':unittest.main()
