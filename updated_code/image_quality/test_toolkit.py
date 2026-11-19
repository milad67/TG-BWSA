"""Scientific invariants plus a small synthetic end-to-end recorder check.

Synthetic outputs are temporary and are never study results.
Run with the same Python environment that has numpy/scipy/torch installed.
"""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from degradation import CONDITIONS, apply_condition, foreground_from_clean, noise_seed, to_chwd

HERE=Path(__file__).resolve().parent
ORDER=["FLAIR","T1ce","T1","T2"]


class TransformTests(unittest.TestCase):
    def setUp(self):
        self.x=np.full((4,40,36,32),-1.25,dtype=np.float32)
        self.x[:,3:-3,3:-3,3:-3]=np.random.default_rng(12).normal(size=(4,34,30,26)).astype(np.float32)
        self.fg,self.info=foreground_from_clean(self.x)

    def test_clean_exact_and_input_unmodified(self):
        before=self.x.copy()
        y=apply_condition(self.x,self.fg,CONDITIONS[0],ORDER,"case1")
        self.assertTrue(np.array_equal(y,self.x))
        self.assertFalse(np.shares_memory(y,self.x))
        for cond in CONDITIONS[1:]:
            out=apply_condition(self.x,self.fg,cond,ORDER,"case1")
            channel=ORDER.index(cond.modality)
            self.assertTrue(np.array_equal(out[np.arange(4)!=channel],self.x[np.arange(4)!=channel]))
            self.assertTrue(np.array_equal(out[:,~self.fg],self.x[:,~self.fg]))
        self.assertTrue(np.array_equal(self.x,before))

    def test_same_noise_after_other_calls_and_severity_pair(self):
        low=next(c for c in CONDITIONS if c.name=="noise_T1ce_010")
        high=next(c for c in CONDITIONS if c.name=="noise_T1ce_020")
        a=apply_condition(self.x,self.fg,low,ORDER,"case1")
        apply_condition(self.x,self.fg,high,ORDER,"other_case")
        b=apply_condition(self.x,self.fg,low,ORDER,"case1")
        h=apply_condition(self.x,self.fg,high,ORDER,"case1")
        self.assertTrue(np.array_equal(a,b))
        delta=(a-self.x)[1,self.fg]
        self.assertAlmostEqual(float(delta.std()),0.10,delta=0.003)
        self.assertAlmostEqual(float(delta.mean()),0.0,delta=0.003)
        np.testing.assert_allclose((h-self.x)[1,self.fg],2*delta,atol=5e-7,rtol=1e-5)
        self.assertNotEqual(noise_seed("case1","T1ce"),noise_seed("case1","FLAIR"))

    def test_blur_is_three_dimensional(self):
        from scipy.ndimage import gaussian_filter
        cond=next(c for c in CONDITIONS if c.name=="blur_FLAIR_100")
        y=apply_condition(self.x,self.fg,cond,ORDER,"case1")
        expected=gaussian_filter(self.x[0],sigma=(1,1,1),mode="reflect",truncate=4.0)
        np.testing.assert_array_equal(y[0,self.fg],expected[self.fg])

    def test_layout_and_foreground_nonzero_background(self):
        np.testing.assert_array_equal(to_chwd(np.moveaxis(self.x,0,-1),"HWDC"),self.x)
        self.assertAlmostEqual(self.info["background_values_by_channel"][0],-1.25)
        self.assertFalse(self.fg[0,0,0])
        self.assertTrue(self.fg[10,10,10])
        with self.assertRaises(ValueError):
            foreground_from_clean(self.x,"zero_background")


class IntegrationTest(unittest.TestCase):
    def test_three_isolated_projects_and_complete_collection(self):
        import torch
        with tempfile.TemporaryDirectory(prefix="rfnet_degradation_synthetic_") as tmp:
            root=Path(tmp)
            data=root/"processed"
            (data/"vol").mkdir(parents=True)
            (data/"seg").mkdir()
            ids=["synthetic_case_1","synthetic_case_2"]
            for n,case in enumerate(ids):
                x=np.full((12,12,12,4),-1.25,dtype=np.float32)
                x[2:10,2:10,2:10]=np.random.default_rng(10+n).normal(size=(8,8,8,4))
                y=np.zeros((12,12,12),dtype=np.uint8)
                y[4:7,4:7,4:7]=3
                np.save(data/"vol"/(case+"_vol.npy"),x)
                np.save(data/"seg"/(case+"_seg.npy"),y)
            test=root/"test.txt"
            test.write_text("\n".join(ids)+"\n")
            models=[]
            for name in ("RFNet","RFNet_BWSA","RFNet_TG_BWSA"):
                repo=root/name
                repo.mkdir()
                (repo/"models.py").write_text('''import torch
class Model(torch.nn.Module):
    def __init__(self,num_cls=4):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.ones(1))
        self.is_training=False
    def forward(self,x,mask):
        assert bool(mask.all())
        labels=torch.clamp(((x[:,0]+1.25)*self.weight).long(),0,3)
        return torch.nn.functional.one_hot(labels,4).permute(0,4,1,2,3).float()
''')
                (repo/"predict.py").write_text('''import numpy as np
import torch
def softmax_output_dice_class4(output,target):
    values=[]
    for p,t in zip(output,target):
        row=[]
        for a,b in [(p>0,t>0),((p==1)|(p==3),(t==1)|(t==3)),(p==3,t==3)]:
            row.append(float((2*(a&b).sum()+1e-8)/(a.sum()+b.sum()+1e-8)))
        values.append(row+[row[2]])
    return np.zeros((len(values),3)),np.asarray(values)
def test_softmax(loader,model,dataname=None,feature_mask=None,mask_name=None):
    assert not torch.is_grad_enabled()
    model.eval()
    rows=[]
    for x,y,names in loader:
        mask=torch.tensor([feature_mask],dtype=torch.bool)
        pred=model(x,mask).argmax(1)
        _,values=softmax_output_dice_class4(pred,y)
        rows.append(values[0])
    return np.mean(rows,axis=0)
''')
                ckpt=repo/"checkpoint.pth"
                torch.save({"state_dict":{"module.weight":torch.ones(1)}},ckpt)
                models.append({"name":name,"repo_dir":str(repo),"model_module":"models","model_class":"Model",
                    "model_kwargs":{"num_cls":4},"checkpoint":str(ckpt),"checkpoint_key":"state_dict",
                    "state_prefix_to_strip":"module.","predict_module":"predict","evaluation_function":"test_softmax",
                    "evaluation_kwargs":{"dataname":"BRATS2023"},"metric_function":"softmax_output_dice_class4",
                    "metric_return_index":1,"dice_indices":[0,1,3]})
            train_ids = root/"train_ids.txt"
            val_ids = root/"val_ids.txt"
            train_ids.write_text("training_patient\n")
            val_ids.write_text("validation_patient\n")
            config={"train_file":str(train_ids),"val_file":str(val_ids),"dataset":"BraTS2023","training_regime":"full_input","data_root":str(data),
                "test_file":str(test),"expected_cases":2,"checkpoint_split_confirmed":True,
                "layout":"HWDC","channel_order":ORDER,"channel_order_confirmed":True,
                "foreground_method":"constant_background","foreground_dir":None,"noise_base_seed":20261113,
                "evaluation_seed":1024,"output_dir":str(root/"results"),"models":models}
            cfg=root/"config.json"
            cfg.write_text(json.dumps(config))
            log=root/"test.log"
            with log.open("w") as f:
                subprocess.run([sys.executable,str(HERE/"run_degradation.py"),"--config",str(cfg),"--device","cpu"],
                               check=True,stdout=f,stderr=subprocess.STDOUT)
                subprocess.run([sys.executable,str(HERE/"collect_results.py"),"--config",str(cfg)],
                               check=True,stdout=f,stderr=subprocess.STDOUT)
                subprocess.run([sys.executable,str(HERE/"preview.py"),"--config",str(cfg),"--case-id",ids[0],
                                "--out",str(root/"qa")],check=True,stdout=f,stderr=subprocess.STDOUT)
            with open(root/"results"/"degradation_per_case.csv") as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),54)
            self.assertEqual(len({(r["case_id"],r["model"],r["condition"]) for r in rows}),54)
            self.assertTrue(all(r["hd95_wt_mm"]=="" for r in rows))
            self.assertTrue((root/"results_results.zip").is_file())
            self.assertTrue((root/"qa"/(ids[0]+".png")).is_file())
            # Fail rather than silently overwriting a completed experiment.
            with log.open("a") as f:
                retry=subprocess.run([sys.executable,str(HERE/"run_degradation.py"),"--config",str(cfg),
                                      "--model","RFNet","--device","cpu"],stdout=f,stderr=subprocess.STDOUT)
            self.assertNotEqual(retry.returncode,0)


if __name__=="__main__":
    unittest.main(verbosity=2)
