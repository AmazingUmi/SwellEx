"""Controlled evidence/alias contracts; render tests execute only on haiqin."""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from scripts.data_scale_pack.localization_networks.common.scaling import resolve_capacity
from scripts.data_scale_pack.localization_networks.workflow import budget_study as contract
from scripts.data_scale_pack.localization_networks.workflow.config import load_config
from scripts.data_scale_pack.localization_networks.workflow.remote import DEFAULT_CONFIG
from scripts.data_scale_pack.localization_networks.workflow.tasks import model_plan
from scripts.data_scale_pack.localization_networks.workflow.worker import expected_controlled_budget
from scripts.data_scale_pack.visualization.budget_results import load_budget_run, load_budget_study, POLICIES
from scripts.data_scale_pack.visualization.plot_budget_study import ROOT, RENDERER_PATHS, renderer_provenance, render_budget_run, render_budget_study
from scripts.data_scale_pack.visualization.results import read_predictions, summarize
from scripts.data_scale_pack.visualization.study_results import digest
from scripts.data_scale_pack.visualization.tests.test_study_results import write, split, coverage
from scripts.run_identity import make_run_id


def fixture(root):
    directory=root/'budget-fixture'
    directory.mkdir()
    source_path='scripts/data_scale_pack/localization_networks/workflow/scm_capacity_calibration_v2.json'
    raw=(ROOT/source_path).read_bytes()
    artifact=json.loads(raw)
    (directory/'calibration.json').write_bytes(raw)
    config=load_config(DEFAULT_CONFIG)
    config['training'].update(epochs=2,batch_size=64,base_channels=2)
    plan={'schema_version':1,'base_config':config,'fractions':[.5,1.0],'seeds':[2029,2030],'method':'SCM',
          'epoch_budget':2,'update_budget':8,'validation_interval_updates':4,
          'calibration':{'path':source_path,'sha256':hashlib.sha256(raw).hexdigest()}}
    inventory=[{'path':p,'size':(ROOT/p).stat().st_size,'sha256':hashlib.sha256((ROOT/p).read_bytes()).hexdigest()} for p in RENDERER_PATHS]
    inventory.append({'path':source_path,'size':len(raw),'sha256':plan['calibration']['sha256']})
    ledger={'schema_version':1,'kind':contract.KIND,'pack':'data_scale_pack','study_id':directory.name,'created_at':'fixture','updated_at':'fixture',
            'status':'PASS','plan':plan,'plan_sha256':digest(plan,compact=True),'source_files':inventory,'source_sha256':digest(inventory,compact=True),
            'calibration_artifact':artifact,'children':[]}
    identity={'schema_version':1,'pack':'data_scale_pack','inputs':[{'path':'fixture','size':32,'sha256':'a'*64}]}
    identity['cache_key']=digest(identity)
    jobs=model_plan('budget',contract.selection_for(plan))
    for seed,fraction,arm in contract.matrix(plan):
        effective=contract.child_config(plan,seed,fraction,arm)
        run_id=make_run_id('data_scale_pack',effective,contract.selection_for(plan))
        run=directory/"children"/run_id
        count=int(256*fraction)
        budget=expected_controlled_budget(effective['training'],count)
        child={'run_id':run_id,'seed':seed,'train_fraction':fraction,'execution_arm':arm,'config_sha256':digest(effective),'status':'PASS',
               'attempted_at':'fixture','observed_budget':budget,'shared_full_verified':fraction==1}
        ledger['children'].append(child)
        request={'schema_version':1,'pack':'data_scale_pack','run_id':run_id,'suite':'budget','selection':contract.selection_for(plan),'config':effective,'use_cache':True}
        manifest={k:request[k] for k in ('schema_version','pack','run_id','suite','selection')}
        manifest.update(files=inventory,source_sha256=ledger['source_sha256'],config_sha256=child['config_sha256'])
        report={**manifest,'status':'PASS','purpose':'controlled_epoch_vs_update_budget_with_frozen_validation_calibration','model_plan':jobs,
                'calibration':plan['calibration'],'uv_lock_sha256':'f'*64,'dataset_cache':{'cache_key':identity['cache_key']},'metrics':[],'checks':[]}
        train=split((i,float(i),float(i)/100) for i in range(1,count+1))
        val=split((i,float(i),float(i)/100) for i in range(257,261))
        test=split([(1001,1001.,2.),(1002,1002.,3.)])
        members={'schema_version':1,'train':train,'validation':val,'validation_coverage':coverage(train,val)}
        archive={'schema_version':1,'pack':'data_scale_pack','run_id':run_id,'models':[]}
        for index,job in enumerate(jobs):
            scaling=json.loads(json.dumps({**resolve_capacity(job['architecture'].removeprefix('complex_CNN_'),train_count=count,base_channels=2,scaling_policy=effective['scaling']),'calibration_status':'uncalibrated'}))
            sampling={'version':'temporal-block-prefix-v1','train_fraction':fraction,'candidate_train_count':256,'train_count':count,'validation_count':4,'block_rows':32,'seed':seed,
                      'source_identity_sha256':hashlib.sha256(','.join(str(i) for i in range(1,count+1)).encode()).hexdigest()}
            sampler={'version':'independent-torch-generators-v1','train_seed':seed+1000003,'validation_seed':seed+2000003,'batch_digest':'sha256-json-source-id-batches-v1'}
            scheduler={'policy':'cosine_per_update_v1','step_unit':'optimizer_update','total_updates':budget['total_optimizer_updates'],'initial_lr':.001,'eta_min':0.0}
            initial=hashlib.sha256(f'{seed}/{index}'.encode()).hexdigest()
            sidecar={'pack':'data_scale_pack','method':'SCM','model':job['model'],'scaling':scaling,'sampling':sampling,'training_inventory':members,
                     'model_config':{**{k:v for k,v in scaling.items() if k!='calibration_status'},'input_pairs':231,'input_freq_bins':22,'dropout':.15,'base_channels':2},
                     'budget_plan':budget,'sampler_plan':sampler,'scheduler_plan':scheduler,'initial_model_state_sha256':initial}
            history=[];lrhash=hashlib.sha256();batchhash=hashlib.sha256();next_lr=.001;seen=set();pass_hashes=[]
            steps,total=budget['validation_interval_updates'],budget['total_optimizer_updates']
            for record_index in range(1,budget['validation_checks']+1):
                first_lr=next_lr;recordhash=hashlib.sha256()
                for update in range((record_index-1)*steps+1,record_index*steps+1):
                    first=((update-1)%budget['steps_per_data_pass'])*64+1
                    ids=list(range(first,first+64));seen.update(ids)
                    encoded=(json.dumps(ids,separators=(',',':'))+'\n').encode();batchhash.update(encoded);recordhash.update(encoded)
                    if update%budget['steps_per_data_pass']==0:pass_hashes.append(batchhash.hexdigest())
                    last_lr=next_lr;lrhash.update((json.dumps(last_lr)+'\n').encode())
                    next_lr=((1+math.cos(math.pi*update/total))/(1+math.cos(math.pi*(update-1)/total)))*last_lr
                updates=record_index*steps
                val_score=.5-index*.08+(seed-2029)*.01+.01*(budget['validation_checks']-record_index)
                history.append({'epoch':record_index,'validation_index':record_index,'record_unit':'validation_checkpoint','loss_space':'km','lr':first_lr,'last_update_lr':last_lr,'next_lr':next_lr,
                                'train_loss':.01,'train_mae_km':.08,'train_rmse_km':.1,'val_loss':.02,'val_mae_km':val_score,'val_rmse_km':val_score,
                                'epoch_elapsed_s':.1,'optimizer_steps_epoch':steps,'optimizer_steps':updates,'sample_presentations_record':steps*64,'sample_presentations':updates*64,
                                'data_passes_completed':updates//budget['steps_per_data_pass'],'updates_in_current_pass':updates%budget['steps_per_data_pass'],
                                'unique_train_sources_seen':len(seen),'batch_sequence_sha256':batchhash.hexdigest(),'record_batch_sequence_sha256':recordhash.hexdigest(),
                                'lr_sequence_sha256':lrhash.hexdigest(),'seen_source_identity_sha256':hashlib.sha256(json.dumps(sorted(seen),separators=(',',':')).encode()).hexdigest()})
            output=run/'project/outputs/networks_results/SCM'/job['model']
            architecture_path=output/'train_outputs/fixture_architecture_1015_010203.json'
            write(architecture_path,sidecar);write(output/'train_outputs/fixture_history_1015_010203.json',history)
            error=.1*(index+1)
            csv=('sample,source_segment_idx,window_center_s,pred_range_km,true_range_km,abs_error_km\n'+f'0,1001,1001.0,{2+error:.6f},2.0,{error:.6f}\n1,1002,1002.0,{3+error:.6f},3.0,{error:.6f}\n').encode()
            path=output/'test_outputs/fixture_predictions_1015_010203.csv';path.parent.mkdir(parents=True);path.write_bytes(csv)
            best,last=history[-1],history[-1]
            training={'best_epoch':best['epoch'],'best_validation_index':best['validation_index'],'best_optimizer_step':best['optimizer_steps'],'best_val_rmse_km':best['val_rmse_km'],'best_val_mae_km':best['val_mae_km'],
                      'best_train_loss':.01,'best_train_mae_km':.08,'best_train_rmse_km':.1,'total_train_wall_s':.4,'sum_epoch_elapsed_s':.2,'optimizer_steps':total,'parameter_count':scaling['parameter_count'],
                      'train_identity_sha256':train['identity_sha256'],'val_identity_sha256':val['identity_sha256'],'test_identity_sha256':test['identity_sha256'],
                      'history_record_unit':'validation_checkpoint','validation_checks':2,'budget_plan':budget,'sampler_plan':sampler,'scheduler_plan':scheduler,'initial_model_state_sha256':initial,
                      'final_batch_sequence_sha256':last['batch_sequence_sha256'],'final_lr_sequence_sha256':last['lr_sequence_sha256'],'final_seen_source_identity_sha256':last['seen_source_identity_sha256'],
                      'data_pass_batch_sequence_sha256':pass_hashes,'sample_presentations':total*64,'unique_train_sources_seen':count}
            metric={**summarize(read_predictions(csv)),**training,'method':'SCM','model':job['model'],'epochs':effective['training']['epochs'],'scaling':scaling,'train_fraction':fraction,
                    'architecture_path':architecture_path.relative_to(run).as_posix(),'training_inventory_path':'reports/training_inventory.json'}
            report['metrics'].append(metric);report['checks'].append({'name':f"evaluate_SCM_{job['model']}",'status':'PASS'})
            archive['models'].append({'method':'SCM','model':job['model'],'seed':seed,'train_fraction':fraction,'train_indices':list(range(count)),'val_indices':[256,257,258,259],
                                     'source_identity':{'version':1,'train_sources':[{'path':'/fixture/train.h5','size_bytes':32,'sha256':'a'*64}],'validation_sources':[{'path':'/fixture/train.h5','size_bytes':32,'sha256':'a'*64}]},
                                     'sampling':sampling,'training_inventory':members,'test':test,'test_coverage':coverage(train,test),'training_metrics':training})
        for filename,value in (('request.json',request),('manifest.json',manifest),('report.json',report),('reports/training_inventory.json',archive),('data_identity.json',identity)):
            write(run/filename,value)
    ledger['analysis_cells']=contract.analysis_cells(plan,ledger['children'])
    path=directory/'study.json';write(path,ledger)
    return path,ledger


class BudgetEvidenceTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory(prefix='swellex-budget-');self.addCleanup(temp.cleanup)
        self.root=Path(temp.name);self.path,self.study=fixture(self.root)

    def test_unique_logical_and_virtual_counts(self):
        data=load_budget_study(self.path)
        self.assertEqual(data.status,'PASS')
        self.assertEqual((len(data.unique_rows),len(data.rows),len(data.policy_rows)),(30,40,16))
        self.assertEqual(len(data.histories),60)
        shared=[r for r in data.paired_rows if r['comparison']=='updates-minus-epochs' and r['train_fraction']==1]
        self.assertTrue(all(r['same_observation'] and r['test_rmse_km_difference']==0 and r['total_train_wall_s_difference']==0 for r in shared))
        self.assertTrue(all(r['n_seeds']==2 for r in data.summaries))

    def test_validation_reference_excludes_baseline_and_never_uses_test_ranking(self):
        data=load_budget_study(self.path)
        self.assertTrue(all(r['selected_architecture']=='complex_CNN_xlarge' for r in data.selection))
        self.assertTrue(any(r['test_rmse_km_difference']<0 for r in data.paired_rows if r['comparison']=='policy-minus-fresh-validation-reference'))

    def test_same_profile_policy_views_do_not_create_independent_replicates(self):
        self.study['calibration_artifact']['policy']=dict(contract.OLD_POLICY)
        # Exercise mapping without rewriting immutable calibration provenance.
        from scripts.data_scale_pack.visualization.budget_results import policy_views
        data=load_budget_study(self.path)
        views=policy_views(data.rows,self.study['plan'],self.study['calibration_artifact'])
        differences=[r for r in views[-1] if r['comparison']=='calibrated-minus-original']
        self.assertTrue(all(r['same_observation'] and r['test_rmse_km_difference']==0 for r in differences))

    def test_missing_child_keeps_planned_analysis_cells(self):
        self.study['children'][0]['status']='FAIL';self.study['children'][0]['error']='fixture failure';write(self.path,self.study)
        data=load_budget_study(self.path)
        self.assertEqual(data.status,'PARTIAL');self.assertEqual(len(data.rows),40)
        self.assertEqual(sum(r['status']=='NOT_RUN' for r in data.rows),5)
        self.assertTrue(any(r['status']=='NOT_RUN' for r in data.paired_rows))

    def test_shared_alias_tampering_is_rejected(self):
        self.study['analysis_cells'][-1]['shared']=False;write(self.path,self.study)
        with self.assertRaisesRegex(ValueError,'shared observation mapping'):load_budget_study(self.path)

    def test_calibration_byte_tampering_is_rejected(self):
        with (self.path.parent/'calibration.json').open('ab') as handle:handle.write(b' ')
        with self.assertRaisesRegex(ValueError,'Calibration bytes'):load_budget_study(self.path)

    def test_history_update_counters_are_not_epochs(self):
        child=self.study['children'][1];run=self.path.parent/"children"/child['run_id']
        path=next((run/'project/outputs/networks_results').rglob('*_history_*.json'))
        history=json.loads(path.read_text());history[0]['optimizer_steps']=1;write(path,history)
        with self.assertRaisesRegex(ValueError,'counters'):load_budget_run(run)

    def test_learning_rate_trajectory_tampering_is_rejected(self):
        run=self.path.parent/"children"/self.study['children'][0]['run_id'];path=next((run/'project/outputs/networks_results').rglob('*_history_*.json'))
        history=json.loads(path.read_text());history[0]['lr_sequence_sha256']='a'*64;write(path,history)
        with self.assertRaisesRegex(ValueError,'learning-rate sequence'):load_budget_run(run)

    def test_complete_pass_prefix_mismatch_between_arms_is_rejected(self):
        child=self.study['children'][1];run=self.path.parent/"children"/child['run_id']
        report=json.loads((run/'report.json').read_text());archive=json.loads((run/'reports/training_inventory.json').read_text())
        # First completed pass is between update-arm validation points: history stays valid.
        for row,entry in zip(report['metrics'],archive['models']):
            row['data_pass_batch_sequence_sha256'][0]='b'*64
            entry['training_metrics']['data_pass_batch_sequence_sha256'][0]='b'*64
        write(run/'report.json',report);write(run/'reports/training_inventory.json',archive)
        with self.assertRaisesRegex(ValueError,'complete-pass batch-stream prefixes'):load_budget_study(self.path)

    def test_renderer_drift_requires_explicit_update(self):
        files=copy.deepcopy(self.study['source_files']);files[0]['sha256']='c'*64
        with self.assertRaisesRegex(ValueError,'Renderer differs'):renderer_provenance(files,self.study['source_sha256'])
        p=renderer_provenance(files,self.study['source_sha256'],True)
        self.assertFalse(p['matches_training_snapshot']);self.assertFalse(p['retrained']);self.assertEqual(p['changed_paths'],[RENDERER_PATHS[0]])

    def test_render_budget_child_manifest_and_no_overwrite(self):
        run=self.path.parent/"children"/self.study['children'][0]['run_id'];output=self.root/'rendered'
        result=render_budget_run(run,output,dpi=80)
        self.assertEqual(result['status'],'PASS');self.assertEqual(len(result['figures']),11)
        self.assertTrue(all((output/p).is_file() for p in result['figures']))
        for item in result['files']:
            self.assertEqual(hashlib.sha256((output/item['path']).read_bytes()).hexdigest(),item['sha256'])
        with self.assertRaisesRegex(ValueError,'new directory'):render_budget_run(run,output,dpi=80)

    def test_render_budget_study_all_figures_and_alias_counts(self):
        output=self.root/'study-rendered'
        result=render_budget_study(self.path,output,dpi=80)
        self.assertEqual(result['status'],'PASS')
        self.assertEqual(len(result['figures']),14)
        self.assertEqual(result['counts']['planned_unique_training'],30)
        self.assertEqual(result['counts']['logical_analysis_cells'],8)
        self.assertEqual(result['counts']['logical_model_observations'],40)
        self.assertEqual(result['counts']['virtual_policy_views'],16)
        for item in result['files']:
            self.assertEqual(hashlib.sha256((output/item['path']).read_bytes()).hexdigest(),item['sha256'])

    def test_legacy_reader_still_rejects_budget_suite(self):
        from scripts.data_scale_pack.visualization.results import load_run
        with self.assertRaisesRegex(ValueError,'selected/smoke/experiment'):
            load_run(self.path.parent/"children"/self.study['children'][0]['run_id'])
