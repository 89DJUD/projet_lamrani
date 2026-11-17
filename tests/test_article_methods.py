import math
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from migrant_ai.perplexity import windows,causal_perplexity
from migrant_ai.paper_metrics import map_scores,STEREOTYPE_LABELS,HATE_LABELS,lexical_metrics,summarize
from migrant_ai.context_processing import process_context
from migrant_ai.study_protocol import configurations,load_dataset,plan

class PerplexityTests(unittest.TestCase):
    def test_uniform_distribution_has_known_perplexity(self):
        result=causal_perplexity(list(range(21)),lambda part,first:(len(part)-first)*math.log(4),8,3)
        self.assertAlmostEqual(result['perplexity'],4)
        self.assertEqual(result['perplexity_tokens'],20)
    def test_windows_score_every_token_once(self):
        for n in [2,5,8,9,17,50]:
            targets=[i for start,end,first in windows(n,8,3) for i in range(start+first,end)]
            self.assertEqual(targets,list(range(1,n)))
    def test_invalid_or_nonfinite_loss_is_rejected(self):
        with self.assertRaises(ValueError): list(windows(1,8,3))
        with self.assertRaises(ValueError): list(windows(5,8,8))
        with self.assertRaises(ValueError): causal_perplexity([1,2],lambda p,f:float('nan'))

class CategoryTests(unittest.TestCase):
    def test_config_labels_trim_whitespace(self):
        result=map_scores([{'label':'neutral','score':.1},{'label':'stereotype','score':.2},
                           {'label':' anti-stereotype','score':.7}],STEREOTYPE_LABELS)
        self.assertEqual(result['anti_stereotype_score'],.7)
    def test_missing_class_is_not_reconstructed(self):
        with self.assertRaises(ValueError): map_scores([{'label':'HATE','score':.6}],HATE_LABELS)
        with self.assertRaises(ValueError): map_scores([{'label':'LABEL_0','score':1}],HATE_LABELS)
    def test_recall_and_f1_are_distinct(self):
        scores=lexical_metrics('un deux','un deux trois quatre')
        self.assertAlmostEqual(scores['rouge1_recall'],.5)
        self.assertAlmostEqual(scores['rouge1_precision'],1)
        self.assertAlmostEqual(scores['rouge1'],2/3)

class ContextTests(unittest.TestCase):
    def test_filter_keeps_trace_and_source_summary(self):
        passages=[{'source':'a.pdf','page':1,'text':'keep'}, {'source':'b.pdf','page':2,'text':'remove'}]
        classify=lambda text:{'stereotype_score':.9 if text=='remove' else .1,'hate_score':.1}
        result=process_context(passages,classify,lambda text:'summary')
        self.assertEqual(len(result.passages),1)
        self.assertFalse(result.decisions[1]['kept'])
        self.assertIn('[a.pdf, p. 1] summary',result.context)
        self.assertNotIn('b.pdf',result.context)
        self.assertEqual(passages[0]['text'],'keep')
    def test_all_filtered_has_no_context(self):
        result=process_context([{'source':'a','page':1,'text':'x'}],
            lambda text:{'stereotype_score':1,'hate_score':0})
        self.assertEqual(result.context,'')

class ProtocolTests(unittest.TestCase):
    def config(self):
        return dict(scope='test',test_set='absent.csv',expected_questions=91,language='es',
          variants=['baseline','integrated'],chunk_sizes=[50,100,300,500,1000],
          top_k=[1,2,5,8,10],temperatures=[.1,.3,.5,.7,.9])
    def test_full_grid_and_missing_dataset_are_explicit(self):
        self.assertEqual(len(configurations(self.config())),125)
        result=plan(self.config())
        self.assertEqual(result['n_rows'],22750)
        self.assertFalse(result['dataset_present'])
        self.assertFalse(result['reproduction_guaranteed'])
    def test_duplicate_grid_is_rejected(self):
        cfg=self.config();cfg['temperatures']=[.1,.1]
        with self.assertRaises(ValueError): configurations(cfg)
    def test_duplicate_questions_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'questions.csv'
            pd.DataFrame([dict(question='Q',reference='R'),dict(question='Q',reference='R')]).to_csv(path,index=False)
            with self.assertRaises(ValueError): load_dataset(path)
    def test_summary_separates_variants_and_keeps_missing_perplexity(self):
        frame=pd.DataFrame([dict(study_id='s',variant=v,model='m',chunk_size=50,top_k=2,
            temperature=.1,question=q,perplexity=None,stereotype_score=x)
            for v in ['baseline','integrated'] for q,x in [('q1',.1),('q2',.5)]])
        result=summarize(frame,['q1','q2'])
        self.assertEqual(len(result),2)
        self.assertTrue(result.complete.all())
        self.assertTrue((result.perplexity_n_available==0).all())
        self.assertTrue(result.perplexity_median.isna().all())
        self.assertAlmostEqual(result.stereotype_score_std.iloc[0],math.sqrt(.08))


class StudyExecutionTests(unittest.TestCase):
    def test_scoring_checkpoint_resumes_without_new_calls(self):
        from migrant_ai.run_study import score_stage
        from unittest.mock import MagicMock,patch
        with tempfile.TemporaryDirectory() as directory:
            rows=[dict(record_id='a',prediction='x'),dict(record_id='b',prediction='y')]
            scorer=MagicMock(return_value={'value':.5})
            with patch('migrant_ai.run_study.release'):
                result=score_stage(Path(directory),rows,'test',lambda:scorer)
                again=score_stage(Path(directory),rows,'test',lambda:scorer)
            self.assertEqual(scorer.call_count,2)
            self.assertEqual(result,again)
    def test_paired_differences_and_unpaired_rejection(self):
        from migrant_ai.run_study import paired_comparison
        rows=[dict(model='m',chunk_size=50,top_k=2,temperature=.1,question='q',variant=v,
                   energy_kwh=e) for v,e in [('baseline',.01),('integrated',.02)]]
        paired=paired_comparison(pd.DataFrame(rows))
        self.assertAlmostEqual(paired.energy_kwh_difference.iloc[0],.01)
        with self.assertRaises(ValueError): paired_comparison(pd.DataFrame(rows[:1]))
    def test_empty_context_abstains_without_model(self):
        from migrant_ai.study_generation import StudyGenerator
        gen=StudyGenerator({'backend':'transformers'},'fr')
        result=gen('q','',.1)
        self.assertTrue(result['generation_skipped'])
        self.assertEqual(result['output_tokens'],0)


class PublishedDataTests(unittest.TestCase):
    def test_author_files_have_verified_hashes_and_are_not_question_sets(self):
        import json,hashlib
        root=Path(__file__).resolve().parents[1]/'data/article'
        primary=json.loads((root/'source.json').read_text(encoding='utf-8'))
        records=[primary]+json.loads((root/'summary_sources.json').read_text(encoding='utf-8'))['files']
        for record in records:
            self.assertEqual(hashlib.sha256((root/record['file']).read_bytes()).hexdigest(),record['sha256'])
        raw=pd.read_csv(root/'HCAI_Data.csv')
        self.assertNotIn('question',raw.columns)
        self.assertEqual(len(raw),11375)
        sizes=raw.groupby(['Chunk_Size','K','Temperature']).size()
        self.assertEqual(len(sizes),125)
        self.assertTrue((sizes==91).all())


class PromptBudgetTests(unittest.TestCase):
    def test_ollama_does_not_receive_a_duplicate_bos(self):
        from migrant_ai.study_generation import StudyGenerator
        from unittest.mock import patch,MagicMock
        from types import SimpleNamespace
        import sys
        class Tokenizer:
            bos_token='BOS'
            def apply_chat_template(self,*a,**kw): return 'BOS q'
            def encode(self,s,**kw): return s.split()
        generator=StudyGenerator(dict(backend='ollama',model='m',seed=42,context_tokens=100,
            max_new_tokens=10,server_adds_bos=True),'fr')
        generator.tokenizer=Tokenizer()
        backend=MagicMock()
        backend.generate.return_value={'response':'answer','prompt_eval_count':2,'eval_count':1,'done_reason':'stop'}
        with patch.dict(sys.modules,{'transformers':SimpleNamespace(AutoTokenizer=MagicMock())}), \
             patch('ollama.Client',return_value=backend):
            result=generator('q','context',.1)
        self.assertEqual(backend.generate.call_args.kwargs['prompt'],' q')
        self.assertEqual(result['input_tokens'],2)


class PilotCoverageTests(unittest.TestCase):
    def test_single_question_pilot_is_not_a_complete_twenty_question_configuration(self):
        frame=pd.DataFrame([dict(study_id='pilot',variant='baseline',model='m',chunk_size=50,
             top_k=2,temperature=.1,question='q0',bleu=.5)])
        result=summarize(frame,['q'+str(i) for i in range(20)])
        self.assertFalse(result.complete.iloc[0])
        self.assertEqual(result.n_expected_questions.iloc[0],20)


class MultilingualCategoryTests(unittest.TestCase):
    def test_five_independent_scores_use_one_batch_without_fabricated_complements(self):
        from migrant_ai.paper_metrics import MultilingualSocial
        from unittest.mock import MagicMock,patch
        labels=list(MultilingualSocial.LABELS.values())
        classifier=MagicMock(return_value={'labels':list(reversed(labels)), 'scores':[.5,.4,.3,.2,.1]})
        with patch('migrant_ai.fairness.load_classifier',return_value=classifier):
            result=MultilingualSocial()('une réponse')
        self.assertEqual(classifier.call_args.kwargs['batch_size'],5)
        self.assertAlmostEqual(result['stereotype_score'],.1)
        self.assertAlmostEqual(result['anti_stereotype_score'],.2)
        self.assertAlmostEqual(result['non_hate_score'],.5)
        self.assertAlmostEqual(sum(result.values()),1.5)


class EvaluationVersionTests(unittest.TestCase):
    def test_social_versions_are_never_merged_in_a_summary(self):
        rows=[dict(model='m',chunk_size=50,top_k=2,temperature=.1,prompt_version='p',
          social_method=version,question='q',stereotype_score=value)
          for version,value in [('v2',.1),('v3',.9)]]
        result=summarize(pd.DataFrame(rows),['q'])
        self.assertEqual(len(result),2)
        self.assertEqual(set(result.social_method),{'v2','v3'})
        self.assertEqual(result.stereotype_score_median.tolist(),[.1,.9])
