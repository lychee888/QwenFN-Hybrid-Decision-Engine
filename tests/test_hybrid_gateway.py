"""Production-only boolean bridge and provenance tests. No model or network."""
from pathlib import Path
import sys,dataclasses,types,importlib.util
import pytest
R=Path(__file__).resolve().parents[1]
from qfn_decision import backend,protocol,templates
from qfn_decision.hybrid import HybridBackend
ROWS=[3721,1802,15,16,17,18,19,20]+list(range(32,48))
@dataclasses.dataclass
class Out:
 result:object
 captured:object
 arm_block:dict
 capture_block:dict

def app():
 a=HybridBackend.__new__(HybridBackend);a.config=types.SimpleNamespace(default_arm='B0',template_id=templates.TEMPLATE_QWEN_CHAT_LETTERS);return a

def fake(monkeypatch,seen,rows=None):
 def decide(self,req,**kw):
  seen.append(req);assert req.kind=='choice'
  v=[0.0]*24;v[8]=.2;v[9]=.8
  cap=types.SimpleNamespace(record=types.SimpleNamespace(verbalizer_ids=ROWS if rows is None else rows))
  return Out(protocol.DecisionResult.from_slot_probabilities(req,v),cap,{'arm':'B0','trained':False},{'engine_request_id':'owned'})
 monkeypatch.setattr(backend.DecisionBackend,'decide',decide)

def test_noul_bridge_false_true_descriptions_and_project(monkeypatch):
 seen=[];fake(monkeypatch,seen);q=protocol.DecisionRequest.create('noul','State','Question');out=app().decide(q)
 assert [(v.key,v.description) for v in seen[0].options]==[('false','false'),('true','true')]
 assert seen[0].state==q.state and seen[0].question==q.question and seen[0].limits is q.limits
 assert out.result.kind=='noul' and out.result.selected_key=='true' and out.result.p_true==.8
 assert out.result.slot_probabilities[:2]==(.2,.8) and all(v==0 for v in out.result.slot_probabilities[2:])
 assert out.capture_block['engine_request_id']=='owned' and out.arm_block['trained'] is False
 assert out.capture_block['wire_kind']=='choice' and out.capture_block['public_kind']=='noul'

def test_choice_identity_unchanged(monkeypatch):
 seen=[];fake(monkeypatch,seen);q=protocol.DecisionRequest.create('choice','S','Q',[protocol.ChoiceOption('b','Bee'),protocol.ChoiceOption('a','Aye')]);out=app().decide(q)
 assert seen==[q] and seen[0] is q and out.result.selected_key=='a'

def test_bad_row_order_refuses(monkeypatch):
 seen=[];rows=list(ROWS);rows[8],rows[9]=rows[9],rows[8];fake(monkeypatch,seen,rows)
 with pytest.raises(backend.BackendError):app().decide(protocol.DecisionRequest.create('noul','S','Q'))

def test_native_score_refuses_before_acquisition(monkeypatch):
 seen=[];fake(monkeypatch,seen)
 with pytest.raises(backend.ArmUnavailable):app().decide(protocol.DecisionRequest.create('score','S','Q',score_rubrics=['a']*6))
 assert not seen

def test_trained_arm_refuses_before_acquisition(monkeypatch):
 seen=[];fake(monkeypatch,seen)
 with pytest.raises(backend.ArmUnavailable):app().decide(protocol.DecisionRequest.create('noul','S','Q'),arm='L1O')
 assert not seen

def test_disabled_LoRA_registry_does_not_require_a_nonexistent_arm():
 from qfn_decision.backend import load_api_config
 cfg=load_api_config(R/'configs/decision_api.json')
 actual=HybridBackend(cfg,provider=object())  # no capture acquisition or network in this registry test
 assert cfg.default_arm=='B0' and actual.available_arms()=={'B0':True,'H1':False}
 assert not actual.available_arms().get('L1O',False)
