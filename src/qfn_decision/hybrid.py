"""Untrained B0 gateway on the frozen chat-letter surface. No trained adapter loaded."""
import dataclasses,json,signal,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
from . import api,backend,protocol,templates
ROWS=(3721,1802,15,16,17,18,19,20,*range(32,48))
class HybridBackend(backend.DecisionBackend):
 def health(self):
  doc=super().health();doc['protocol']['kinds']=['noul','choice'];doc['production_hybrid']={'trained_adapter_loaded':False,'native_score_supported':False,'readout_surface':'qwen_chat_letters-v1','boolean_bridge':'A=false / B=true','calibrated_confidence_claim':False};return doc
 def decide(self,request,*,arm=None,**kwargs):
  if (arm or self.config.default_arm)!='B0':raise backend.ArmUnavailable('Production hybrid exposes untrained B0 only; H1 and LoRA are disabled')
  if self.config.template_id!=templates.TEMPLATE_QWEN_CHAT_LETTERS:raise backend.BackendError('Production hybrid requires the frozen chat-letter surface')
  if request.kind=='score':raise backend.ArmUnavailable('Native score is not supported by this chat-letter readout; score-choice surrogate is benchmark-only')
  wire=request
  if request.kind=='noul':
   wire=protocol.DecisionRequest.create('choice',request.state,request.question,options=[protocol.ChoiceOption('false','false'),protocol.ChoiceOption('true','true')],limits=request.limits)
  outcome=super().decide(wire,arm='B0',**kwargs)
  if tuple(outcome.captured.record.verbalizer_ids)!=ROWS:raise backend.BackendError('Captured verbalizer IDs/order differ from the pinned base readout')
  if request.kind!='noul':return outcome
  vector=[0.0]*24;vector[0:2]=outcome.result.option_probabilities
  result=protocol.DecisionResult.from_slot_probabilities(request,vector)
  return dataclasses.replace(outcome,result=result,
   arm_block={**outcome.arm_block,'wire_kind':'choice','public_kind':'noul','wire_active_slots':[8,9],'public_active_slots':[0,1],'bridge':'A=false / B=true, descriptions equal keys'},
   capture_block={**outcome.capture_block,'wire_kind':'choice','public_kind':'noul','wire_template_identity_preserved':True})

def make_server(config):
    """Reject trained arms and unsupported surfaces before loading artifacts."""
    if (config.default_arm != 'B0' or config.h1.get('artifact_path') is not None
            or (config.decision_adapter is not None and config.decision_adapter.enabled)):
        raise backend.ConfigError('Only unchanged-base B0 is enabled in this release')
    if (config.template_id != templates.TEMPLATE_QWEN_CHAT_LETTERS
            or config.chat_template_kwargs != {'enable_thinking': False}):
        raise backend.ConfigError('The launcher requires chat letters with thinking disabled')
    return api.GatewayServer(config, backend_instance=HybridBackend(config))


def main(argv=None):
    import argparse
    import threading
    parser = argparse.ArgumentParser(description='Untrained B0 gateway; never starts an engine')
    parser.add_argument('--config', required=True, help='Generated gateway.json path')
    parser.add_argument('--check', action='store_true', help='Validate without opening a listener')
    args = parser.parse_args(argv)
    try:
        server = make_server(backend.load_api_config(args.config))
    except backend.BackendError as exc:
        parser.error(str(exc))
    if args.check:
        print(json.dumps(server.health(), indent=2))
        return 0
    stopped = threading.Event()
    previous = signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    try:
        server.start()
        while not stopped.wait(1):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        signal.signal(signal.SIGTERM, previous)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

