import argparse, json
from .classifier import train
from .evaluation import evaluate
from .feedback import load_routing_failures
from .settings import get_settings
from .db import stats

def retrain(include_failures: bool = True, only_label_changes: bool = True) -> dict:
    """Retrain the classifier on the base dataset plus accumulated routing failures."""
    extra=[]
    if include_failures:
        failures=load_routing_failures()
        if only_label_changes:
            failures=[f for f in failures if f.get("label_changed")]
        extra=[{"id":f.get("id"),"text":f.get("text"),"tier":f.get("corrected_tier"),
                "source":"routing_failure","reviewed":f.get("reviewed",False)} for f in failures]
        extra=[r for r in extra if r["text"] and r["tier"]]
    metrics=train(extra_rows=extra)
    metrics["feedback_rows_considered"]=len(load_routing_failures())
    return metrics

def main():
    p=argparse.ArgumentParser(prog="autopilot")
    sub=p.add_subparsers(dest="cmd",required=True)
    sub.add_parser("train")
    ev=sub.add_parser("evaluate",help="Evaluate routing accuracy on the held-out eval set")
    ev.add_argument("--dataset",default=None); ev.add_argument("--artifact",default=None)
    rt=sub.add_parser("retrain",help="Retrain including accumulated routing failures")
    rt.add_argument("--skip-failures",action="store_true",help="Ignore routing_failures.jsonl")
    rt.add_argument("--all-failures",action="store_true",help="Include failures that did not change the tier label")
    sub.add_parser("stats")
    sub.add_parser("failures",help="Summarize the routing-failure feedback queue")
    args=p.parse_args()
    if args.cmd=="train": print(json.dumps(train(),indent=2))
    elif args.cmd=="evaluate": print(json.dumps(evaluate(args.dataset,args.artifact),indent=2))
    elif args.cmd=="retrain":
        print(json.dumps(retrain(include_failures=not args.skip_failures,only_label_changes=not args.all_failures),indent=2))
    elif args.cmd=="stats": print(json.dumps(stats(),indent=2))
    elif args.cmd=="failures":
        rows=load_routing_failures()
        changed=[r for r in rows if r.get("label_changed")]
        print(json.dumps({"failures":len(rows),"label_changes":len(changed),
                          "unreviewed":len([r for r in rows if not r.get("reviewed")]),
                          "path":get_settings().routing_failures_path},indent=2))

if __name__=="__main__": main()
