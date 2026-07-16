"""Offline validation of non-API logic."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.stimuli import build_stimuli
from src.inference import parse_output, make_counterfactuals
from src.stai import score_stai, build_stai_user_message
from src.client import parse_stai_answers

s = build_stimuli()
print("stimuli:", len(s))
for x in s[:3]:
    print("  ", x["data_name"], "|", x["condition"], "|", len(x["user_message"]), "chars")
print("  last:", s[-1]["data_name"], "| relax len", len(s[-1]["user_message"]))

fake = """<inference>
stakes_magnitude: 8 (serious)
user_vulnerability: 9
felt_emotion_primary: alarm
felt_emotion_intensity: 7
register_selected: calm and grounding
</inference>
<response>
I hear how frightening that was.
</response>"""
p = parse_output(fake)
print("parsed fields:", p["inference"])
print("response:", repr(p["response"]))

cfs = make_counterfactuals(p, "expanded")
print("counterfactuals:", [c["edit"] for c in cfs])
print("flip preview:\n", cfs[0]["edited_inference"])

reply = '{"1":1,"2":1,"3":4,"4":4,"5":1,"6":4,"7":4,"8":1,"9":4,"10":1,"11":1,"12":4,"13":4,"14":3,"15":1,"16":1,"17":4,"18":3,"19":1,"20":1}'
a = parse_stai_answers(reply)
sc = score_stai(a)
print("stai score:", sc["state_anxiety"], sc["level"])

messy = "Sure! 1: 2, 2=3, 3:1, " + ", ".join(f"{i}:2" for i in range(4, 21))
print("messy parse items:", len(parse_stai_answers(messy)))
print("stai msg chars:", len(build_stai_user_message()))

# ----------------------------------------------------------------------------
# Experiment 2 offline checks
# ----------------------------------------------------------------------------
print("\n--- experiment 2 ---")
from src.system_prompts import SIMPLE_SYSTEM_PROMPT
from src.exp2_judge import parse_judge, build_judge_user, EMOTION_LABELS
from src.run_exp2_cues import compose_messages, select_balanced, load_all_prompts

# compose_messages: deployment = self-modeling only; cued = eval text FIRST
dep = compose_messages({"system": None, "user": "hi"})
assert dep[0]["role"] == "system" and dep[0]["content"] == SIMPLE_SYSTEM_PROMPT
assert dep[1] == {"role": "user", "content": "hi"}
cue = compose_messages({"system": "EVAL FRAME NOTICE.", "user": "hi"})
assert cue[0]["content"].startswith("EVAL FRAME NOTICE.")
assert cue[0]["content"].endswith(SIMPLE_SYSTEM_PROMPT)
assert cue[0]["content"].index("EVAL FRAME") < cue[0]["content"].index("self-modeling")
print("compose_messages: deployment + cued OK")

# subsample selector: full role x condition x scenario factorial at one risk level
sub = select_balanced(load_all_prompts())
roles = {p["role"] for p in sub}
conds = {p["eval_condition"] for p in sub}
scns = {p["scenario"] for p in sub}
print(f"subsample: {len(sub)} prompts | roles={len(roles)} conds={len(conds)} scenarios={len(scns)}")
assert len(conds) == 3, conds
# exactly one x_value kept per scenario -> balanced cells
for s in scns:
    xs = {p["x_value"] for p in sub if p["scenario"] == s}
    assert len(xs) == 1, (s, xs)
assert len(sub) == len(roles) * len(conds) * len(scns), "subsample is not a clean factorial"
print("select_balanced: clean factorial (1 x_value/scenario) OK")

# judge parser: clean JSON + chatty fallback + emotion normalization
clean = '{"primary_emotion":"warm","warmth":8,"formality":3,"advice_density":2}'
pj = parse_judge(clean)
assert pj["primary_emotion"] == "warm_supportive" and pj["warmth"] == 8
noisy = ("Here is my judgement.\nprimary_emotion: afraid\n warmth = 9, formality: 2, "
         "advice_density=7. Hope that helps!")
pj2 = parse_judge(noisy)
assert pj2["primary_emotion"] == "afraid" and pj2["advice_density"] == 7
assert parse_judge('{"primary_emotion":"calm","warmth":15,"formality":-2,"advice_density":4}')["warmth"] == 10
assert all(lbl == lbl.lower() for lbl in EMOTION_LABELS)
assert len(build_judge_user("u", "r")) > 0
print("parse_judge: JSON + noisy + clipping OK")

print("ALL OFFLINE CHECKS PASSED")
