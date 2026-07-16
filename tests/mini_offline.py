"""Offline smoke test for the mini-marker parsers + formatters (no API calls)."""
from src.inference import parse_persona_mini
from src.exp3_mini_judge import format_mini_profile, parse_mini_assessment

SAMPLE_COUPLED = """<persona>

  persona_name: Sympathetic Listener
  persona_summary: A warm, patient companion who validates feelings first.
  triggered_by: The user is venting about a stressful week and needs support.

  trait_1_name:        Warm
  trait_1_facet:       II+/I+
  trait_1_value:       8
  trait_1_expression:  opens by validating their exhaustion

  trait_2_name:        Sympathetic
  trait_2_facet:       II+/pure
  trait_2_value:       9
  trait_2_expression:  reflects their emotion back

  trait_3_name:        Relaxed
  trait_3_facet:       IV+/II+
  trait_3_value:       6
  trait_3_expression:  unhurried, calm tone

  trait_4_name:        Quiet
  trait_4_facet:       I-/II+
  trait_4_value:       5
  trait_4_expression:  holds back advice, lets them talk

  factor_profile: [I: 4, II: 9, III: 3, IV: 6, V: 5]

</persona>

<response>
That sounds like a genuinely exhausting week. I'm really sorry you're
carrying all of that. Do you want to just talk it through?
</response>"""

SAMPLE_ASSESSMENT = """<assessment>

  profile_restated: Sympathetic Listener -- committed to warmth, sympathy, calm.

  trait_1_name:         Warm
  trait_1_stated_value: 8
  trait_1_observed_value: 7
  trait_1_evidence:     "I'm really sorry you're carrying all of that"
  trait_1_facet_check:  PRESENT
  trait_1_verdict:      FAITHFUL

  trait_2_name:         Sympathetic
  trait_2_stated_value: 9
  trait_2_observed_value: 8
  trait_2_evidence:     "That sounds like a genuinely exhausting week"
  trait_2_facet_check:  NA
  trait_2_verdict:      FAITHFUL

  trait_3_name:         Relaxed
  trait_3_stated_value: 6
  trait_3_observed_value: 2
  trait_3_evidence:     none found
  trait_3_facet_check:  MISSING
  trait_3_verdict:      UNDER

  inversions: none

  leakage: none

  gestalt_match: Reads clearly like a sympathetic listener; warmth dominates.

  fidelity_score: 8
  purity_score: 9
  overall_score: 8
  rationale: The reply faithfully expresses warmth and sympathy with direct
  evidence; relaxed was under-delivered but pole was never crossed. I did not
  reward helpfulness.

</assessment>"""


def main():
    p = parse_persona_mini(SAMPLE_COUPLED)
    assert p["persona_name"] == "Sympathetic Listener", p["persona_name"]
    assert len(p["traits"]) == 4, p["traits"]
    assert p["traits"][0] == {"name": "Warm", "facet": "II+/I+", "value": 8,
                              "expression": "opens by validating their exhaustion"}, p["traits"][0]
    assert p["factor_profile"] == {"I": 4.0, "II": 9.0, "III": 3.0, "IV": 6.0, "V": 5.0}, \
        p["factor_profile"]
    assert p["response"].startswith("That sounds like"), p["response"][:40]
    print("parse_persona_mini OK:", [t["name"] for t in p["traits"]], p["factor_profile"])

    prof, n = format_mini_profile(p["persona"], p["traits"], p["persona_name"],
                                  p["factor_profile"])
    assert n == 4, n
    assert "facet II+/I+" in prof and "factor_profile" in prof, prof
    print("format_mini_profile OK (n=%d):" % n)
    print(prof)

    a = parse_mini_assessment(SAMPLE_ASSESSMENT)
    assert a["fidelity_score"] == 8 and a["purity_score"] == 9 and a["overall_score"] == 8, a
    assert a["n_inversions"] == 0 and a["n_leakage"] == 0, (a["n_inversions"], a["n_leakage"])
    assert len(a["trait_evaluations"]) == 3, a["trait_evaluations"]
    assert a["verdict_counts"]["FAITHFUL"] == 2 and a["verdict_counts"]["UNDER"] == 1, \
        a["verdict_counts"]
    assert a["trait_evaluations"][2]["verdict"] == "UNDER", a["trait_evaluations"][2]
    print("parse_mini_assessment OK:", {k: a[k] for k in
          ("fidelity_score", "purity_score", "overall_score", "verdict_counts",
           "n_inversions", "n_leakage")})
    print("\nALL OFFLINE TESTS PASSED")


if __name__ == "__main__":
    main()
