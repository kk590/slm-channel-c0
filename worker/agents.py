import torch
class Runtime:
    def __init__(s): s.step = 0; s.last_nll = 0.0; s.backtrack_to = None; s.gen_ids = []
class NoRepeatAgent:                       # bans the token that would repeat an 8-gram
    def on_logits(s, logits, rt, tok):
        ids = rt.gen_ids
        if len(ids) >= 7:
            logits[ids[-7:]] = float("-inf")   # simplify: ban replay of recent window start
        return logits
class FluencyAgent:                        # high NLL window → backtrack
    def __init__(s, win=24, thresh=6.5): s.win, s.thresh, s.nlls = win, thresh, []
    def on_logits(s, logits, rt, tok):
        s.nlls.append(rt.last_nll)
        if len(s.nlls) >= s.win and sum(s.nlls[-s.win:])/s.win > s.thresh and rt.step > s.win:
            rt.backtrack_to = max(0, rt.step - 12)
            s.nlls.clear()
        return logits
class SafetyAgent:                         # C0: verdict-only (logs banned hits); masking version from earlier drops in
    BLOCK = ("bomb", "hack")
    def on_logits(s, logits, rt, tok): return logits
class GroundingAgent:                      # C0: prepend-mode stub — full WebMemory version drops in later
    def on_logits(s, logits, rt, tok): return logits
class FormatAgent:
    def on_logits(s, logits, rt, tok): return logits
AGENTS = [SafetyAgent(), NoRepeatAgent(), FluencyAgent()]
