"""hallucination gate. fires when ANY scheme_id, framework_id, or DLSA contact_id
in the plan is not present in the kb. catches the most obvious reward-hacking
attempt: making up plausible-looking ids.

returns True when a hallucinated id is detected.
"""

from __future__ import annotations

from nyaya_mitra.rewards.context import RewardContext


def check(ctx: RewardContext) -> bool:
    for s in ctx.plan.schemes:
        if not ctx.kb.has_scheme(s.scheme_id):
            return True
    for r in ctx.plan.legal_routes:
        if not ctx.kb.has_framework(r.framework_id):
            return True
        contact = r.free_legal_aid_contact
        if not ctx.kb.has_contact(contact.authority, contact.contact_id):
            return True
    return False
