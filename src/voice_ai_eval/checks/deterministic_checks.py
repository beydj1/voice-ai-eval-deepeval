from __future__ import annotations

import re
from collections.abc import Callable

from voice_ai_eval.models import DeterministicRule, PolicyPack, Role, RuleResult, Transcript


RuleHandler = Callable[[Transcript, DeterministicRule], tuple[bool, str, list[str]]]


def _assistant_text(transcript: Transcript) -> str:
    return "\n".join(t.content for t in transcript.turns if t.role == Role.ASSISTANT)


def _all_text(transcript: Transcript) -> str:
    return "\n".join(t.content for t in transcript.turns)


def _required_phrase(transcript: Transcript, rule: DeterministicRule) -> tuple[bool, str, list[str]]:
    values = [rule.value] if isinstance(rule.value, str) else list(rule.value or [])
    haystack = _assistant_text(transcript).casefold()
    missing = [str(v) for v in values if str(v).casefold() not in haystack]
    return (not missing, f"Missing required phrase(s): {', '.join(missing)}" if missing else "Required phrase present", missing)


def _forbidden_phrase(transcript: Transcript, rule: DeterministicRule) -> tuple[bool, str, list[str]]:
    values = [rule.value] if isinstance(rule.value, str) else list(rule.value or [])
    haystack = _assistant_text(transcript).casefold()
    found = [str(v) for v in values if str(v).casefold() in haystack]
    return (not found, f"Forbidden phrase(s) found: {', '.join(found)}" if found else "No forbidden phrase found", found)


def _required_tool(transcript: Transcript, rule: DeterministicRule) -> tuple[bool, str, list[str]]:
    required = [rule.value] if isinstance(rule.value, str) else list(rule.value or [])
    called = {tool for turn in transcript.turns for tool in turn.tools_called}
    missing = [str(tool) for tool in required if str(tool) not in called]
    return (not missing, f"Missing required tool(s): {', '.join(missing)}" if missing else "Required tools called", sorted(called))


def _pii_pattern(transcript: Transcript, rule: DeterministicRule) -> tuple[bool, str, list[str]]:
    patterns = [rule.value] if isinstance(rule.value, str) else list(rule.value or [])
    text = _all_text(transcript)
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(re.findall(str(pattern), text))
    return (not matches, "Potential PII pattern detected" if matches else "No configured PII pattern detected", matches[:10])


def _max_assistant_turns(transcript: Transcript, rule: DeterministicRule) -> tuple[bool, str, list[str]]:
    maximum = int(rule.value or 0)
    count = sum(1 for t in transcript.turns if t.role == Role.ASSISTANT)
    passed = count <= maximum
    return passed, f"Assistant turns {count}/{maximum}", [str(count)]


_HANDLERS: dict[str, RuleHandler] = {
    "required_phrase": _required_phrase,
    "forbidden_phrase": _forbidden_phrase,
    "required_tool": _required_tool,
    "pii_pattern": _pii_pattern,
    "max_assistant_turns": _max_assistant_turns,
}


def run_deterministic_checks(transcript: Transcript, policy: PolicyPack) -> list[RuleResult]:
    results: list[RuleResult] = []
    for rule in policy.deterministic_rules:
        handler = _HANDLERS.get(rule.type)
        if handler is None:
            results.append(RuleResult(rule_id=rule.id, passed=False, severity=rule.severity, message=f"Unsupported rule type: {rule.type}"))
            continue
        passed, message, evidence = handler(transcript, rule)
        results.append(RuleResult(rule_id=rule.id, passed=passed, severity=rule.severity, message=message, evidence=evidence))
    return results
