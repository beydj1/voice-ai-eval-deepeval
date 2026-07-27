# Production-readiness roadmap

The repository is production-structured and interview-ready. Before handling real customer data in a deployed service, complete the controls below.

1. Calibrate every metric against a human-labeled golden dataset and define acceptable agreement.
2. Pin and record judge model, prompt, metric, policy, and application versions with each result.
3. Add encrypted storage, retention controls, redaction, authentication, authorization, and audit logging.
4. Add retries with idempotency keys, rate limits, timeouts, circuit breakers, and provider fallback behavior.
5. Add observability for cost, latency, parse failures, score drift, provider errors, and policy failure rates.
6. Add a review queue for critical failures, low-confidence cases, and judge disagreement.
7. Run robustness tests against prompt injection embedded in transcripts and require the judge to treat transcript text as untrusted evidence.
8. Add an API adapter only after threat modeling and privacy approval.
