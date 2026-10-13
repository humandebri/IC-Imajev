# Security reporting

Do not post credentials, private keys, unpublished inputs or exploitable operational details in public issues. Report a suspected vulnerability through a private channel agreed with the repository maintainer; no dedicated security mailbox or response-time commitment is configured here.

Include the affected revision, relevant source locations, prerequisites and a minimal reproduction using local test data. Avoid testing against shared or production canisters without authorization.

The repository's publication checker detects common credential formats and machine-specific paths. It is a hygiene check, not an application security audit. A clean result does not prove that a file or its Git history is safe to publish. See the [publication audit](docs/OSS_PUBLICATION_AUDIT.md) for measured scope and remaining risks.
