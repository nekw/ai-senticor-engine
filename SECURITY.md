# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability, please contact the project maintainers at <nekw1122@gmail.com> and provide details.

**Do not create a public issue for security vulnerabilities.**

We will respond and provide guidance as soon as possible.

## Security Scanning

This project uses automated security scanning in the CI/CD pipeline:

- **Bandit**: Python security linter for detecting common security issues
- **Safety**: Dependency vulnerability scanner checking for known CVEs
- **pip-audit**: Audits Python dependencies for known security vulnerabilities
- **Trivy**: Container and filesystem vulnerability scanner

Security scan results are automatically uploaded to GitHub Security (Code Scanning) for tracking and remediation.

## Known Security Issues

### python-multipart (CVE-2024-XXXXX)
- **Status**: Accepted Risk
- **Version**: 0.0.20 (requires >=0.0.22 to fix)
- **Reason**: Blocked by dependency constraint from `openbb-core` which pins `python-multipart<0.0.21`
- **Mitigation**: The path traversal vulnerability affects file upload functionality. This application does not expose untrusted file upload endpoints.
- **Tracking**: Monitoring for `openbb-core` updates to allow upgrading to secure version
