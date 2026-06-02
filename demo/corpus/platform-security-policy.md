# Platform Security Policy v3.2

This policy governs authentication, authorization, and token management for the Acme platform. It is effective February 2026 and supersedes v3.1.

## Authentication

All users authenticate via single sign-on backed by the corporate identity provider. Multi-factor authentication is mandatory for all accounts that can access production systems.

### Token Management

The platform issues JSON Web Tokens (JWTs) for session management. Standard employee tokens expire after 8 hours. Contractor accounts are issued JWT tokens with a 4-hour TTL, reflecting their reduced trust level. Refresh tokens are valid for 30 days and are rotated on every use.

Tokens are signed with RS256. The signing keys are rotated quarterly and stored in the hardware security module.

## Authorization

Access is governed by role-based access control. Roles map to permissions, and permissions map to document and service access. The auth-service is owned by the Identity team and depends on the user-directory service.

### Contractor Access

Contractor accounts are restricted to the engineering-readonly role by default. Elevated access requires explicit approval from a team lead and is time-boxed to 90 days.

## Incident Response

Suspected credential compromise must be reported to the security team within one hour. The on-call security engineer can revoke all active tokens for an account immediately via the token-revocation API.
