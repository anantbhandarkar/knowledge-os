# Data Retention Policy v2.0

This policy defines how long Acme retains different categories of data and when data is purged. Effective January 2026.

## Retention Periods

Customer transaction records are retained for 7 years to meet financial compliance requirements. Application logs are retained for 90 days. Audit logs are retained for 2 years. Personal data is deleted within 30 days of a verified deletion request.

## Backups

Production databases are backed up nightly. Backups are encrypted at rest with AES-256 and retained for 35 days. Restore drills are performed monthly by the platform team.

## Deletion Requests

Verified data-subject deletion requests are processed within 30 days. The data-platform service cascades deletions across the warehouse, the search index, and all downstream caches.
