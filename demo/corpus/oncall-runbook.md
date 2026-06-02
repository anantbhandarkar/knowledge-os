# On-Call Runbook

Operational guidance for the platform on-call rotation. Owned by the platform team.

## Paging

The on-call engineer is paged via the incident tool. Acknowledge within 5 minutes. If you cannot, the page escalates to the secondary on-call after 10 minutes.

## Common Incidents

### Elevated Auth Failure Rate

If auth failure rates spike, first check whether the identity provider is degraded. The auth-service depends on the user-directory service; a user-directory outage will surface as auth failures. Roll back the most recent auth-service deploy if the spike correlates with it.

### Database Connection Exhaustion

If the API reports connection-pool exhaustion, check for a runaway migration or a stuck long-running query. The runbook for pool tuning lives in the platform wiki.

## Escalation

Sev-1 incidents require notifying the incident commander and opening a status-page entry within 15 minutes.
