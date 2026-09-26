# Account Management

## Resetting a forgotten password
Users can reset via the "Forgot password" link on the login screen, which
sends a reset email valid for 30 minutes. If the email doesn't arrive within
5 minutes, ask them to check spam and confirm the email address on file
matches — support agents cannot see plaintext passwords or manually reset
without the email flow for security reasons.

## Changing the email address on an account
Users can change their email from Settings > Account > Email. This requires
confirming the new email via a verification link. Until verified, the account
remains associated with the old email for login and billing purposes.

## Deleting an account
Account deletion is permanent after a 14-day grace period (to allow accidental
deletions to be reversed). During the grace period, the account is
deactivated but data is retained. Users can request immediate deletion by
contacting support directly, which bypasses the grace period but cannot be
undone. Any active subscription must be cancelled separately — deleting the
account does not automatically cancel Stripe billing.

## Merging duplicate accounts
FitTrack does not support automatic account merging. If a user has two
accounts (e.g. signed up with Apple ID and separately with email using the
same address), support can manually transfer workout history from one
account to another on request, but this requires manual data export and
should be escalated to the data team with both account IDs.

## Transferring a subscription between accounts
Subscriptions are tied to the Stripe customer ID, not the FitTrack account.
To move a subscription to a new account, the subscription must be cancelled
on the old account and a new one started — there is no direct transfer
mechanism. Warn users this may cause a brief gap in Premium access.
