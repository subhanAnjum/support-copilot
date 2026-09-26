# Technical Issues

## Workout sync not appearing across devices
FitTrack syncs workout data every 15 minutes when the app is in the
foreground, and on app open. If a workout logged on one device isn't showing
on another after 15+ minutes:
1. Confirm both devices are signed into the same account (Settings > Account).
2. Confirm both devices have an active internet connection during sync.
3. Force a manual sync from Settings > Data > Sync Now.
If the issue persists after a manual sync, it is likely a sync-token conflict
and should be escalated to engineering with the user's account ID and both
device IDs.

## AI coaching feedback not generating
AI coaching feedback (Premium feature) is generated asynchronously after a
workout is logged and typically appears within 60 seconds. Common causes of
missing feedback:
- The workout type isn't yet supported for AI feedback (currently supported:
  strength training, running, cycling — not yet supported: swimming, yoga).
- The user's subscription lapsed mid-generation (check subscription status
  first before troubleshooting further).
- A backend generation queue backlog during peak hours (6-8pm local time) —
  feedback can take up to 5 minutes during peak load; this is expected.

## App crashing on workout log screen
The most common cause of crashes on the workout log screen is a corrupted
local cache after an app update. Steps to resolve:
1. Ask the user to update to the latest app version if they haven't already.
2. Have them log out and log back in (this clears the local cache without
   losing synced data, since data is stored server-side).
3. If it persists, ask for the device model and OS version and escalate to
   engineering — this is usually device-specific.

## Push notifications not arriving
Check Settings > Notifications is enabled in-app AND at the OS level. iOS
users who deny notification permission on first install must manually
re-enable via iOS Settings > FitTrack > Notifications, since the in-app
toggle cannot override a system-level denial.
