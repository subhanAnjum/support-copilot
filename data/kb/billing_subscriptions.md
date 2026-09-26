# Billing & Subscriptions

## Subscription tiers
FitTrack Pro offers three tiers: Free, Plus (£4.99/month), and Premium (£9.99/month).
Premium includes AI-generated coaching feedback, unlimited workout history, and
priority support. Plus includes unlimited workout logging but no AI coaching.

## Being charged twice in one month
Double charges almost always come from one of two causes:
1. The user resubscribed after a failed payment retry without cancelling the
   original pending charge — Stripe will show two separate charge IDs in this case.
2. A plan change (e.g. Plus → Premium) mid-cycle triggers a prorated charge
   alongside the regular renewal charge — this is expected behaviour, not an error.

To resolve: check the Stripe dashboard for the charge IDs. If both charges
reference the *same* subscription ID, it is a duplicate and should be refunded
in full via Stripe's refund API. If they reference different subscription IDs
or include a proration line item, explain the proration behaviour to the user
rather than refunding.

## Cancelling a subscription
Users can cancel from Settings > Subscription > Cancel Plan. Cancellation
takes effect at the end of the current billing period; users retain Premium
features until then. There is no partial-month refund for standard cancellations.

## Refund policy
Refunds are issued for: duplicate charges, charges after a user cancelled but
was still billed due to a system error, and charges within 48 hours of
accidental resubscription. Refunds are NOT issued for: "I forgot to cancel"
after using the service for more than 7 days into the billing period, or
dissatisfaction with AI coaching quality (offer a downgrade to Plus instead).

## Failed payment handling
On a failed payment, the user has a 7-day grace period during which Premium
features remain active. After 7 days, the account is automatically downgraded
to Free. Users can update their payment method from Settings > Billing at any
point during the grace period to retry the charge immediately.
