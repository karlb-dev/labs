# Freeze amendment (E2, single field)

`pcmech_difficulty_selected = d3`

Basis: GPU S3 calibration on train+validation incidentals only
(development tier): validation |neutral margin| per variant =
{d1: 3.828, d2: 1.508, d3: 1.793, d4: 0.012};
band [0.5, 3.0] nats; rule = closest to the 1.75-nat midpoint among
in-band variants. Holdout incidentals untouched during calibration.
This is the only permitted post-freeze write (prereg §8).
