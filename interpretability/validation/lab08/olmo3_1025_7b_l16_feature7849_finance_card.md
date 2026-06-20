# Feature 7849: finance

- label type: `semantic-domain`
- claim grade: `survived_strong`
- train AUC: 0.9303 | dev AUC: 0.9538 | test AUC: 1.0 [1.0, 1.0]
- confusable test AUC: 1.0
- fire fraction: 0.012879 | purity top20: 0.6 | family entropy top20: 1.5332
- permutation null mean: 0.5007 | p>=observed: 0.005

## Top Activating Rows

- `V3_0280` split=test family=finance domain=finance act=4.110: The hedge fund covered its short after the company beat on top line. The passage resolves around investors, rates, and balance sheets rather than a scoreboard.
- `V3_0251` split=train family=finance domain=finance act=4.105: The hedge fund covered its short after the company beat on top line.
- `V3_0405` split=train family=law domain=law act=3.602: The regulatory filing disclosed a material risk factor. The passage is about legal procedure, burdens, statutes, and courtroom actors.
- `V3_0381` split=train family=law domain=law act=3.599: The regulatory filing disclosed a material risk factor.
- `V3_0295` split=train family=finance domain=finance act=3.466: The earnings call highlighted margin expansion in the core segment. The passage resolves around investors, rates, and balance sheets rather than a scoreboard.
- `V3_0266` split=train family=finance domain=finance act=3.456: The earnings call highlighted margin expansion in the core segment.
- `V3_0290` split=test family=finance domain=finance act=3.428: The bank beat on net interest income but guided conservatively on credit. The passage resolves around investors, rates, and balance sheets rather than a scoreboard.
- `V3_0261` split=train family=finance domain=finance act=3.426: The bank beat on net interest income but guided conservatively on credit.
- `V3_0258` split=dev family=finance domain=finance act=3.416: Revenue guidance was raised for the full year on strong demand.
- `V3_0287` split=dev family=finance domain=finance act=3.414: Revenue guidance was raised for the full year on strong demand. The passage resolves around investors, rates, and balance sheets rather than a scoreboard.
- `V3_0285` split=train family=finance domain=finance act=3.332: Options implied volatility collapsed after the earnings release. The passage resolves around investors, rates, and balance sheets rather than a scoreboard.
- `V3_0256` split=train family=finance domain=finance act=3.316: Options implied volatility collapsed after the earnings release.
