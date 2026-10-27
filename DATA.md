# Data and licensing

The code is licensed under PolyForm Noncommercial 1.0.0 (see LICENSE). The **texts** in `examples/` and `eval/data/` come from several sources. Most are not ours
to relicense.

| class | source | in this repo | redistributable? |
|---|---|---|---|
| high (3) | [What's up with Gefilte Fish?](https://verkish.substack.com/p/whats-up-with-gefilte-fish), [You Need to Be Drinking So Much Coffee](https://thejauntwriting.substack.com/p/you-need-to-be-drinking-so-much-coffee), [Neural Networks Are Already Clean](https://already-clean.goodfire.com/) | full text: `eval/data/chunks_*.jsonl`, `examples/high_gefilte.txt` | **no**: authors' copyright |
| high (14) | The Atlantic essays from its public RSS feed, published Oct 20–21, 2026 | full text in `eval/data/chunks_*.jsonl` | **no**: The Atlantic's copyright |
| crank (6) | conspiracy / misinformation Substack posts from 3 authors, Sep–Oct 2026 | full text: `eval/data/chunks_main.jsonl`, `examples/crank_riots.txt`; author names in document ids | **no**: authors' copyright; naming them under a "crank" label is also a reputational call |
| claude (17 + 16 rewrites) | written by Claude Opus for this project | full text | yes |
| wiki (6) | English Wikipedia articles (Gefilte fish, Ashkenazi Jewish cuisine, Coffee, Caffeine, Mechanistic interpretability, Neural network (machine learning)) | full text, `examples/wiki_gefilte.txt` | yes, under CC BY-SA 4.0 with attribution (this table) |
| gib (4) | generated from the texts above (random characters, word salad, bigram Markov, shuffled passages) | full text, `examples/gib_markov.txt` | gray area: it mixes words from the texts above |
| 2×2 rewrites | Claude rewrites of the high essays (same ideas, Claude's voice), plus Claude essays rewritten in the authors' voices | full text, `eval/data/chunks_restyle.jsonl` | derivative of copyrighted essays: treat as **no** |

Scored outputs (`eval/data/<model>_*.jsonl`) repeat each passage's `text`, and newer runs also include its `tokens`.

## Publishing

Use `scripts/make_public_copy.py` rather than this folder as it stands:

```bash
python scripts/make_public_copy.py ../alpha-score-public --anonymize
```

The script:
- strips `text`, `tokens` and `sample0` from every eval file. Scores and per-token NLLs stay, so `eval/evaluate.py`
  and `scripts/build_calibration.py` still reproduce every number;
- keeps only the Claude, Wikipedia and gibberish examples;
- with `--anonymize`, renames the crank documents to `crank_01`..`crank_06`.

Things to check yourself:
- `docs/screenshots/` quotes a few short snippets of `high_gefilte.txt`;
- the README's demo of `alpha-score score examples/` will show only three files in the public copy.
