The "Notebooks" in this folder are written in a simple .md format, with HTML tags denoting the start of new code / markdown / output cells:

```
<!-- code -->
<!-- md -->
<!-- output -->
```

This makes it lighterweight for getting Agent support in working on them.

The `md_convert.py` utility (stdlib only, one direction) converts one of these into a runnable `.py` script (just the code cells) or into an actual `.ipynb` notebook:

```
python md_convert.py "03 - Analyze Rollout Nuclei.md" --py      # -> 03 - Analyze Rollout Nuclei.py
python md_convert.py "03 - Analyze Rollout Nuclei.md" --ipynb   # -> 03 - Analyze Rollout Nuclei.ipynb
python md_convert.py in.md --py -o out.py                       # explicit output path
```

`--py` keeps only the code cells (joined with `# %%` markers) and rewrites IPython magics
(`!pip ...`, `%matplotlib ...`) to `get_ipython()...`, so the script still parses under a
plain `python` run and the magic only fires under IPython/Colab. `--ipynb` keeps markdown +
code cells and drops `<!-- output -->` blocks. For round-tripping, output capture, image
hosting, or Colab upload, use the separate `colab-utils` tooling instead.

I have copies of the actual Notebooks on Google Colab, here:

(TODO)
