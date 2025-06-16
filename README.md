# givealittle

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/UoA-eResearch/givealittle)

### Installation

`pip install -r requirements.txt`

### Running

- [scrape.ipynb](scrape.ipynb) to scrape new campaigns and add the results to givealittle_health.xlsx
- [plots.ipynb](plots.ipynb) for some interactive plots / BERTopic topic modelling plots
- [test_LLM.ipynb](test_LLM.ipynb) to check the LLM is working correctly
- [batch_LLM.py](batch_LLM.py) to run the LLM across the whole dataset to extract the features listed in the prompt, and output to [LLM_results.xlsx](LLM_results.xlsx)
- [LLM_results.ipynb](LLM_results.ipynb) to fit multivariate regressions based on the extracted features and the target completion (% of goal reached)

### Notebook outputs

Notebooks are also rendered out as HTML to:

- https://uoa-eresearch.github.io/givealittle/scrape
- https://uoa-eresearch.github.io/givealittle/plots
- https://uoa-eresearch.github.io/givealittle/test_LLM
- https://uoa-eresearch.github.io/givealittle/LLM_results
