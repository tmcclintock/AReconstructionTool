ART: A Reconstruction Tool
==========================

.. image:: https://img.shields.io/badge/arxiv-1905.09299-orange
    :target: https://arxiv.org/abs/1905.09299


A tool for reconstructing log-probability distributions using Gaussian processes. This tool requires an existing MCMC chain, or similar set of samples from a probability distribution, including the log-probabilities.

If you use this tool, please cite our paper here: `arxiv:1905.09299<https://arxiv.org/abs/1905.09299>`_.

Requirements
------------

The requirements for the ART resampler are mild. They include:

- numpy
- scipy
- `george<http://dfm.io/george/current/>`_
- `pyDOE2<https://pypi.org/project/pyDOE2/>`_

If you want to run the notebooks and produce the figures then you also need:

- jupyter
- matplotlib
- `emcee<http://dfm.io/emcee/current/>`_
- `corner<https://corner.readthedocs.io/en/latest/>`_
- `ChainConsumer<https://samreay.github.io/ChainConsumer/>`_

Note that ChainConsumer is a bit finicky with different versions of matplotlib. It may be the case that you have to downgrade some things to get those figures working.

To run the example in the Planck 2018 notebook, you should download the 2018 chains for the TT,TE,EE+lowE analysis `here<https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/Cosmological_Parameters>`_. You then have to pull out the data in the files to make them amenable to the notebook by putting the chain into its own file and the log-posteriors in another. Alternatively feel free to email me and I'll send you the files I have.

Development
-----------

This code is in live development. The API may break at any time. This won't be and issue when version 1.0 is released.
