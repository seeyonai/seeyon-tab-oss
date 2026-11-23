"""Counter-based deterministic episode sampling, portable across devices."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .data import GENERATOR_VERSION, SPLIT_FAMILIES, sample_episode, EpisodeCorpus
from .checkpoint import sha256

class EpisodeSource:
    def __init__(self, config):
        self.config=config
        self.corpus=None
        self._epoch=None
        self._order=None
        from . import preprocess
        from .data import factory, corpus
        # Data semantics, not just raw seeds, must survive a backend migration.
        semantic_hashes = [sha256(__file__), sha256(preprocess.__file__),
                           sha256(factory.__file__), sha256(corpus.__file__)]
        common = ["seeyon-source-v2", GENERATOR_VERSION, np.__version__, *semantic_hashes]
        if config["data"]["source"]=="corpus":
            self.corpus=EpisodeCorpus(config["data"]["corpus_path"],"train",for_training=True)
            # Fingerprint the actual bytes, not just a manifest's filenames.
            files=[self.corpus.directory/"manifest.jsonl"] + [
                self.corpus.directory/r["array_path"] for r in self.corpus.records]
            self.fingerprint=hashlib.sha256("\n".join(common+["corpus"]+[sha256(p) for p in files]).encode()).hexdigest()
        else:
            self.fingerprint=hashlib.sha256("\n".join(common+["online"]).encode()).hexdigest()

    def get(self,index):
        if type(index) is not int or index<0: raise ValueError("Episode index must be nonnegative")
        c=self.config; seed=c["training"]["seed"]
        if self.corpus is not None:
            epoch,offset=divmod(index,len(self.corpus))
            if self._epoch!=epoch:
                self._epoch=epoch
                self._order=np.random.default_rng(np.random.SeedSequence([seed,epoch,734])).permutation(len(self.corpus))
            return self.corpus[int(self._order[offset])]
        families=c["data"]["families"]
        family=None
        if families:
            family=families[int(np.random.default_rng(np.random.SeedSequence([seed,index,991])).integers(len(families)))]
        task="classification" if index%2==0 else "regression"
        ep=c["episode"]
        return sample_episode(seed+index,split="train",family=family,
                              n_support=ep["support"],n_query=ep["query"],n_features=ep["features"],
                              problem_type=task)

def evaluation_episode(config, split, index):
    if split not in ("validation","test_iid","test_ood"):
        raise ValueError("Evaluation cannot use train")
    ep=config["episode"]
    return sample_episode(config["evaluation"]["seed"]+index,split=split,
                          n_support=ep["support"],n_query=ep["query"],n_features=ep["features"],
                          problem_type="classification" if index%2==0 else "regression")

