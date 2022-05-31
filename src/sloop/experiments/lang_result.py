from sciex import Experiment, Trial, Event, Result,\
    YamlResult, PklResult, PostProcessingResult
from .result_types import RewardsResult, StatesResult
from .plotting import *
from .pd_utils import *
from .constants import *
import pandas as pd
import yaml
import matplotlib.pyplot as plt

class LangResult(YamlResult):
    """Results that are related to language characteristics"""

    def __init__(self, lang, sg_dict):
        """list of state objects"""
        super().__init__({"lang": lang,
                          "sg_dict": sg_dict})

    @classmethod
    def FILENAME(cls):
        return "lang.yaml"

    @classmethod
    def collect(cls, path):
        # Collect both language as well as reward info and states info
        trial_path = os.path.dirname(path)
        with open(os.path.join(trial_path, RewardsResult.FILENAME())) as f:
            rewards = yaml.load(f)
        # Collect the states info
        with open(os.path.join(trial_path, StatesResult.FILENAME()), "rb") as f:
            states = pickle.load(f)
        # Collect the pickle file
        with open(path) as f:
            langres = yaml.load(f)
        return (langres, rewards, states)


    @classmethod
    def get_target_symbol(cls, states):
        """WARNING: ASSUMES EACH TIME THE ROBOT ONLY SEARCHES FOR ONE TARGET."""
        id_obj_map = {obj_id_map[k]:k for k in obj_id_map}
        letter_symbol_map = {symbol_letter_map[k]:k for k in symbol_letter_map}
        for objid in states[0].object_states:
            si = states[0].object_states[objid]
            if si.objclass == "target":
                letter = id_obj_map[objid]
                symbol = letter_symbol_map[letter]
                return symbol
        raise ValueError("Didn't find target. Something wrong.")

    @classmethod
    def relevant_relations(cls, sg_dict, target_symbol):
        res = []
        for rel in sg_dict["relations"]:
            if rel[0] == target_symbol:
                res.append(rel)
        return res

    @classmethod
    def gather(cls, results):
        # First, identify languages and annotated parses
        lang2dct = {}
        for specific_name in sorted(results):
            prior_type, _ = get_prior_type(specific_name)
            if prior_type.startswith("rule") and "auto" not in prior_type:
                for seed in sorted(results[specific_name]):
                    langres, rewards, states = results[specific_name][seed]
                    if langres["lang"] not in lang2dct\
                       and len(langres["sg_dict"]) > 0:
                        lang2dct[langres["lang"]] = langres["sg_dict"]

        # Next, collect the results.
        myresult1, myresult2 = {}, {}
        results_count = 0
        for specific_name in sorted(results):
            map_name = specific_name.split("-")[1]
            prior_type, _ = get_prior_type(specific_name)
            sensor_range = int(specific_name.split("-")[2].split(":")[3][-1])
            for seed in sorted(results[specific_name]):
                langres, rewards, states = results[specific_name][seed]
                lang = langres["lang"]
                if lang not in lang2dct:
                    continue
                sg_dict = lang2dct[lang]

                # Get the relations relevant to the target symbol for this case
                target_symbol = cls.get_target_symbol(states)
                rels = cls.relevant_relations(sg_dict, target_symbol)

                # Rewards and Number of detections
                cum_reward = sum(rewards)
                disc_reward = RewardsResult.discounted_reward(rewards, gamma=0.95)
                num_detected = 0
                for r in rewards:
                    if r == 1000:
                        num_detected += 1

                # Number of relations
                num_rels = len(rels)
                case_numrels = "-".join((str(sensor_range), map_name, seed, prior_type))
                if case_numrels not in myresult1:
                    myresult1[case_numrels] = []
                myresult1[case_numrels].append([str(num_rels), num_detected,
                                                cum_reward, disc_reward])

                # Predicate-specific
                for rel in rels:
                    predicate = rel[2]
                    case_predicates = "-".join((str(sensor_range), map_name, seed, prior_type))
                    if case_predicates not in myresult2:
                        myresult2[case_predicates] = []
                    myresult2[case_predicates].append([str(predicate), num_detected,
                                                       cum_reward, disc_reward])
                # increment results count
                results_count += 1
        return myresult1, myresult2

    @classmethod
    def save_gathered_results(cls, gathered_results, path):
        with open(os.path.join(path, "rewards_lang.yaml"), "w") as f:
            yaml.dump(gathered_results, f)
 #       if path[-1] == "/":
  #          path = path[:-1]
#        exp_path = os.path.dirname(path)
        with open(os.path.join(path, "good_foref_predictions.yaml")) as f:
            good_forefs = yaml.load(f)
        cls.do_it(gathered_results, path)
        print("--goood---")
        cls.do_it(gathered_results, path, good_forefs=good_forefs)
        print("---bad---")
        cls.do_it(gathered_results, path, good_forefs=good_forefs, do_bad=True)

    @classmethod
    def do_it(cls, gathered_results, path, good_forefs={}, do_bad=False):
        suffix = ""
        if len(good_forefs) > 0:
            suffix = "_good-forefs"
            if do_bad:
                suffix = "_bad-forefs"

        df_numrels, df_predicates = cls.organize(gathered_results, good_forefs=good_forefs, do_bad=do_bad)
        df_numrels.to_csv(os.path.join(path, "rewards_lang-numrels%s.csv" % suffix))
        df_predicates.to_csv(os.path.join(path, "rewards_lang-predicates%s.csv" % suffix))
        if len(df_numrels) > 0:
            summary = cls.summarize_numrels(df_numrels, num_detected=1)
            summary.to_csv(os.path.join(path, "rewards_lang-numrels_summary%s.csv" % suffix))

            summary = cls.summarize_predicates(df_predicates, num_detected=1)
            summary.to_csv(os.path.join(path, "rewards_lang-predicates_summary%s.csv" % suffix))

    ## (ATTEMPT 1) THESE METHODS REPORT PER-PREDICATE / PER-NUMRELS DISCOUNTED REWARDS
    @classmethod
    def organize(cls, gathered_results, good_forefs={}, do_bad=False):
        """good_forefs: seeds that have good foref predictions"""
        # Get a dictionary that maps from
        # sensor range -> { prior_type -> {nrels -> reward}}

        # prior_types = {"informed", "informed#15", "informed#5",
        #                "keyword", "keyword#auto", "rule#based#ego#ctx",
        #                "rule#based#ego#ctx#auto", "uniform"}

        prior_types1 = set()
        prior_types2 = set()
        casemap_numrels = {}  # map from case to prior types; Used to make sure we only
                              # use trials where all baselines have results
        casemap_predicates = {}  # map from case to prior types; Used to make sure we only
                                 # use trials where all baselines have results
        for global_name in gathered_results:
            results_numrels, results_predicates = gathered_results[global_name]
            for case in results_numrels:
                # Results concerning the number of predicates
                sensor_range, map_name, seed, prior_type = case.split("-")
                prior_types1.add(prior_type)
                for num_rels, num_detected, cum_reward, disc_reward in results_numrels[case]:
                    trial_id = (sensor_range, map_name, seed)
                    if trial_id not in casemap_numrels:
                        casemap_numrels[trial_id] = []
                    casemap_numrels[trial_id].append([
                        prior_type, num_rels, num_detected, cum_reward, disc_reward
                    ])

            for case in results_predicates:
                # Results concerning the spatial relation keyword
                sensor_range, map_name, seed, prior_type = case.split("-")
                prior_types2.add(prior_type)
                for predicate, num_detected, cum_reward, disc_reward in results_predicates[case]:

                    if predicate in good_forefs:
                        if not do_bad:
                            if not (map_name in good_forefs[predicate]\
                                    and int(seed) in good_forefs[predicate][map_name]):
                                print("Skipping %s-%s-%s since it's not a good foref prediction"
                                      % (predicate, map_name, seed))
                                continue
                        else:
                            if (map_name in good_forefs[predicate]\
                                and int(seed) in good_forefs[predicate][map_name]):
                                print("Skipping %s-%s-%s since it's not a bad foref prediction"
                                      % (predicate, map_name, seed))
                                continue

                    trial_id = (sensor_range, map_name, seed)
                    if trial_id not in casemap_predicates:
                        casemap_predicates[trial_id] = []
                    casemap_predicates[trial_id].append([
                        prior_type, predicate, num_detected, cum_reward, disc_reward
                    ])

        rows_numrels = []
        counts_numrels = {}
        for trial_id in casemap_numrels:
            if set(t[0] for t in casemap_numrels[trial_id]) != prior_types1:
                # We do not have all prior types for this case.
                continue
            for prior_type, num_rels, num_detected, cum_reward, disc_reward in casemap_numrels[trial_id]:
                sensor_range, map_name, seed = trial_id
                if (sensor_range, map_name, prior_type) not in counts_numrels:
                    counts_numrels[(sensor_range, map_name, prior_type)] = 0
                rows_numrels.append([
                    sensor_range, prior_type, num_rels, map_name, seed,
                    num_detected, cum_reward, disc_reward
                ])
                counts_numrels[(sensor_range, map_name, prior_type)] += 1
        df_numrels = pd.DataFrame(rows_numrels,
                                  columns=["sensor_range", "prior_type", "num_rels",
                                           "map_name", "seed", "num_detected",
                                           "cum_reward", "disc_reward"])

        rows_predicates = []
        counts_predicates = {}
        for trial_id in casemap_predicates:
            if set(t[0] for t in casemap_predicates[trial_id]) != prior_types2:
                # We do not have all prior types for this case.
                continue
            for prior_type, predicate, num_detected, cum_reward, disc_reward in casemap_predicates[trial_id]:
                sensor_range, map_name, seed = trial_id
                if (sensor_range, map_name, prior_type) not in counts_predicates:
                    counts_predicates[(sensor_range, map_name, prior_type)] = 0
                rows_predicates.append([
                    sensor_range, prior_type, predicate, map_name, seed,
                    num_detected, cum_reward, disc_reward
                ])
                counts_predicates[(sensor_range, map_name, prior_type)] += 1
        df_predicates = pd.DataFrame(rows_predicates,
                                     columns=["sensor_range", "prior_type", "predicate",
                                              "map_name", "seed", "num_detected",
                                              "cum_reward", "disc_reward"])
        return df_numrels, df_predicates

    @classmethod
    def summarize_numrels(cls, df, num_detected=-1):
        """If `num_detected` is greater than 0, then only rewards
        with at least this number of detections will be summarized."""
        # if num_detected > 0:
        #     df = df.loc[df["num_detected"] >= num_detected]
        summary = df.groupby(["prior_type", "sensor_range", "num_rels"])\
                    .agg([("ci95", lambda x: ci_normal(x, confidence_interval=0.95)),
                          ("ci90", lambda x: ci_normal(x, confidence_interval=0.9)),
                          ("sem", lambda x: stderr(x)),
                          ('avg', 'mean'),
                          ('median', lambda x: np.median(x)),
                          'std',
                          'count',
                          'sum'])  # this is not relevant for most except for foref_prediction_count
        flatten_column_names(summary)
        return summary

    @classmethod
    def summarize_predicates(cls, df, num_detected=-1):
        """If `num_detected` is greater than 0, then only rewards
        with at least this number of detections will be summarized."""
        summary = df.groupby(["prior_type", "sensor_range", "predicate"])\
                    .agg([("ci95", lambda x: ci_normal(x, confidence_interval=0.95)),
                          ("ci90", lambda x: ci_normal(x, confidence_interval=0.9)),
                          ("sem", lambda x: stderr(x)),
                          ('avg', 'mean'),
                          ('median', lambda x: np.median(x)),
                          'std',
                          'count',
                          'sum'])  # this is not relevant for most except for foref_prediction_count
        flatten_column_names(summary)
        return summary

    ##########################################

    ## (ATTEMPT 2) THESE AIM TO BETTER REFLECT THE INFLUENCE OF THE PREDICATE ON THE PERFORMA
