VERSION_PREFIX = "BFCL_v4"


ALL_AVAILABLE_MEMORY_BACKENDS = [
    "kv",
    "vector",
    "rec_sum",
]

NON_LIVE_CATEGORY = [
    "simple_python",
    "simple_java",
    "simple_javascript",
    "multiple",
    "short_multiple",
    "short_multiple_tool_counts",
    "events_multiple",
    "events_multiple_tool_counts",
    # Scaled BFCL multiple catalogs (random / hard distractors).
    # Generator: tool-lora/ruler/create_multiple_tool_scale_dataset.py
    "multiple_scale_random_10",
    "multiple_scale_hard_10",
    "multiple_scale_random_10_smoke",
    "multiple_scale_hard_10_smoke",
    "multiple_scale_random_20",
    "multiple_scale_random_20_smoke",
    "multiple_scale_random_10_anon_smoke",
    "multiple_scale_hard_10_anon",
    # BFCL's OWN `multiple` catalogues (2-4 tools) with names anonymised, so the
    # only thing removed is the name signal. The hard_10 sets stack ten
    # lexically-similar distractors, confounding "can it read schemas" with
    # "can it survive a hard 10-way choice under compression"; at 2-4 tools the
    # 8x512 bottleneck is under almost no pressure, so what remains is
    # discrimination. Chance is 38.4%, not 10%.
    # Generator: build_multiple_anon (see session notes)
    "multiple_scale_natural_anon",
    # Gold + its 3 MOST SIMILAR distractors from hard_10_smoke. natural_anon
    # showed 99% at 2-4 tools, but BFCL's own catalogues are semantically far
    # apart, so that measured easy discrimination under no compression. These
    # keep N=4 (negligible compression) while making the distractors maximally
    # confusable (mean Jaccard 0.44), isolating discrimination itself.
    # Generator: build_hard4 (see session notes)
    "multiple_scale_hard_4_smoke",
    "multiple_scale_hard_4_anon_smoke",
    # Capability smoke tests (10-15 rows each): irrelevance-as-routing (an
    # explicit abstain tool appended to each catalogue) and parallel_multiple
    # run through the single-call staged pipeline.
    # Generator: doc-to-lora/smoke_tests/build_smoke.py
    "multiple_scale_irrel_smoke",
    "multiple_scale_irrel_noabst",
    "multiple_scale_prose_val",
    "multiple_scale_par_smoke",
    # The reserved-tool N-grid: 600 schemas held out of ALL training, hard
    # (max-Jaccard) distractors, permuted aliases, 150-row screening subsets.
    # ICL and BM25 are already measured on these; D2L never was, because
    # gate_reserved.py was broken. Running them through the BFCL harness -- the
    # only inference path in this repo that has reproduced known numbers --
    # gives the accuracy-vs-N curve the paper is built on.
    # Generator: build_reserved_eval.py
    "multiple_scale_reserved_2_anon_s150",
    "multiple_scale_reserved_4_anon_s150",
    "multiple_scale_reserved_6_anon_s150",
    "multiple_scale_reserved_8_anon_s150",
    "multiple_scale_reserved_10_anon_s150",
    "multiple_scale_reserved_20_anon_s150",
    "multiple_scale_reserved_50_anon_s150",
    "multiple_scale_reserved_100_anon_s150",
    # The _named_ halves existed on disk but were never registered, so the
    # harness raised "Invalid test category name" and the grid recorded rc=1 /
    # 0 rows rather than a result. Registered now: reserved is the only family
    # with NATURAL queries (21.0% of query terms appear in the gold schema, vs
    # 54.7% for the templated rand_*/hard_* families), which is where lexical
    # retrieval stops being a ceiling baseline.
    "multiple_scale_reserved_2_named_s150",
    "multiple_scale_reserved_4_named_s150",
    "multiple_scale_reserved_6_named_s150",
    "multiple_scale_reserved_8_named_s150",
    "multiple_scale_reserved_8_named",
    "multiple_scale_reserved_10_named",
    "multiple_scale_reserved_20_named",
    "multiple_scale_reserved_2_named",
    "multiple_scale_reserved_4_named",
    "multiple_scale_reserved_6_named",
    "multiple_scale_reserved_50_named",
    "multiple_scale_reserved_100_named",
    "multiple_scale_reserved_4_anon",
    "multiple_scale_reserved_8_anon",
    "multiple_scale_reserved_10_anon",
    "multiple_scale_reserved_20_anon",
    "multiple_scale_reserved_10_named_s150",
    "multiple_scale_reserved_20_named_s150",
    "multiple_scale_reserved_50_named_s150",
    "multiple_scale_reserved_100_named_s150",
    # hard_N beyond N=10, built this session with
    # ruler/create_multiple_tool_scale_dataset.py --modes hard. BM25-schema is
    # FLAT at ~80% across N=10..32 here (80.0/80.0/80.5/78.5), i.e. lexical
    # matching has a stable error floor on confusable catalogues; these measure
    # whether ICL degrades with N over the same catalogues.
    # Distinctive-label anon variants. tool_a..tool_h tokenise as ['tool','_x'] --
    # shared token 14172, one differing token slot -- which is close to the worst
    # case for a rank-8 lossy channel, and predicts the observed collapse onto
    # tool_9/tool_10 (the shortest token sequences) at large N. These labels are
    # multi-token with no shared discriminative position, and still carry NO
    # semantic relation to the tool, so schema-reading is still required.
    # Generator: $CLAUDE_JOB_DIR/tmp/make_distinct_eval.py
    "multiple_scale_rand_8_distinct",
    "multiple_scale_reserved_8_distinct_s150",
    "multiple_scale_reserved_20_distinct_s150",
    "multiple_scale_hard_4",
    "multiple_scale_hard_16",
    "multiple_scale_hard_24",
    "multiple_scale_hard_32",
    # WS3: BFCL `multiple` padded with RANDOM distractors to N in 4..32.
    # Random, not Jaccard-hard, on purpose -- hard_4 already measured
    # discrimination under maximal confusability; this holds difficulty roughly
    # constant and moves only catalogue size, isolating the COMPRESSION axis.
    # Named + anon share row ids so every method scores identical items.
    "multiple_scale_rand_4",
    "multiple_scale_rand_4_anon",
    "multiple_scale_rand_8",
    "multiple_scale_rand_8_anon",
    "multiple_scale_rand_12",
    "multiple_scale_rand_12_anon",
    "multiple_scale_rand_16",
    "multiple_scale_rand_16_anon",
    "multiple_scale_rand_24",
    "multiple_scale_rand_24_anon",
    "multiple_scale_rand_32",
    "multiple_scale_rand_32_anon",
    # Same items as multiple_scale_hard_10_anon with the aliases in sorted
    # order, matching how the router's training data presents them. Isolates
    # how much routing depends on alias order rather than on the schemas.
    # Generator: make_sorted_alias_variant.py
    "multiple_scale_hard_10_anon_sorted",
    # Held-out Nemotron/Toucan routing rows in the same candidate-scoring
    # harness, to separate schema-distribution shift from alias ordering.
    # Generator: export_indomain_routing_eval.py
    "multiple_scale_indomain_10_sorted",
    "multiple_scale_indomain_10_shuffled",
    # The other half of the same split, keeping real tool names. Separates
    # "routes from the name" from "routes from the schema".
    "multiple_scale_indomain_10_named_sorted",
    "multiple_scale_indomain_10_named_shuffled",
    # The permuted-alias validation split, in the catalogue order the training
    # rows store, so the gate and the in-training route_at_1 score identical
    # inputs and any gap between them is protocol rather than data.
    # Same Nemotron/Toucan distribution as training, but every tool in every
    # catalogue is absent from the training inventory, so tool novelty is the
    # only thing that changes against the in-domain gate. Separates memorised
    # inventory from schema-distribution shift.
    # Generator: build_tool_disjoint_eval.py
    "multiple_scale_heldout_10_anon",
    "multiple_scale_heldout_10_named",
    "multiple_scale_v2val_10_source",
    "multiple_scale_v2val_10_sorted",
    "multiple_scale_v2val_10_shuffled",
    "multiple_scale_hard_8_anon",
    "multiple_scale_hard_6_anon",
    "multiple_scale_hard_4_anon",
    "multiple_scale_hard_2_anon",
    "parallel",
    "parallel_multiple",
    "irrelevance",
    # "exec_simple",
    # "exec_parallel",
    # "exec_multiple",
    # "exec_parallel_multiple",
    # "rest",
    # "sql",
    # "chatable",
]
LIVE_CATEGORY = [
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "live_irrelevance",
    "live_relevance",
]
MULTI_TURN_CATEGORY = [
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
    # "multi_turn_composite",
]
WEB_SEARCH_CATEGORY = [
    "web_search_base",
    "web_search_no_snippet",
]

MEMORY_CATEGORY = [f"memory_{backend}" for backend in ALL_AVAILABLE_MEMORY_BACKENDS]
MEMORY_SCENARIO_NAME = [
    "student",
    "customer",
    "finance",
    "healthcare",
    "notetaker",
]


SINGLE_TURN_CATEGORY = NON_LIVE_CATEGORY + LIVE_CATEGORY
AGENTIC_CATEGORY = MEMORY_CATEGORY + WEB_SEARCH_CATEGORY
NON_SCORING_CATEGORY = ["format_sensitivity"]

ALL_SCORING_CATEGORIES = SINGLE_TURN_CATEGORY + MULTI_TURN_CATEGORY + AGENTIC_CATEGORY
ALL_CATEGORIES = ALL_SCORING_CATEGORIES + NON_SCORING_CATEGORY

TEST_COLLECTION_MAPPING = {
    "all": ALL_CATEGORIES,
    "all_scoring": ALL_SCORING_CATEGORIES,
    "multi_turn": MULTI_TURN_CATEGORY,
    "single_turn": SINGLE_TURN_CATEGORY,
    "live": LIVE_CATEGORY,
    "non_live": NON_LIVE_CATEGORY,
    "non_python": [
        "simple_java",
        "simple_javascript",
    ],
    "python": [
        "simple_python",
        "irrelevance",
        "parallel",
        "multiple",
        "parallel_multiple",
        "live_simple",
        "live_multiple",
        "live_parallel",
        "live_parallel_multiple",
        "live_irrelevance",
        "live_relevance",
    ],
    "memory": MEMORY_CATEGORY,
    "web_search": WEB_SEARCH_CATEGORY,
    "agentic": AGENTIC_CATEGORY,
}
