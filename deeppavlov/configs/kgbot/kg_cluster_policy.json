{
  "chainer": {
    "in": ["utterance"],
    "pipe": [
      {
        "name": "kg_data",
        "id": "kg_data"
      },
      {
        "name": "kg_ranker",
        "data": "#kg_data.data",
        "data_type": "events",
        "n_top": -1,
        "in": ["utterance"],
        "out": ["id", "score"]
      },
      {
        "name": "kg_tagger",
        "data": "#kg_data.data",
        "data_type": "events",
        "threshold": 75,
        "in": ["utterance"],
        "out": ["tag_scores"]
      },
      {
        "name": "kg_filter",
        "data": "#kg_data.data",
        "data_type": "events",
        "n_top": 30,
        "in": ["id", "score", "tag_scores"],
        "out": ["filtered_events"]
      },
      {
        "name": "kg_cluster_policy",
        "clusters": "#kb_data.data['clusters']",
        "in": ["filtered_events"],
        "out": ["cluster_id", "message"]
      },
    ],
    "out": ["message"]
  },
}
