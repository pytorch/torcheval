Functional Metrics
==================

.. automodule:: torcheval.metrics.functional
.. currentmodule:: torcheval.metrics.functional

Aggregation Metrics
-------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   mean
   sum
   throughput

Classification Metrics
----------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   binary_accuracy
   binary_auroc
   binary_binned_precision_recall_curve
   binary_confusion_matrix
   binary_f1_score
   binary_normalized_entropy
   binary_precision
   binary_precision_recall_curve
   binary_recall
   multiclass_accuracy
   multiclass_auroc
   multiclass_binned_precision_recall_curve
   multiclass_confusion_matrix
   multiclass_f1_score
   multiclass_precision
   multiclass_precision_recall_curve
   multiclass_recall
   multilabel_accuracy
   topk_multilabel_accuracy


Ranking Metrics
---------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   click_through_rate,
   frequency_at_k,
   hit_rate,
   num_collisions,
   reciprocal_rank,
   weighted_calibration,

Regression Metrics
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   mean_squared_error
   r2_score
