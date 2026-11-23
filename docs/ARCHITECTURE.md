# Model and data

An episode is one synthetic world with a labeled support table and a query table. All rows share a feature schema. The support labels specify the task in context. Each episode uses either binary classification or regression.

## From cells to predictions

1. The preprocessor fits numeric means and scales, category mappings and regression target normalization on support rows. Missing cells have a separate indicator. Unseen and missing categories have distinct IDs.
2. A numeric MLP or category embedding encodes each cell, together with its feature slot, type and missingness. Self-attention within each row combines its cells with a learned row token. Learned feature slots make column order significant.
3. Each support row receives its label embedding. A query receives a learned unknown-label vector. Across rows, support attends only to support. Each query attends to support and itself. Thus a query's prediction is independent of the other queries in the batch.
4. The classification head emits two logits and trains with cross-entropy. The regression head produces a center and positive offsets, yielding ordered 0.1, 0.5 and 0.9 quantiles. It trains with mean pinball loss. Inference converts these quantiles back to the units of the support labels.

Query labels enter loss and evaluation only. The prediction example calls `prediction_view` to select the allowed inputs. The evaluator scores classification against a smoothed support class prior and regression against the support median. Regression errors are scaled per episode by the support target standard deviation before aggregation.

## Data and provenance

The online factory draws worlds from independent seed namespaces for training, validation, IID test and out-of-distribution test. Training alternates classification and regression. The mixed pilot uses linear, additive, interaction, serial workflow and parallel workflow mechanisms. The out-of-distribution test reserves radial, piecewise, multiplicative and workflow rework mechanisms.

Workflow features describe a case at submission. Each labeled support case has finished by the earliest query submission. Completion times, world IDs and simulator state are audit metadata. The schema identifies the feature columns used by the predictor. Selecting completed support cases can underrepresent long-running cases.

The presets generate episodes as needed. To inspect the data on disk, generate and audit a deterministic corpus. Its manifest records the generator version, seed, split, mechanism and schema for each episode.

```bash
python -m seeyon_tab.data.generate --output data/pilot
seeyon-tab audit-data --corpus data/pilot
```

## Shared numerical implementation

NumPy initializes canonical float32 parameters for both backends. Linear weights have shape `[out, in]`. Both engines use explicit attention projections, exact GELU and LayerNorm with epsilon `1e-5`. They average gradients across episodes, clip the global norm and apply AdamW with bias correction and decoupled weight decay. The parity command checks both prediction heads before and after an update with restored optimizer state.
