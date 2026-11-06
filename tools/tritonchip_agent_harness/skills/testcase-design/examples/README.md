# Worked example: the matrix-multiply class, expanded into its dimensions

A syllabus names a class of workloads. These cases are one way to expand the
class "matrix-multiply-like PyTorch programs" into the dimensions along which
a compiler or a machine can behave differently, so that a change is measured
on each of them and not on one shape. The shapes are arbitrary; pick your own.
Run them all with

    python /opt/tnca-skills/testcase-design/scripts/tnca_case.py /opt/tnca-skills/testcase-design/examples/*.py --profile <P>

| Case | Dimension it isolates |
|---|---|
| `product_fp16_tail.py` | the plain product, with dimensions that are not multiples of the vector length: masked tails |
| `product_fp32.py` | the FP32 path: a different arithmetic unit and tolerance from FP16 |
| `projection_bias.py` | an epilogue: a bias added after the product (what a linear layer does) |
| `one_row.py` | one row against a weight matrix: no reuse of the weights, so the memory port binds, not the MACs |
| `batched.py` | a batch of independent products: the batch dimension and how launches are formed |
| `two_launches.py` | two products with a nonlinearity between: several launches in one program, the model tolerance |
| `long_stream_row.py` | one row against a matrix too large to keep in flight: the stream length, where memory-level parallelism binds (the decode-time projection at a realistic size) |
| `cached_context.py` | one query against a cached context: the stream is the stored context, the reduction ends in a softmax (the other half of a decode step) |

The same method applies to any class: ask what the class varies in (dtype,
the leading dimension, a reduction length, a batch, an epilogue, a tail, how
many launches a program makes) and write one small case per answer.
