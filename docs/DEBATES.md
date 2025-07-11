
TODO: what do we need to reuse the ground truth related logic?

- **Philosophy**: 
  - we impose conventions on OWN code and internal design
  - we support configurable conventions for users
  - so fixed structure are just support built-in methods, users can define their own structure

OK!!!!!!!!!!!!!!!!!! how about, framework just supports very basic ground truth retrieval:
- `ground_truth = dataset_item.get("data", {}).get("answer", None)`
And if user wants to use custom evaluation methods
- they just ignore the ground truth parameter
- then they retrieve the ground truth on their own
  - but they need at least `item.id` to link back to the dataset item!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

do we need to define a fixed structure for ground truth???
if the framework does not do much on ground truth **retrieval**, then,
we will not impose much on the structure of ground truth!!!!!!!!!!!!!!!!!!!!!!!!!!

- for built-in evaluation methods, yes, for example, the `exact_match` evaluation method requires a ground truth answer.
  - other built-in evaluation methods??????????
    - TODO: can labeled results be supported? (exact match on every label value)
        - we must provide a built-in evaluation method for this
          - there's indeed something we can provide as reusable parsing logic

should ground truth be required in dataset items?

NOW, to make code consistent:
- search for "answer" in codebase to find its usage
- search for "ground_truth" in codebase to find its usage
