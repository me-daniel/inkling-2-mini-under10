| check | Q8_0 reference | c1 | rule | result |
|---|---|---|---|---|
| lowest distinct 4-gram ratio over the long generations | 0.9894 | 0.9791 | no generation below 0.30 | pass |
| longest repeated-line run over the long generations | 1 | 1 | no generation above 5 | pass |
| planted fact found in the long prompt | True | True | found if the reference finds it | pass |
| valid tool calls | 5 of 5 | 5 of 5 | at most 1 fewer | pass |
| non-English replies that pass | 2 of 2 | 2 of 2 | all pass | pass |
| GSM8K correct | 30 of 30 | 30 of 30 | at most 4 below | pass |
| GSM8K replies that hit the cap | 0 | 0 | at most 2 more | pass |

verdict: PASS
