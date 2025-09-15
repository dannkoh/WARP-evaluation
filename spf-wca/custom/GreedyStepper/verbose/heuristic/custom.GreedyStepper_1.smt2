(declare-const in0 Int)

(assert (not ( = ( mod  in0 2) 0)))

(check-sat)
(get-model)