(declare-const in0 Int)

(assert (and (not ( = ( mod  in0 2) 0))  ( =  ( mod  in0 3) 0)))

(check-sat)
(get-model)