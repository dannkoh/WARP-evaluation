(declare-const in0 Int)
(declare-const in1 Int)

(assert (and (and (and (not ( = ( mod  in0 2) 0))  ( =  ( mod  in0 3) 0)) (not ( = ( mod  in1 2) 0)))  ( =  ( mod  in1 3) 0)))

(check-sat)
(get-model)