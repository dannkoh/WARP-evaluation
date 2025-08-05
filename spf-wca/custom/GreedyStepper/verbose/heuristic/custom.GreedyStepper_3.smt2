(declare-const in0 Int)
(declare-const in2 Int)
(declare-const in1 Int)

(assert (and (and (not ( = ( mod  in0 2) 0)) (not ( = ( mod  in1 2) 0))) (not ( = ( mod  in2 2) 0))))

(check-sat)
(get-model)