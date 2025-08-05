(declare-const in5 Int)
(declare-const in0 Int)
(declare-const in2 Int)
(declare-const in1 Int)
(declare-const in4 Int)
(declare-const in3 Int)

(assert (and (and (and (and (and (not ( = ( mod  in0 2) 0)) (not ( = ( mod  in1 2) 0))) (not ( = ( mod  in2 2) 0))) (not ( = ( mod  in3 2) 0))) (not ( = ( mod  in4 2) 0))) (not ( = ( mod  in5 2) 0))))

(check-sat)
(get-model)