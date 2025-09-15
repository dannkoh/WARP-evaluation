(declare-const in0 Int)
(declare-const in1 Int)

(assert (and (and (and (not ( = in0 65))  ( =  in0 66)) (not ( = in1 69)))  ( =  in1 70)))

(check-sat)
(get-model)