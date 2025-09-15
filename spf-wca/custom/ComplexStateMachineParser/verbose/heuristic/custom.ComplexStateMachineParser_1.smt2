(declare-const in0 Int)

(assert (and (not ( = in0 65))  ( =  in0 66)))

(check-sat)
(get-model)