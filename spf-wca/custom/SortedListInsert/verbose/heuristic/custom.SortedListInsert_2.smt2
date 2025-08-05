(declare-const in Int)
(declare-const in0 Int)
(declare-const in1 Int)

(assert (and (and  ( >  in1 in0)  ( >  in in0))  ( >  in in1)))

(check-sat)
(get-model)