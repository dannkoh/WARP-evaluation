(declare-const t Int)
(declare-const in0 Int)
(declare-const in1 Int)

(assert (and (and (and (not ( = in0 t))  ( <  in0 t)) (not ( = in1 t)))  ( <  in1 t)))

(check-sat)
(get-model)