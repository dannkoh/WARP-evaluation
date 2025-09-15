(declare-const t Int)
(declare-const in2 Int)
(declare-const in1 Int)

(assert (and (and (and (not ( = in1 t))  ( <  in1 t)) (not ( = in2 t)))  ( <  in2 t)))

(check-sat)
(get-model)