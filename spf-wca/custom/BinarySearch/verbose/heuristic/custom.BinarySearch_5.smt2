(declare-const t Int)
(declare-const in2 Int)
(declare-const in4 Int)
(declare-const in3 Int)

(assert (and (and (and (and (and (not ( = in2 t))  ( <  in2 t)) (not ( = in3 t)))  ( <  in3 t)) (not ( = in4 t)))  ( <  in4 t)))

(check-sat)
(get-model)