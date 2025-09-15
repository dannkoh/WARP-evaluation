(declare-const in5 Int)
(declare-const t Int)
(declare-const in2 Int)
(declare-const in4 Int)

(assert (and (and (and (and (and (not ( = in2 t))  ( <  in2 t)) (not ( = in4 t)))  ( <  in4 t)) (not ( = in5 t)))  ( <  in5 t)))

(check-sat)
(get-model)